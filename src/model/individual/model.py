import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from .core import IndividualTemporalTransformer


class IndividualActivityRecognition(LightningModule):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.data_type = config.data_type
        self.seq_len = config.seq_len
        self.lr = config.lr
        self.add_position_patch = config.add_position_patch
        h, w = config.img_size
        checkpoint_dir = f"models/individual/{self.data_type}/"
        filename = f"individual_{self.data_type}_seq_len{config.seq_len}-stride{config.stride}-{h}x{w}_loss_min"
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.callbacks = [
            ModelCheckpoint(
                checkpoint_dir,
                filename=filename,
                monitor="l",
                mode="min",
                save_last=True,
            ),
        ]
        last_name = f"individual_{self.data_type}_seq_len{config.seq_len}-stride{config.stride}-{h}x{w}_last"
        self.callbacks[0].CHECKPOINT_NAME_LAST = last_name

        self.model = None

    def configure_model(self):
        config = self.config
        if self.model is not None:
            return
        self.model = IndividualTemporalTransformer(
            config.data_type,
            config.n_clusters,
            config.seq_len,
            config.hidden_ndim,
            config.latent_ndim,
            config.nheads,
            config.nlayers,
            config.emb_hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.dropout,
            config.emb_dropout,
            config.add_position_patch,
            config.patch_size,
            config.img_size,
        )

    @staticmethod
    def loss_kl_gaussian(m, logv, m_p, logv_p):
        return -0.5 * torch.sum(
            1
            + logv
            - logv_p
            - logv.exp() / logv_p.exp()
            - (m_p - m) ** 2 / logv_p.exp()
        )

    @staticmethod
    def loss_kl_clustering(q, p, eps=1e-20):
        return (q * (torch.log(q + eps) - torch.log(p + eps))).sum()

    def loss_func(
        self, x, fake_x, bboxs, fake_bboxs, mu, logvar, mu_prior, logvar_prior, y
    ):
        logs = {}
        mask = torch.any(torch.isnan(bboxs), dim=[2, 3])

        # reconstruct loss of x
        x = x[~mask]
        fake_x = fake_x[~mask]
        lrc_x = F.mse_loss(x, fake_x)
        lrc_x *= self.config.lrc_x
        logs["x"] = lrc_x

        # reconstruct loss of bbox
        if self.add_position_patch:
            bboxs = bboxs[~mask]
            fake_bboxs = fake_bboxs[~mask]
            lrc_bbox = F.mse_loss(bboxs, fake_bboxs)
            lrc_bbox *= self.config.lrc_bbox
            logs["b"] = lrc_bbox

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior)
        lg *= self.config.lg
        logs["g"] = lg

        # clustering loss
        y_prior = (torch.ones(y.size()) / y.size(1)).to(next(self.parameters()).device)
        lc = self.loss_kl_clustering(y, y_prior)
        lc *= self.config.lc
        logs["c"] = lc

        loss = lrc_x + lg + lc
        if self.add_position_patch:
            loss += lrc_bbox
        logs["l"] = loss

        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch, batch_idx):
        ids, x, bboxs = batch
        ids = ids[0]
        x = x[0]
        bboxs = bboxs[0]

        mask = torch.any(torch.isnan(bboxs), dim=[2, 3])
        if not self.add_position_patch:
            bboxs = None

        fake_x, fake_bboxs, z, mu, logvar, mu_prior, logvar_prior, y = self.model(
            x, mask, bboxs
        )
        loss = self.loss_func(
            x, fake_x, bboxs, fake_bboxs, mu, logvar, mu_prior, logvar_prior, y
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
