from types import SimpleNamespace

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .core import IndividualTemporalTransformer


class IndividualActivityRecognition(LightningModule):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.feature_type = config.feature_type
        self.seq_len = config.seq_len
        self.lr = config.lr
        self.add_position_patch = config.add_position_patch
        self.model = None

    def configure_model(self):
        config = self.config
        if self.model is not None:
            return
        self.model = IndividualTemporalTransformer(config)

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
        self, x, fake_x, bboxs, fake_bboxs, mu, logvar, mu_prior, logvar_prior, y, mask
    ):
        logs = {}

        # reconstruct loss of x
        lrc_x = F.mse_loss(x[~mask], fake_x[~mask])
        lrc_x *= self.config.lrc_x
        logs["x"] = lrc_x.item()

        # reconstruct loss of bbox
        if self.add_position_patch:
            lrc_bbox = F.mse_loss(bboxs[~mask], fake_bboxs[~mask])
            lrc_bbox *= self.config.lrc_bbox
            logs["b"] = lrc_bbox.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior)
        lg *= self.config.lg
        logs["g"] = lg.item()

        # clustering loss
        y_prior = (torch.ones(y.size()) / y.size(1)).to(next(self.parameters()).device)
        lc = self.loss_kl_clustering(y, y_prior)
        lc *= self.config.lc
        logs["c"] = lc.item()

        loss = lrc_x + lg + lc
        if self.add_position_patch:
            loss += lrc_bbox
        logs["l"] = loss.item()

        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch, batch_idx):
        keys, ids, x, bboxs, mask = batch
        x = x[0]
        bboxs = bboxs[0]
        mask = mask[0]

        if not self.add_position_patch:
            bboxs = None

        fake_x, fake_bboxs, z, mu, logvar, mu_prior, logvar_prior, y = self.model(
            x, mask, bboxs
        )
        loss = self.loss_func(
            x, fake_x, bboxs, fake_bboxs, mu, logvar, mu_prior, logvar_prior, y, mask
        )

        del batch, x, bboxs, mask  # release memory
        del fake_x, fake_bboxs, z, mu, logvar, mu_prior, logvar_prior, y
        torch.cuda.empty_cache()

        return loss

    def predict_step(self, batch):
        keys, ids, x, bboxs, mask = batch

        if not self.add_position_patch:
            bboxs = None

        fake_x, fake_bboxs, z, mu, logvar, mu_prior, logvar_prior, y = self.model(
            x, mask, bboxs
        )
        mse_x = F.mse_loss(x[~mask], fake_x[~mask]).item()
        mse_bbox = F.mse_loss(bboxs[~mask], fake_bboxs[~mask]).item()
        data = {
            "key": keys[0],
            "id": ids[0],
            "x": x[0].cpu().numpy(),
            "fake_x": fake_x[0].cpu().numpy(),
            "mse_x": mse_x,
            "bboxs": bboxs[0].cpu().numpy(),
            "fake_bboxs": fake_bboxs[0].cpu().numpy(),
            "mse_bboxs": mse_bbox,
            "z": z[0].cpu().numpy(),
            "mu": mu[0].cpu().numpy(),
            "logvar": logvar[0].cpu().numpy(),
            "y": y[0].cpu().numpy(),
        }
        return data

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
