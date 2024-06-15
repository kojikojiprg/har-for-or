from types import SimpleNamespace

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .core import IndividualTemporalTransformer


class IndividualActivityRecognition(LightningModule):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.lr = config.lr
        self.model = None

    def configure_model(self):
        if self.model is not None:
            return
        self.model = IndividualTemporalTransformer(self.config)

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
        self, x_vis, fake_x_vis, x_spc, fake_x_spc, mu, logvar, mu_prior, logvar_prior, y, mask
    ):
        logs = {}

        # reconstruct loss of x
        lrc_x_vis = F.mse_loss(x_vis[~mask], fake_x_vis[~mask])
        lrc_x_vis *= self.config.lrc_x_vis
        logs["x_vis"] = lrc_x_vis.item()

        # reconstruct loss of bbox
        lrc_x_spc = F.mse_loss(x_spc[~mask], fake_x_spc[~mask])
        lrc_x_spc *= self.config.lrc_x_spc
        logs["x_spc"] = lrc_x_spc.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior)
        lg *= self.config.lg
        logs["g"] = lg.item()

        # clustering loss
        y_prior = (torch.ones(y.size()) / y.size(1)).to(next(self.parameters()).device)
        lc = self.loss_kl_clustering(y, y_prior)
        lc *= self.config.lc
        logs["c"] = lc.item()

        loss = lrc_x_vis + lrc_x_spc + lg + lc
        logs["l"] = loss.item()

        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        mask = mask[0]

        fake_x_vis, fake_x_spc, z, mu, logvar, mu_prior, logvar_prior, y = self.model(
            x_vis, x_spc, mask
        )
        loss = self.loss_func(
            x_vis,
            fake_x_vis,
            x_spc,
            fake_x_spc,
            mu,
            logvar,
            mu_prior,
            logvar_prior,
            y,
            mask,
        )

        del batch, x_vis, x_spc, mask  # release memory
        del fake_x_vis, fake_x_spc, z, mu, logvar, mu_prior, logvar_prior, y
        torch.cuda.empty_cache()

        return loss

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, mask = batch

        fake_x_vis, fake_x_spc, z, mu, logvar, mu_prior, logvar_prior, y = self.model(
            x_vis, x_spc, mask
        )
        mse_x_vis = F.mse_loss(x_vis[~mask], fake_x_vis[~mask]).item()
        mse_x_spc = F.mse_loss(x_spc[~mask], fake_x_spc[~mask]).item()
        data = {
            "key": keys[0],
            "id": ids[0],
            "x_vis": x_vis[0].cpu().numpy(),
            "fake_x_vis": fake_x_vis[0].cpu().numpy(),
            "mse_x_vis": mse_x_vis,
            "x_spc": x_spc[0].cpu().numpy(),
            "fake_x_spc": fake_x_spc[0].cpu().numpy(),
            "mse_x_spc": mse_x_spc,
            "z": z[0].cpu().numpy(),
            "mu": mu[0].cpu().numpy(),
            "logvar": logvar[0].cpu().numpy(),
            "y": y[0].cpu().numpy(),
        }
        return data

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
