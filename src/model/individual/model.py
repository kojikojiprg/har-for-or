from types import SimpleNamespace

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .core import Px_z, Pz_y, Q


class IndividualActivityRecognition(LightningModule):
    def __init__(self, config: SimpleNamespace, is_pretrain: bool = False):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.lr = config.lr
        self.is_pretrain = is_pretrain
        self.Q = None
        self.Pz_y = None
        self.Px_z = None

    def configure_model(self):
        if self.Q is not None:
            return
        self.Q = Q(self.config)
        vis_npatchs = self.Q.emb.emb_vis.npatchs
        self.Pz_y = Pz_y(self.config)
        self.Px_z = Px_z(self.config, vis_npatchs)

        if self.is_pretrain:
            self.Q.qy_x.requires_grad_(False)
            self.Pz_y.requires_grad_(False)

    def forward(self, x_vis, x_spc, mask):
        z, mu, logvar, y = self.Q(x_vis, x_spc, mask)

        resampled_y = F.gumbel_softmax(y, self.config.tau)

        z_prior, mu_prior, logvar_prior = self.Pz_y(resampled_y)
        fake_x_vis, fake_x_spc = self.Px_z(z, mask)

        return fake_x_vis, fake_x_spc, mu, logvar, mu_prior, logvar_prior, y

    @staticmethod
    def loss_x_vis(x_vis, fake_x_vis, mask):
        b, seq_len = x_vis.size()[:2]
        lrc_x_vis = F.mse_loss(x_vis[~mask], fake_x_vis[~mask], reduction="none")
        if x_vis.ndim == 5:
            lrc_x_vis = lrc_x_vis.mean(dim=[1, 2, 3]).sum()
        elif x_vis.ndim == 4:
            lrc_x_vis = lrc_x_vis.mean(dim=[1, 2]).sum()
        else:
            raise ValueError
        return lrc_x_vis / (b * seq_len)

    @staticmethod
    def loss_x_spc(x_spc, fake_x_spc, mask):
        b, seq_len = x_spc.size()[:2]
        lrc_x_spc = F.mse_loss(x_spc[~mask], fake_x_spc[~mask], reduction="none")
        lrc_x_spc = lrc_x_spc.mean(dim=[1, 2]).sum()
        return lrc_x_spc / (b * seq_len)

    @staticmethod
    def loss_kl_gaussian(mu1, logv1, mu2, logv2, mask):
        b, seq_len, latent_ndim = mu1.size()
        # mu, log (b, seq_len, latent_ndim)
        lg = -0.5 * torch.sum(
            1
            + logv1[~mask]
            - logv2[~mask]
            - logv1[~mask].exp() / logv2[~mask].exp()
            - (mu2[~mask] - mu1[~mask]) ** 2 / logv2[~mask].exp()
        )
        return lg / (b * seq_len * latent_ndim)

    @staticmethod
    def loss_kl_clustering(q, p, eps=1e-20):
        lc = (q * (torch.log(q + eps) - torch.log(p + eps))).sum()
        return lc

    def loss_func(
        self,
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
    ):
        logs = {}

        # reconstruct loss of x
        lrc_x_vis = self.loss_x_vis(x_vis, fake_x_vis, mask)
        lrc_x_vis *= self.config.lrc_x_vis
        logs["x_vis"] = lrc_x_vis.item()

        # reconstruct loss of bbox
        lrc_x_spc = self.loss_x_spc(x_spc, fake_x_spc, mask)
        lrc_x_spc *= self.config.lrc_x_spc
        logs["x_spc"] = lrc_x_spc.item()

        if self.is_pretrain:
            loss = lrc_x_vis + lrc_x_spc
            logs["l"] = loss.item()

            self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior, mask)
        lg *= self.config.lg
        logs["g"] = lg.item()

        # clustering loss
        y_prior = (torch.ones(y.size()) / y.size(1)).to(next(self.parameters()).device)
        lc = self.loss_kl_clustering(y, y_prior)
        lc *= self.config.lc
        logs["c"] = lc.item()

        loss = lrc_x_vis + lrc_x_spc + lg + lc
        logs["l"] = loss.item()

        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis[0].detach()
        x_spc = x_spc[0].detach()
        mask = mask[0].detach()

        fake_x_vis, fake_x_spc, mu, logvar, mu_prior, logvar_prior, y = self(
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
        del fake_x_vis, fake_x_spc, mu, logvar, mu_prior, logvar_prior, y
        torch.cuda.empty_cache()

        return loss

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        mask = mask.to(next(self.parameters()).device)

        fake_x_vis, fake_x_spc, mu, logvar, mu_prior, logvar_prior, y = self(
            x_vis, x_spc, mask
        )
        mse_x_vis = self.loss_x_vis(x_vis, fake_x_vis, mask)
        mse_x_spc = self.loss_x_spc(x_spc, fake_x_spc, mask)
        data = {
            "key": keys[0],
            "id": ids[0].cpu().numpy().item(),
            # "x_vis": x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
            # "fake_x_vis": fake_x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
            "x_vis": x_vis[0].cpu().numpy(),
            "fake_x_vis": fake_x_vis[0].cpu().numpy(),
            "mse_x_vis": mse_x_vis.item(),
            "x_spc": x_spc[0].cpu().numpy(),
            "fake_x_spc": fake_x_spc[0].cpu().numpy(),
            "mse_x_spc": mse_x_spc.item(),
            "mu": mu[0].cpu().numpy(),
            "logvar": logvar[0].cpu().numpy(),
            "y": y[0].cpu().numpy(),
            "mask": mask[0].cpu().numpy(),
        }
        return data

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
