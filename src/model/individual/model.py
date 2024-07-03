from types import SimpleNamespace

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .core import Px_z, Pz_y, Qy_x, Qz_xy


class IndividualActivityRecognition(LightningModule):
    def __init__(self, config: SimpleNamespace, is_pretrain: bool = False):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.lr = config.lr
        self.is_pretrain = is_pretrain
        self.Qy_x = None
        self.Qz_xy = None
        self.Pz_y = None
        self.Px_z = None

    def configure_model(self):
        if self.Qy_x is not None:
            return
        self.Qy_x = Qy_x(self.config)
        self.Qz_xy = Qz_xy(self.config)
        vis_npatchs = self.Qy_x.emb.emb_vis.npatchs
        self.Pz_y = Pz_y(self.config)
        self.Px_z = Px_z(self.config, vis_npatchs)

        if self.is_pretrain:
            self.Qy_x.qy_x.requires_grad_(False)
            self.Pz_y.requires_grad_(False)

    @staticmethod
    def loss_x_vis(x_vis, fake_x_vis, mask):
        return F.mse_loss(x_vis[~mask], fake_x_vis[~mask])

    @staticmethod
    def loss_x_spc(x_spc, fake_x_spc, mask):
        return F.mse_loss(x_spc[~mask], fake_x_spc[~mask])

    @staticmethod
    def loss_kl_clustering(q, p, eps=1e-10):
        lc = (q * (torch.log(q + eps) - torch.log(p + eps))).sum()
        return lc

    @staticmethod
    def loss_kl_gaussian(mu1, logv1, mu2, logv2, mask):
        # mu, log (b, seq_len, latent_ndim)
        # lg = -0.5 * torch.mean(
        #     1
        #     + logv1[~mask]
        #     - logv2[~mask]
        #     - logv1[~mask].exp() / logv2[~mask].exp()
        #     - (mu1[~mask] - mu2[~mask]) ** 2 / logv2[~mask].exp()
        # )
        # return lg
        # mu, log (b, latent_ndim)
        lg = -0.5 * torch.mean(
            1
            + logv1
            - logv2
            - logv1.exp() / logv2.exp()
            - (mu1 - mu2) ** 2 / logv2.exp()
        )
        return lg

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
        pi,
        mask,
    ):
        logs = {}

        # reconstruct loss of x
        lrc_x_vis = self.loss_x_vis(x_vis, fake_x_vis, mask)
        lrc_x_vis *= self.config.lrc_x_vis
        logs["vis"] = lrc_x_vis.item()

        # reconstruct loss of bbox
        lrc_x_spc = self.loss_x_spc(x_spc, fake_x_spc, mask)
        lrc_x_spc *= self.config.lrc_x_spc
        logs["spc"] = lrc_x_spc.item()

        # clustering loss
        y_prior = (torch.ones(pi.size()) / pi.size(1)).to(next(self.parameters()).device)
        lc = self.loss_kl_clustering(pi, y_prior)
        lc *= self.config.lc
        logs["c"] = lc.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior, mask)
        lg *= self.config.lg
        logs["g"] = lg.item()

        if self.is_pretrain:
            loss = lrc_x_vis + lrc_x_spc
        else:
            loss = lrc_x_vis + lrc_x_spc + lc + lg

        logs["l"] = loss.item()
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def forward(self, x_vis, x_spc, mask, stage):
        # inference
        pi, x_emb = self.Qy_x(x_vis, x_spc)
        if stage == "train":
            y = F.gumbel_softmax(torch.log(pi), self.config.tau, dim=1)
        else:
            y = pi
        z, mu, logvar = self.Qz_xy(x_emb, y, mask)

        # generate
        recon_x_vis, recon_x_spc = self.Px_z(z, mask)
        z_prior, mu_prior, logvar_prior = self.Pz_y(y)
        # fake_x_vis, fake_x_spc = self.Px_z(z_prior, mask)

        return recon_x_vis, recon_x_spc, mu, logvar, mu_prior, logvar_prior, pi

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis[0].detach()
        x_spc = x_spc[0].detach()
        mask = mask[0].detach()

        fake_x_vis, fake_x_spc, mu, logvar, mu_prior, logvar_prior, pi = self(
            x_vis, x_spc, mask, "train"
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
            pi,
            mask,
        )

        return loss

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        mask = mask.to(next(self.parameters()).device)
        if x_vis.ndim == 5:
            ids = ids[0]
            x_vis = x_vis[0]
            x_spc = x_spc[0]
            mask = mask[0]

        fake_x_vis, fake_x_spc, mu, logvar, mu_prior, logvar_prior, y = self(
            x_vis, x_spc, mask, "pred"
        )
        mse_x_vis = self.loss_x_vis(x_vis, fake_x_vis, mask)
        mse_x_spc = self.loss_x_spc(x_spc, fake_x_spc, mask)

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i][0],
                "id": ids[i].cpu().numpy().item(),
                # "x_vis": x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                # "fake_x_vis": fake_x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                "x_vis": x_vis[i].cpu().numpy(),
                "fake_x_vis": fake_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "fake_x_spc": fake_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "mu": mu[i].cpu().numpy(),
                "logvar": logvar[i].cpu().numpy(),
                "y": y[i].cpu().numpy(),
                "mask": mask[i].cpu().numpy(),
            }
            results.append(data)
        return results

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        return torch.optim.RAdam(self.parameters(), lr=self.lr)
