from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import (
    MLP,
    IndividualEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class VAE(LightningModule):
    def __init__(self, config: SimpleNamespace, n_batches: int = None):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.seq_len = config.seq_len
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters
        self.accumulate_grad_batches = config.accumulate_grad_batches
        self.tau = config.tau
        self.alpha = config.alpha
        self.n_batches = n_batches

        self.Qy_x = None
        self.Qz_xy = None
        self.Py = None
        self.Pz_y = None
        self.Px_z = None

    def configure_model(self):
        if self.Qy_x is not None:
            return
        self.Qy_x = Qy_x(self.config)
        self.Qz_xy = Qz_xy(self.config)
        vis_npatchs = self.Qy_x.emb.emb_vis.npatchs
        self.Py = Py(self.config)
        self.Pz_y = Pz_y(self.config)
        self.Px_z = Px_z(self.config, vis_npatchs)

        self.new_pi = (torch.ones(self.n_clusters) / self.n_clusters).to(self.device)

    @staticmethod
    def loss_x_vis(x_vis, fake_x_vis, mask):
        diff = torch.diff(x_vis[~mask], dim=0)
        fake_diff = torch.diff(fake_x_vis[~mask], dim=0)
        return F.mse_loss(x_vis[~mask], fake_x_vis[~mask]) + F.mse_loss(diff, fake_diff)

    @staticmethod
    def loss_x_spc(x_spc, fake_x_spc, mask):
        diff = torch.diff(x_spc[~mask], dim=0)
        fake_diff = torch.diff(fake_x_spc[~mask], dim=0)
        return F.mse_loss(x_spc[~mask], fake_x_spc[~mask]) + F.mse_loss(diff, fake_diff)

    def loss_kl_clustering(self, q, eps=1e-10):
        lc = (q * (torch.log(q + eps) - torch.log(self.Py.pi + eps))).mean()
        return lc

    @staticmethod
    def loss_kl_gaussian(mu1, logv1, mu2, logv2):
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
        y,
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
        y_prior = (torch.ones(y.size()) / y.size(1)).to(y.device)
        lc = self.loss_kl_clustering(y, y_prior)
        lc *= self.config.lc
        logs["c"] = lc.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior)
        lg *= self.config.lg
        logs["g"] = lg.item()

        loss = lrc_x_vis + lrc_x_spc + lc + lg

        # logs["l"] = loss.item()
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @staticmethod
    def log_normal(z, mu, logvar):
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + logvar + (z - mu) ** 2 / logvar.exp(), dim=-1
        )

    def label_prob(self, z):
        y = F.one_hot(torch.arange(self.n_clusters), self.n_clusters).to(
            self.device, torch.float32
        )
        z_prior, mu_prior, logvar_prior = self.Pz_y(y)

        b = z.size(0)
        z = (
            z.view(b, 1, self.latent_ndim)
            .repeat((1, self.n_clusters, 1))
            .to(self.device)
        )
        mu_prior = mu_prior.view(1, self.n_clusters, self.latent_ndim)
        logvar_prior = logvar_prior.view(1, self.n_clusters, self.latent_ndim)
        pdfs = self.Py.pi * self.log_normal(z, mu_prior, logvar_prior)
        pdfs = pdfs / pdfs.sum(dim=-1).view(-1, 1)

        return pdfs

    def update_py(self):
        # update py
        loss = (self.Py.pi - self.new_pi) * self.config.lr_py
        next(self.Py.parameters()).data = self.Py.pi - loss
        self.log("lpy", loss.mean().item(), prog_bar=True)

        # init new_pi
        self.new_pi = torch.zeros_like(self.new_pi)

    def update_pz_y(self, opt_pz_y, b):
        self.toggle_optimizer(opt_pz_y)

        for i in range(self.config.epochs_pz_y):
            y = np.random.choice(self.n_clusters, b, p=self.Py.pi.detach().cpu().numpy())
            y = torch.tensor(y, dtype=torch.long)
            y = F.one_hot(y, self.n_clusters).to(self.device, torch.float32)
            z, mu, logvar = self.Pz_y(y)

            mu = mu.repeat((1, self.n_clusters)).view(self.n_clusters, b, self.latent_ndim)
            norm = torch.linalg.norm((z - mu).permute(1, 0, 2), dim=2)
            pdfs = (1 + norm / self.alpha) ** (-(self.alpha + 1) / 2)
            pdfs = pdfs / pdfs.sum(dim=1).view(-1, 1)

            tij = pdfs**2 / pdfs.sum(dim=0)
            tij = tij / tij.sum(dim=-1).view(-1, 1)
            loss = (pdfs * (torch.log(pdfs) - torch.log(tij))).mean()
            self.manual_backward(loss)
            opt_pz_y.step()
            opt_pz_y.zero_grad(set_to_none=True)

        self.log("lpzy", loss.item(), prog_bar=True)
        self.untoggle_optimizer(opt_pz_y)

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        mask = mask[0]
        b = mask.size(0)

        opt_pz_y, opt = self.optimizers()

        # update p(y) and p(z|y)
        if batch_idx == 0:
            self.update_py()
            self.update_pz_y(opt_pz_y, b)

        # train VAE
        self.toggle_optimizer(opt)
        logits = self.Qy_x(x_vis, x_spc, mask)
        y = F.gumbel_softmax(logits, self.tau, hard=True, dim=1)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, y, mask)
        z_prior, mu_prior, logvar_prior = self.Pz_y(y)
        recon_x_vis, recon_x_spc = self.Px_z(x_vis, x_spc, z, mask)

        pi = F.softmax(logits, dim=1)
        loss = self.loss_func(
            x_vis,
            recon_x_vis,
            x_spc,
            recon_x_spc,
            mu,
            logvar,
            mu_prior,
            logvar_prior,
            pi,
            mask,
        )
        self.manual_backward(loss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.untoggle_optimizer(opt)

        # calc new_pi
        with torch.no_grad():
            label_prob = self.label_prob(z)
            self.new_pi = self.new_pi + label_prob.sum(dim=0) / (b * self.n_batches)

        del keys, ids, x_vis, x_spc, mask
        del z, mu, logvar, z_prior, mu_prior, logvar_prior, y

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

        y = self.Qy_x(x_vis, x_spc, mask)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, y, mask)
        recon_x_vis, recon_x_spc = self.Px_z(x_vis, x_spc, z, mask)
        mse_x_vis = self.loss_x_vis(x_vis, recon_x_vis, mask)
        mse_x_spc = self.loss_x_spc(x_spc, recon_x_spc, mask)

        results = []
        for i in range(len(keys)):
            # label_prob = self.clustering_prob(z[i], mu[i], logvar[i], mask[i])
            data = {
                "key": keys[i][0],
                "id": ids[i].cpu().numpy().item(),
                # "x_vis": x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                # "fake_x_vis": fake_x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                "x_vis": x_vis[i].cpu().numpy(),
                "fake_x_vis": recon_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "fake_x_spc": recon_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "mu": mu[i].cpu().numpy(),
                "logvar": logvar[i].cpu().numpy(),
                "label_prob": y[i].cpu().numpy(),
                "label": y[i].cpu().numpy().argmax().item(),
                "mask": mask[i].cpu().numpy(),
            }
            results.append(data)
        return results

    def configure_optimizers(self):
        opt_pz_y = torch.optim.Adam(self.parameters(), lr=self.config.lr_pz_y, maximize=True)
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return [opt_pz_y, opt], []


class Qy_x(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.cls = nn.Parameter(torch.randn((1, 1, self.hidden_ndim)))
        self.cls_mask = nn.Parameter(torch.full((1, 1), False), requires_grad=False)

        self.emb = IndividualEmbedding(
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.patch_size,
            config.img_size,
        )
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp = nn.Sequential(
            MLP(config.hidden_ndim, config.n_clusters),
            # nn.Softmax(dim=1),
        )

    def forward(self, x_vis, x_spc, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc)

        # concat y to x
        x = torch.cat([self.cls.repeat((b, 1, 1)), x], dim=1)
        mask = torch.cat([self.cls_mask.repeat((b, 1)), mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        # x (b, seq_len+1, hidden_ndim)
        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        logits = self.mlp(x[:, 0, :])
        # (b, n_clusters)
        return logits


class Qz_xy(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.cls_mask = nn.Parameter(torch.full((1, 1), False), requires_grad=False)

        self.emb = IndividualEmbedding(
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.patch_size,
            config.img_size,
        )
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)
        self.emb_y = MLP(config.n_clusters, config.hidden_ndim)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.lin_mu = nn.Linear(config.hidden_ndim, config.latent_ndim)
        self.lin_logvar = nn.Linear(config.hidden_ndim, config.latent_ndim)

    def forward(self, x_vis, x_spc, y, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc)
        y = self.emb_y(y).view(b, 1, self.hidden_ndim)

        # concat y to x
        x = torch.cat([y, x], dim=1)
        mask = torch.cat([self.cls_mask.repeat((b, 1)), mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        mu = self.lin_mu(x[:, 0, :])
        logvar = self.lin_logvar(x[:, 0, :])
        ep = torch.randn_like(logvar)
        z = mu + logvar.mul(0.5).exp_() * ep
        # z, mu, log_sig (b, latent_ndim)

        return z, mu, logvar


class Py(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        pi = torch.ones(config.n_clusters) / config.n_clusters
        self.pi = nn.Parameter(pi, requires_grad=False)


class Pz_y(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.mlp = MLP(config.n_clusters, config.hidden_ndim)
        self.lin_mu = nn.Linear(config.hidden_ndim, config.latent_ndim)
        self.lin_logvar = nn.Linear(config.hidden_ndim, config.latent_ndim)

    def forward(self, y):
        z = self.mlp(y)
        mu_prior = self.lin_mu(z)
        logvar_prior = self.lin_logvar(z)
        ep = torch.randn_like(logvar_prior)
        z_prior = mu_prior + logvar_prior.mul(0.5).exp_() * ep

        return z_prior, mu_prior, logvar_prior


class Px_z(nn.Module):
    def __init__(self, config: SimpleNamespace, vis_npatchs: int):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.emb_hidden_ndim = config.emb_hidden_ndim
        self.x_vis_start = nn.Parameter(
            torch.zeros((1, 1, 17, 2), dtype=torch.float32), requires_grad=False
        )
        self.x_spc_start = nn.Parameter(
            torch.zeros((1, 1, 2, 2), dtype=torch.float32), requires_grad=False
        )

        self.emb = IndividualEmbedding(
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.patch_size,
            config.img_size,
        )
        # self.emb_y = MLP(config.n_clusters, config.hidden_ndim)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)
        self.emb_z = MLP(config.latent_ndim, config.seq_len * config.hidden_ndim)
        self.encoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp = MLP(config.hidden_ndim, config.emb_hidden_ndim * 2)
        self.rec_vis = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 17 * 2),
            nn.Tanh(),
        )
        self.rec_spc = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 2 * 2),
            nn.Tanh(),
        )

    def forward(self, x_vis, x_spc, z, mask=None):
        b, seq_len = x_vis.size()[:2]
        x_vis = torch.cat([self.x_vis_start.repeat((b, 1, 1, 1)), x_vis], dim=1)
        x_vis = x_vis[:, :-1]
        x_spc = torch.cat([self.x_spc_start.repeat((b, 1, 1, 1)), x_spc], dim=1)
        x_spc = x_spc[:, :-1]
        x = self.emb(x_vis, x_spc)
        mask = torch.cat([torch.full((b, 1), False).to(mask.device), mask], dim=1)
        mask = mask[:, :-1]
        # x (b, seq_len, hidden_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        z = self.emb_z(z)
        z = z.view(b, self.seq_len, self.hidden_ndim)
        for layer in self.encoders:
            x = layer(x, z, mask)
        # x (b, seq_len, hidden_ndim)

        # reconstruct
        x = self.mlp(x)
        fake_x_vis, fake_x_spc = (
            x[:, :, : self.emb_hidden_ndim],
            x[:, :, self.emb_hidden_ndim :],
        )
        # fake_x_vis, fake_x_spc (b, seq_len, emb_hidden_ndim)

        # reconstruct x_vis
        fake_x_vis = self.rec_vis(fake_x_vis)
        fake_x_vis = fake_x_vis.view(b, seq_len, 17, 2)

        # reconstruct x_spc
        fake_x_spc = self.rec_spc(fake_x_spc)
        fake_x_spc = fake_x_spc.view(b, seq_len, 2, 2)

        return fake_x_vis, fake_x_spc
