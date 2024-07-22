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


class VAEEM(LightningModule):
    def __init__(self, config: SimpleNamespace, n_batches: int = None):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.seq_len = config.seq_len
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters
        self.accumulate_grad_batches = config.accumulate_grad_batches
        self.update_prior_interval = config.update_prior_interval
        self.tau = config.tau
        self.alpha = config.alpha
        self.batch_size = config.batch_size
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

        self.z_all_samples = torch.empty((0, self.latent_ndim)).cpu()

    @staticmethod
    def loss_x_vis(x_vis, fake_x_vis, mask):
        return F.mse_loss(x_vis[~mask], fake_x_vis[~mask])

    @staticmethod
    def loss_x_spc(x_spc, fake_x_spc, mask):
        return F.mse_loss(x_spc[~mask], fake_x_spc[~mask])

    @staticmethod
    def loss_kl(q, p, eps=1e-10):
        kl = (q * (torch.log(q + eps) - torch.log(p + eps))).mean()
        return kl

    @staticmethod
    def loss_kl_gaussian(mu1, logv1, mu2, logv2, weights=None):
        # mu, log (b, latent_ndim)
        lg = -0.5 * (
            1
            + logv1
            - logv2
            - logv1.exp() / logv2.exp()
            - (mu1 - mu2) ** 2 / logv2.exp()
        )
        if weights is None:
            lg = lg.mean()
        else:
            lg = torch.sum(lg * weights) / weights.sum()
        return lg

    def loss_func(
        self,
        x_vis,
        fake_x_vis,
        x_spc,
        fake_x_spc,
        x_spc_diff,
        fake_x_spc_diff,
        mu,
        logvar,
        mu_prior,
        logvar_prior,
        z_prior,
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

        # reconstruct loss of bbox diff
        lrc_x_spc_diff = self.loss_x_spc(x_spc_diff, fake_x_spc_diff, mask)
        lrc_x_spc_diff *= self.config.lrc_x_spc
        logs["spcd"] = lrc_x_spc_diff.item()

        lrc = lrc_x_vis + lrc_x_spc + lrc_x_spc_diff

        # clustering loss
        lc = self.loss_kl(y, self.Py.pi)
        lc *= self.config.lc
        logs["c"] = lc.item()

        # Gaussian loss
        weights = (lrc_x_vis + lrc_x_spc + lrc_x_spc_diff).detach()
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior, weights)
        lg *= self.config.lg
        logs["g"] = lg.item()

        # discreate loss
        mu_prior = mu_prior.repeat((1, self.n_clusters)).view(
            self.n_clusters, self.batch_size, self.latent_ndim
        )
        norm = torch.linalg.norm((z_prior - mu_prior).permute(1, 0, 2), dim=2)
        pdfs = (1 + norm / self.alpha) ** (-(self.alpha + 1) / 2)
        pdfs = pdfs / pdfs.sum(dim=1).view(-1, 1)

        tij = pdfs**2 / pdfs.sum(dim=0)
        tij = tij / tij.sum(dim=-1).view(-1, 1)
        ld = (pdfs * (torch.log(pdfs) - torch.log(tij))).mean()
        ld *= 8
        logs["d"] = ld.item()

        loss = lrc + lc + lg - ld

        # logs["l"] = loss.item()
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def update_pz_y(self, opt_pz_y):
        b = 128
        self.toggle_optimizer(opt_pz_y)

        for i in range(self.config.epochs_pz_y):
            y = np.random.choice(
                self.n_clusters, b, p=self.Py.pi.detach().cpu().numpy()
            )
            y = torch.tensor(y, dtype=torch.long)
            y = F.one_hot(y, self.n_clusters).to(self.device, torch.float32)
            z, mu, logvar = self.Pz_y(y)

            mu = mu.repeat((1, self.n_clusters)).view(
                self.n_clusters, b, self.latent_ndim
            )
            norm = torch.linalg.norm((z - mu).permute(1, 0, 2), dim=2)
            pdfs = (1 + norm / self.alpha) ** (-(self.alpha + 1) / 2)
            pdfs = pdfs / pdfs.sum(dim=1).view(-1, 1)

            tij = pdfs**2 / pdfs.sum(dim=0)
            tij = tij / tij.sum(dim=-1).view(-1, 1)
            loss = -(pdfs * (torch.log(pdfs) - torch.log(tij))).mean()
            self.manual_backward(loss)
            opt_pz_y.step()
            opt_pz_y.zero_grad(set_to_none=True)

        self.log("lpzy", loss.item(), prog_bar=True)
        self.untoggle_optimizer(opt_pz_y)

    @staticmethod
    def log_normal(z, mu, logvar):
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + logvar + (z - mu) ** 2 / logvar.exp(), dim=-1
        )

    def calc_responsibility(self, z):
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
        # pdfs (b, n_clusters)

        return pdfs

    def next_prior_params(self):
        z = self.z_all_samples
        z = z.to(self.device)
        with torch.no_grad():
            r = self.calc_responsibility(z)

        r_sum = r.sum(dim=0).view(-1, 1)
        b = self.batch_size * self.n_batches
        r = r.view(b, self.n_clusters, 1)
        z = z.view(b, 1, self.latent_ndim)

        next_pi = r.mean(dim=0).view(self.n_clusters)
        next_mu = (r * z).sum(dim=0) / r_sum
        next_sig = (r * ((z - next_mu) ** 2)).sum(dim=0) / r_sum
        next_logvar = torch.log(next_sig) * 2

        # init buffer
        self.responsibilities = torch.empty((0, self.n_clusters)).cpu()
        self.z_all_samples = torch.empty((0, self.latent_ndim)).cpu()

        return (
            next_pi.to(self.device),
            next_mu.to(self.device),
            next_logvar.to(self.device),
        )

    def update_prior_distribution(self, opt_pz_y, next_pi, next_mu, next_logvar):
        # update pi
        loss = (self.Py.pi - next_pi) * self.config.lr_py
        next(self.Py.parameters()).data = self.Py.pi - loss
        self.log("lpy", loss.mean().item(), prog_bar=True)

        b = self.batch_size * self.n_batches
        self.toggle_optimizer(opt_pz_y)

        # update mu and logvar
        for i in range(self.config.epochs_pz_y):
            y = torch.arange(0, self.n_clusters)
            y = F.one_hot(y, self.n_clusters).to(self.device, torch.float32)

            z, mu, logvar = self.Pz_y(y)

            lem = self.loss_kl_gaussian(mu, logvar, next_mu, next_logvar)

            # discreate
            y = np.random.choice(
                self.n_clusters, b, p=self.Py.pi.detach().cpu().numpy()
            )
            y = torch.tensor(y, dtype=torch.long)
            mu = mu.repeat((1, self.n_clusters)).view(
                self.n_clusters, self.n_clusters, self.latent_ndim
            )
            norm = torch.linalg.norm((z - mu).permute(1, 0, 2), dim=2)
            pdfs = (1 + norm / self.alpha) ** (-(self.alpha + 1) / 2)
            pdfs = pdfs / pdfs.sum(dim=1).view(-1, 1)

            tij = pdfs**2 / pdfs.sum(dim=0)
            tij = tij / tij.sum(dim=-1).view(-1, 1)
            ld = (pdfs * (torch.log(pdfs) - torch.log(tij))).mean()

            loss = lem - ld
            self.manual_backward(loss)
            opt_pz_y.step()
            opt_pz_y.zero_grad(set_to_none=True)
            self.log("lpzy", loss.item(), prog_bar=True)
        self.untoggle_optimizer(opt_pz_y)

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, x_spc_diff, mask = batch
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        x_spc_diff = x_spc_diff[0]
        mask = mask[0]

        opt_pz_y, opt = self.optimizers()

        # if self.current_epoch == 0 and batch_idx == 0:
        #     self.update_pz_y(opt_pz_y)

        # if (self.current_epoch + 1) % self.update_prior_interval != 0:
        if True:
            # train VAE
            self.toggle_optimizer(opt)
            logits = self.Qy_x(x_vis, x_spc, x_spc_diff, mask)
            y = F.gumbel_softmax(logits, self.tau, dim=1)
            z, mu, logvar = self.Qz_xy(x_vis, x_spc, x_spc_diff, y, mask)
            z_prior, mu_prior, logvar_prior = self.Pz_y(y)
            recon_x_vis, recon_x_spc, recon_x_spc_diff = self.Px_z(
                x_vis, x_spc, x_spc_diff, z, mask
            )

            y = F.softmax(logits, dim=1)
            loss = self.loss_func(
                x_vis,
                recon_x_vis,
                x_spc,
                recon_x_spc,
                x_spc_diff,
                recon_x_spc_diff,
                mu,
                logvar,
                mu_prior,
                logvar_prior,
                z_prior,
                y,
                mask,
            )
            self.manual_backward(loss)
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            self.untoggle_optimizer(opt)

            del z, mu, logvar, z_prior, mu_prior, logvar_prior, y
        else:
            # update prior distribution
            # E step
            with torch.no_grad():
                logits = self.Qy_x(x_vis, x_spc, x_spc_diff, mask)
                y = F.softmax(logits, dim=1)
                z, mu, logvar = self.Qz_xy(x_vis, x_spc, x_spc_diff, y, mask)
            self.z_all_samples = torch.cat([self.z_all_samples, z.cpu()], dim=0)

            # M step (update p(y) and p(z|y))
            if batch_idx + 1 == self.n_batches:  # last batch?
                next_pi, next_mu, next_logvar = self.next_prior_params()
                self.update_prior_distribution(opt_pz_y, next_pi, next_mu, next_logvar)

        del keys, ids, x_vis, x_spc, x_spc_diff, mask

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, x_spc_diff, mask = batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        x_spc_diff = x_spc_diff.to(next(self.parameters()).device)
        mask = mask.to(next(self.parameters()).device)
        if x_vis.ndim == 5:
            ids = ids[0]
            x_vis = x_vis[0]
            x_spc = x_spc[0]
            x_spc_diff = x_spc_diff[0]
            mask = mask[0]

        logits = self.Qy_x(x_vis, x_spc, x_spc_diff, mask)
        y = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, x_spc_diff, y, mask)
        recon_x_vis, recon_x_spc, recon_x_spc_diff = self.Px_z(
            x_vis, x_spc, x_spc_diff, z, mask
        )
        mse_x_vis = self.loss_x_vis(x_vis, recon_x_vis, mask)
        mse_x_spc = self.loss_x_spc(x_spc, recon_x_spc, mask)
        mse_x_spc_diff = self.loss_x_spc(x_spc_diff, recon_x_spc_diff, mask)

        label_prob = self.calc_responsibility(z)

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
                "fake_x_spc_diff": recon_x_spc_diff[i].cpu().numpy(),
                "mse_x_spc_diff": mse_x_spc_diff.item(),
                "z": z[i].cpu().numpy(),
                "mu": mu[i].cpu().numpy(),
                "logvar": logvar[i].cpu().numpy(),
                "label_prob": label_prob[i].cpu().numpy(),
                "label": label_prob[i].cpu().numpy().argmax().item(),
                "y": y[i].cpu().numpy(),
                "y_argmax": y[i].cpu().numpy().argmax().item(),
                "mask": mask[i].cpu().numpy(),
            }
            results.append(data)
        return results

    def configure_optimizers(self):
        opt_pz_y = torch.optim.Adam(self.Pz_y.parameters(), lr=self.config.lr_pz_y)
        # params = (
        #     list(self.Qy_x.parameters())
        #     + list(self.Qz_xy.parameters())
        #     + list(self.Px_z.parameters())
        # )
        # opt = torch.optim.Adam(params, lr=self.config.lr)
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

    def forward(self, x_vis, x_spc, x_spc_diff, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc, x_spc_diff)

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

    def forward(self, x_vis, x_spc, x_spc_diff, y, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc, x_spc_diff)
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

        self.mlp = MLP(config.hidden_ndim, config.emb_hidden_ndim * 3)
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
        self.rec_spc_diff = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 2 * 2),
            nn.Tanh(),
        )

    def forward(self, x_vis, x_spc, x_spc_diff, z, mask=None):
        b, seq_len = x_vis.size()[:2]
        x_vis = torch.cat([self.x_vis_start.repeat((b, 1, 1, 1)), x_vis], dim=1)
        x_vis = x_vis[:, :-1]
        x_spc = torch.cat([self.x_spc_start.repeat((b, 1, 1, 1)), x_spc], dim=1)
        x_spc = x_spc[:, :-1]
        x_spc_diff = torch.cat(
            [self.x_spc_start.repeat((b, 1, 1, 1)), x_spc_diff], dim=1
        )
        x_spc_diff = x_spc_diff[:, :-1]
        x = self.emb(x_vis, x_spc, x_spc_diff)
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
        fake_x_vis, fake_x_spc, fake_x_spc_diff = (
            x[:, :, : self.emb_hidden_ndim],
            x[:, :, self.emb_hidden_ndim : self.emb_hidden_ndim * 2],
            x[:, :, self.emb_hidden_ndim * 2 :],
        )
        # fake_x_vis, fake_x_spc (b, seq_len, emb_hidden_ndim)

        # reconstruct x_vis
        fake_x_vis = self.rec_vis(fake_x_vis)
        fake_x_vis = fake_x_vis.view(b, seq_len, 17, 2)

        # reconstruct x_spc
        fake_x_spc = self.rec_spc(fake_x_spc)
        fake_x_spc = fake_x_spc.view(b, seq_len, 2, 2)

        # reconstruct x_spc
        fake_x_spc_diff = self.rec_spc_diff(fake_x_spc_diff)
        fake_x_spc_diff = fake_x_spc_diff.view(b, seq_len, 2, 2)

        return fake_x_vis, fake_x_spc, fake_x_spc_diff
