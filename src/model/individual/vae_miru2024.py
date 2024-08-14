from types import SimpleNamespace

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

        self.kmeans = None
        self.mu_all = torch.empty((0, self.latent_ndim)).cpu()
        self.logvar_all = torch.empty((0, self.latent_ndim)).cpu()
        self.ids_all = []

    def configure_model(self):
        if self.Qy_x is not None:
            return
        self.Qy_x = Qy_x(self.config)
        self.Qz_xy = Qz_xy(self.config)
        vis_npatchs = self.Qy_x.emb.emb.npatchs
        self.Py = Py(self.config)
        self.Pz_y = Pz_y(self.config)
        self.Px_z = Px_z(self.config, vis_npatchs)

    def loss_x(self, x, fake_x, mask):
        b = x.size(0)
        mses = torch.empty((0,)).to(self.device)
        for i in range(b):
            mse = F.mse_loss(x[i][~mask[i]], fake_x[i][~mask[i]])
            mses = torch.cat([mses, mse.view(1, 1)])
        return mses.ravel()  # (b,)

    def loss_kl(self, q, p, weights=None, eps=1e-10):
        kl = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)
        if weights is None:
            kl = kl.mean()
        else:
            kl = torch.sum(kl * weights) / weights.sum()
        return kl

    def loss_kl_gaussian(self, mu1, logv1, mu2, logv2, weights=None):
        # mu, log (b, latent_ndim)
        lg = -0.5 * (
            1
            + logv1
            - logv2
            - logv1.exp() / logv2.exp()
            - (mu1 - mu2) ** 2 / logv2.exp()
        ).sum(dim=-1)
        if weights is None:
            lg = lg.mean()
        else:
            lg = torch.sum(lg * weights) / weights.sum()
        return lg

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, x_spc_diff, mask = batch
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        x_spc_diff = x_spc_diff[0]
        mask = mask[0]

        opt = self.optimizers()

        self.toggle_optimizer(opt)
        # VAE
        logits = self.Qy_x(x_vis, x_spc, x_spc_diff, mask)
        y = F.gumbel_softmax(logits, self.tau, dim=1)
        pi = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, x_spc_diff, y, mask)
        z_prior, mu_prior, logvar_prior = self.Pz_y(y)
        recon_x_vis, recon_x_spc, recon_x_spc_diff = self.Px_z(
            x_vis, x_spc, x_spc_diff, z, mask
        )

        # loss
        logs = {}

        # ELBO
        # reconstruct loss of vis
        lrc_x_vis = self.loss_x(x_vis, recon_x_vis, mask)
        lrc_x_vis *= self.config.lrc_x_vis
        logs["vis"] = lrc_x_vis.mean().item()

        # reconstruct loss of spc
        lrc_x_spc = self.loss_x(x_spc, recon_x_spc, mask)
        lrc_x_spc *= self.config.lrc_x_spc
        logs["spc"] = lrc_x_spc.mean().item()

        # reconstruct loss of spc diff
        lrc_x_spc_diff = self.loss_x(x_spc_diff, recon_x_spc_diff, mask)
        lrc_x_spc_diff *= self.config.lrc_x_spc
        logs["spcd"] = lrc_x_spc_diff.mean().item()

        lrc = lrc_x_vis + lrc_x_spc + lrc_x_spc_diff
        lrc = lrc.mean()

        # clustering loss
        lc = self.loss_kl(pi, self.Py.pi)
        lc *= self.config.lc
        logs["c"] = lc.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior)
        lg = lg
        lg *= self.config.lg
        logs["g"] = lg.item()

        loss_elbo = lrc + lc + lg

        loss = loss_elbo
        # logs["l"] = loss.item()
        self.manual_backward(loss)
        self.log_dict(logs, prog_bar=True)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.untoggle_optimizer(opt)

        del z, mu, logvar, z_prior, mu_prior, logvar_prior, y
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
        mse_x_vis = self.loss_x(x_vis, recon_x_vis, mask)
        mse_x_spc = self.loss_x(x_spc, recon_x_spc, mask)
        mse_x_spc_diff = self.loss_x(x_spc_diff, recon_x_spc_diff, mask)

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i],
                "id": ids[i].cpu().numpy().item(),
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
                "label_prob": y[i].cpu().numpy(),
                "label": y[i].cpu().numpy().argmax().item(),
                "mask": mask[i].cpu().numpy(),
            }
            results.append(data)
        return results

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return [opt], []


class Qy_x(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.cls = nn.Parameter(torch.randn((1, 1, self.hidden_ndim)))
        self.cls_mask = nn.Parameter(torch.full((1, 1), True), requires_grad=False)

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
        z = mu + logvar.mul(0.5).exp() * ep
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
        z_prior = mu_prior + logvar_prior.mul(0.5).exp() * ep

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
        self.emb_z = MLP(config.latent_ndim, config.hidden_ndim)
        self.decoders = nn.ModuleList(
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
        z = z.repeat((1, seq_len)).view(b, seq_len, self.hidden_ndim)
        z = z.view(b, self.seq_len, self.hidden_ndim)
        for layer in self.decoders:
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
