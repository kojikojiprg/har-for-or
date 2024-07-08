from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import MLP, IndividualEmbedding, TransformerEncoderBlock


class VAE(LightningModule):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.lr = config.lr
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

    def forward(self, x_vis, x_spc, mask, stage):
        # q
        z, mu, logvar, pi = self.Q(x_vis, x_spc, mask)
        if stage == "train":
            y = F.gumbel_softmax(torch.log(pi), self.config.tau, dim=1)
        else:
            y = pi

        # p(z|y)
        z_prior, mu_prior, logvar_prior = self.Pz_y(y)

        # p(x|z)
        recon_x_vis, recon_x_spc = self.Px_z(x_vis, x_spc, z, mask)

        return recon_x_vis, recon_x_spc, mu, logvar, mu_prior, logvar_prior, pi

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
        weight = torch.sqrt((mu1.detach() - mu2.detach()) ** 2).mean(dim=1)
        lg = -0.5 * (
            1
            + logv1
            - logv2
            - logv1.exp() / logv2.exp()
            - (mu1 - mu2) ** 2 / logv2.exp()
        )
        lg = lg * weight.view(-1, 1)
        lg = torch.mean(lg)
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
        pi_prior = (torch.ones(pi.size()) / pi.size(1)).to(
            next(self.parameters()).device
        )
        lc = self.loss_kl_clustering(pi, pi_prior)
        lc *= self.config.lc
        logs["c"] = lc.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior, mask)
        lg *= self.config.lg
        logs["g"] = lg.item()

        loss = lrc_x_vis + lrc_x_spc + lc + lg

        logs["l"] = loss.item()
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis[0].detach()
        x_spc = x_spc[0].detach()
        mask = mask[0].detach()

        recon_x_vis, recon_x_spc, mu, logvar, mu_prior, logvar_prior, pi = self(
            x_vis, x_spc, mask, "train"
        )
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
                "c": y[i].cpu().numpy().argmax().item(),
                "mask": mask[i].cpu().numpy(),
            }
            results.append(data)
        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # return torch.optim.RAdam(self.parameters(), lr=self.lr)


class Q(nn.Module):
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

        self.qy_x = nn.Sequential(
            nn.LayerNorm(config.hidden_ndim),
            MLP(config.hidden_ndim, config.hidden_ndim),
            nn.Linear(config.hidden_ndim, config.n_clusters),
            nn.Softmax(dim=1),
        )

        self.norm = nn.LayerNorm(config.hidden_ndim * config.seq_len)
        self.mlp_mu = nn.Sequential(
            MLP(config.hidden_ndim * config.seq_len, config.latent_ndim),
            nn.Linear(config.latent_ndim, config.latent_ndim),
        )
        self.mlp_logvar = nn.Sequential(
            MLP(config.hidden_ndim * config.seq_len, config.latent_ndim),
            nn.Linear(config.latent_ndim, config.latent_ndim),
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

        y = x[:, 0, :]
        pi = self.qy_x(y)

        x = x[:, 1:, :]
        x = x.view(b, -1)
        x = self.norm(x)
        mu = self.mlp_mu(x)
        logvar = self.mlp_logvar(x)
        ep = torch.randn_like(logvar)
        z = mu + torch.exp(logvar / 2) * ep
        # z, mu, log_sig (b, seq_len, latent_ndim)

        return z, mu, logvar, pi


class Pz_y(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.mlp = MLP(config.n_clusters, config.hidden_ndim)
        self.norm = nn.LayerNorm(config.hidden_ndim)
        self.mlp_mu = nn.Sequential(
            MLP(config.hidden_ndim, config.latent_ndim),
            nn.Linear(config.latent_ndim, config.latent_ndim),
        )
        self.mlp_logvar = nn.Sequential(
            MLP(config.hidden_ndim, config.latent_ndim),
            nn.Linear(config.latent_ndim, config.latent_ndim),
        )

    def forward(self, y):
        y = self.mlp(y)
        y = self.norm(y)
        mu_prior = self.mlp_mu(y)
        logvar_prior = self.mlp_logvar(y)
        ep = torch.randn_like(logvar_prior)
        z_prior = mu_prior + torch.exp(logvar_prior / 2) * ep

        return z_prior, mu_prior, logvar_prior


class Px_z(nn.Module):
    def __init__(self, config: SimpleNamespace, vis_npatchs: int):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.emb_hidden_ndim = config.emb_hidden_ndim

        self.emb = IndividualEmbedding(
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.patch_size,
            config.img_size,
        )

        self.emb_z = MLP(config.latent_ndim, config.hidden_ndim)

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
            nn.LayerNorm(config.hidden_ndim),
            MLP(config.hidden_ndim, config.emb_hidden_ndim * 2),
        )

        self.rec_vis = nn.Sequential(
            nn.LayerNorm(config.emb_hidden_ndim),
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 17 * 2),
            nn.Tanh(),
        )

        self.rec_spc = nn.Sequential(
            nn.LayerNorm(config.emb_hidden_ndim),
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 2 * 2),
            nn.Tanh(),
        )

    def forward(self, x_vis, x_spc, z, mask=None):
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc)
        z = self.emb_z(z).view(b, 1, self.hidden_ndim)
        x = torch.cat([z, x], dim=1)
        mask = torch.cat([torch.full((b, 1), False).to(mask.device), mask], dim=1)
        # x (b, seq_len + 1, hidden_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)
        for layer in self.encoders:
            x, attn_w = layer(x, mask, mask_type="tgt")
        # x (b, seq_len, hidden_ndim)

        # reconstruct
        x = self.mlp(x[:, :-1, :])
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
