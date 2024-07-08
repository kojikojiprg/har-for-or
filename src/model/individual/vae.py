from types import SimpleNamespace

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import MLP, IndividualEmbedding, TransformerEncoderBlock


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

        self.mlp = MLP(1, config.seq_len)
        self.emb = MLP(config.n_clusters, config.hidden_ndim)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)

        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
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

    def forward(self, y):
        b, n_clusters = y.size()
        y = self.mlp(y.view(b, n_clusters, 1))
        y = y.permute(0, 2, 1)  # (b, seq_len, hidden_ndim)
        y = self.emb(y)
        y = self.pe.rotate_queries_or_keys(y, seq_dim=1, offset=1)

        for layer in self.encoders:
            y, attn_w = layer(y)

        y = y.view(b, -1)
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
        self.img_size = config.img_size
        self.vis_npatchs = vis_npatchs

        self.emb = MLP(config.latent_ndim, config.hidden_ndim * config.seq_len)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)
        self.decoders = nn.ModuleList(
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

    def forward(self, z, mask=None):
        b = z.size()[0]

        # embedding fake x
        fake_x = self.emb(z)
        fake_x = fake_x.view(b, self.seq_len, self.hidden_ndim)
        fake_x = self.pe.rotate_queries_or_keys(fake_x, seq_dim=1, offset=1)

        for layer in self.decoders:
            fake_x, attn_w = layer(fake_x, mask, mask_type="tgt")
        # fake_x (b, seq_len, hidden_ndim)

        # reconstruct
        fake_x = self.mlp(fake_x)
        fake_x_vis, fake_x_spc = (
            fake_x[:, :, : self.emb_hidden_ndim],
            fake_x[:, :, self.emb_hidden_ndim :],
        )
        # fake_x_vis, fake_x_spc (b, seq_len, emb_hidden_ndim)

        # reconstruct x_vis
        fake_x_vis = self.rec_vis(fake_x_vis)
        fake_x_vis = fake_x_vis.view(b, self.seq_len, 17, 2)

        # reconstruct x_spc
        fake_x_spc = self.rec_spc(fake_x_spc)
        fake_x_spc = fake_x_spc.view(b, self.seq_len, 2, 2)

        return fake_x_vis, fake_x_spc
