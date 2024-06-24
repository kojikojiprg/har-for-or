from types import SimpleNamespace

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import (
    MLP,
    IndividualEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class Q(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.tau = config.tau

        self.cls_token = nn.Parameter(
            torch.randn((1, 1, config.hidden_ndim)), requires_grad=True
        )
        self.cls_mask = torch.full((1, 1), False, dtype=torch.bool, requires_grad=False)

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
            MLP(config.hidden_ndim, config.hidden_ndim),
            nn.LayerNorm(config.hidden_ndim),
            nn.Linear(config.hidden_ndim, config.n_clusters),
            nn.Softmax(dim=1),
        )

        self.ff_mu = nn.Sequential(
            MLP(config.hidden_ndim, config.latent_ndim),
            nn.LayerNorm((config.seq_len, config.latent_ndim)),
        )
        self.ff_logvar = nn.Sequential(
            MLP(config.hidden_ndim, config.latent_ndim),
            nn.LayerNorm((config.seq_len, config.latent_ndim)),
        )

    def forward(self, x_vis, x_spc, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc)
        x = x.view(b, seq_len, self.hidden_ndim)

        # concat cls token
        x = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim=1)
        cls_mask = self.cls_mask.repeat(b, 1).to(next(self.parameters()).device)
        mask = torch.cat([cls_mask, mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        # x (b, seq_len+1, hidden_ndim)
        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        # q(y|x)
        y = x[:, 0, :]  # cls token
        y = self.qy_x(y.view(b, self.hidden_ndim))

        # q(z|x, y)
        x = x[:, 1:, :]
        mu = self.ff_mu(x)
        logvar = self.ff_logvar(x)
        ep = torch.randn_like(logvar)
        z = mu + torch.exp(logvar / 2) * ep
        # z, mu, log_sig (b, seq_len, latent_ndim)
        return z, mu, logvar, y


class Pz_y(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()

        self.ff = MLP(1, config.seq_len)
        self.emb = nn.Sequential(
            MLP(config.n_clusters, config.latent_ndim),
            nn.LayerNorm((config.seq_len, config.latent_ndim)),
        )
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=False)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.ff_mu = nn.Sequential(
            MLP(config.latent_ndim, config.latent_ndim),
            nn.LayerNorm((config.seq_len, config.latent_ndim)),
        )
        self.ff_logvar = nn.Sequential(
            MLP(config.latent_ndim, config.latent_ndim),
            nn.LayerNorm((config.seq_len, config.latent_ndim)),
        )

    def forward(self, y, mask=None):
        b, n_clusters = y.size()
        y = self.ff(y.view(b, n_clusters, 1))
        y = y.permute(0, 2, 1)  # (b, seq_len, latent_ndim)
        y = self.emb(y)
        y = self.pe.rotate_queries_or_keys(y, seq_dim=1, offset=1)
        for layer in self.encoders:
            y, attn_w = layer(y, mask)
        mu_prior = self.ff_mu(y)
        logvar_prior = self.ff_logvar(y)

        ep = torch.randn_like(logvar_prior)
        z_prior = mu_prior + torch.exp(logvar_prior / 2) * ep

        return z_prior, mu_prior, logvar_prior


class Px_z(nn.Module):
    def __init__(self, config: SimpleNamespace, vis_npatchs: int):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.latent_ndim = config.latent_ndim
        self.emb_hidden_ndim = config.emb_hidden_ndim
        self.img_size = config.img_size
        self.tau = config.tau
        self.vis_npatchs = vis_npatchs

        self.emb = nn.Sequential(
            MLP(config.latent_ndim, config.hidden_ndim),
            nn.LayerNorm((config.seq_len, config.hidden_ndim)),
        )
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)
        self.ff_z = nn.Sequential(
            MLP(config.latent_ndim, config.hidden_ndim),
            nn.LayerNorm((config.seq_len, config.hidden_ndim)),
        )
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.ff = MLP(config.hidden_ndim, config.emb_hidden_ndim * 2)

        self.lin_vis = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.LayerNorm((config.seq_len, config.emb_hidden_ndim)),
            MLP(config.emb_hidden_ndim),
            nn.LayerNorm((config.seq_len, config.emb_hidden_ndim)),
            nn.Linear(config.emb_hidden_ndim, 17 * 2),
            nn.LayerNorm((config.seq_len, 17 * 2)),
            nn.Tanh(),
        )

        self.lin_spc = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.LayerNorm((config.seq_len, config.emb_hidden_ndim)),
            MLP(config.emb_hidden_ndim),
            nn.LayerNorm((config.seq_len, config.emb_hidden_ndim)),
            nn.Linear(config.emb_hidden_ndim, 2 * 2),
            nn.LayerNorm((config.seq_len, 2 * 2)),
            nn.Tanh(),
        )

    def forward(self, z, mask):
        b = z.size()[0]

        # embedding fake x
        fake_x = self.emb(z)
        fake_x = self.pe.rotate_queries_or_keys(fake_x, seq_dim=1, offset=1)

        z = self.ff_z(z)
        z = z.view(b, self.seq_len, self.hidden_ndim)
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1, offset=1)

        for layer in self.decoders:
            fake_x = layer(fake_x, z, mask)
        # fake_x (b, seq_len, hidden_ndim)

        # reconstruct
        fake_x = fake_x.view(b, self.seq_len, self.hidden_ndim)
        fake_x = self.ff(fake_x)
        fake_x_vis, fake_x_spc = (
            fake_x[:, :, : self.emb_hidden_ndim],
            fake_x[:, :, self.emb_hidden_ndim :],
        )
        # fake_x_vis, fake_x_spc (b * seq_len, emb_hidden_ndim)

        # reconstruct x_vis
        fake_x_vis = self.lin_vis(fake_x_vis)
        fake_x_vis = fake_x_vis.view(b, self.seq_len, 17, 2)

        # reconstruct x_spc
        fake_x_spc = self.lin_spc(fake_x_spc)
        fake_x_spc = fake_x_spc.view(b, self.seq_len, 2, 2)

        return fake_x_vis, fake_x_spc
