from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import (
    IndividualEmbedding,
    SwiGLU,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class IndividualTemporalTransformer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.encoder = IndividualTemporalEncoder(config)
        emb_npatchs = self.encoder.emb.emb_vis.npatchs
        self.decoder = IndividualTemporalDecoder(config, emb_npatchs)

    def forward(self, x_vis, x_spc, mask):
        z, mu, logvar, y = self.encoder(x_vis, x_spc, mask)
        fake_x_vis, fake_x_spc, mu_prior, logvar_prior = self.decoder(z, y, mask)

        return fake_x_vis, fake_x_spc, z, mu, logvar, mu_prior, logvar_prior, y


class IndividualTemporalEncoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

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

        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=True)

        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.ff_y = nn.Linear(config.hidden_ndim, config.n_clusters)

        self.ff_mu = nn.Linear(
            (config.seq_len + 1) * config.hidden_ndim, config.latent_ndim
        )
        self.ff_logvar = nn.Linear(
            (config.seq_len + 1) * config.hidden_ndim, config.latent_ndim
        )

    def forward(self, x_vis, x_spc, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x_emb = self.emb(x_vis[~mask], x_spc[~mask])
        x = torch.full((b, seq_len, self.hidden_ndim), -1e10, dtype=torch.float32).to(
            next(self.parameters()).device
        )
        x[~mask] = x_emb
        # TODO: check x

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
        b, seq_len_1, hidden_ndim = x.size()

        # q(y|x)
        y = x[:, 0, :]  # cls token
        y = self.ff_y(y.view(b, hidden_ndim))
        y = F.softmax(y, dim=1)

        # q(z|x, y)
        x = x.view(b, seq_len_1 * hidden_ndim)
        mu = self.ff_mu(x)
        logvar = self.ff_logvar(x)
        ep = torch.randn_like(logvar)
        z = mu + torch.exp(logvar / 2) * ep
        # z, mu, log_sig (b, latent_ndim)
        return z, mu, logvar, y


class IndividualTemporalDecoder(nn.Module):
    def __init__(self, config: SimpleNamespace, vis_npatchs: int):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.latent_ndim = config.latent_ndim
        self.emb_hidden_ndim = config.emb_hidden_ndim
        self.img_size = config.img_size
        self.vis_npatchs = vis_npatchs

        # p(z|y)
        self.ff_mu = nn.Linear(config.n_clusters, config.latent_ndim)
        self.ff_logvar = nn.Linear(config.n_clusters, config.latent_ndim)

        # p(x|z)
        self.emb = SwiGLU(config.latent_ndim, config.hidden_ndim)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=True)
        self.ff_z = SwiGLU(config.latent_ndim, config.seq_len * config.hidden_ndim)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.ff = SwiGLU(config.hidden_ndim, config.hidden_ndim * 2)

        size = config.patch_size[0] * config.patch_size[1]
        self.lin_vis1 = SwiGLU(config.hidden_ndim, config.emb_hidden_ndim * vis_npatchs)
        self.lin_vis2 = SwiGLU(config.emb_hidden_ndim, size * 5)
        self.act_vis = nn.Tanh()

        self.lin_spc = SwiGLU(config.hidden_ndim, 17 * 2)
        self.act_spc = nn.Sigmoid()

    def forward(self, z, y, mask):
        # p(z|y)
        y = self.gumbel_softmax_sampling(y, y.size(), 1.0)
        mu_prior = self.ff_mu(y)
        logvar_prior = self.ff_logvar(y)

        b = z.size()[0]

        # embedding fake x
        x = z.repeat((1, self.seq_len)).contiguous()
        x = x.view(b * self.seq_len, self.latent_ndim)
        x = self.emb(x)
        x = x.view(b, self.seq_len, self.hidden_ndim)
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        # p(x|z)
        # z (b, latent_ndim)
        z = self.ff_z(z)  # (b, seq_len * hidden_ndim)
        z = z.view(b, self.seq_len, self.hidden_ndim)

        for layer in self.decoders:
            x = layer(x, z, mask)
        # x (b, seq_len, hidden_ndim)

        x = x.view(-1, self.hidden_ndim)
        x = self.ff(x)
        fake_x_vis, fake_x_spc = x[:, : self.hidden_ndim], x[:, self.hidden_ndim :]

        fake_x_vis = self.lin_vis1(fake_x_vis)
        fake_x_vis = fake_x_vis.view(
            b * self.seq_len * self.vis_npatchs, self.emb_hidden_ndim
        )
        fake_x_vis = self.act_vis(self.lin_vis2(fake_x_vis))
        h, w = self.img_size
        fake_x_vis = fake_x_vis.view(b, self.seq_len, 5, h, w)

        fake_x_spc = self.act_spc(self.lin_spc(fake_x_spc))
        fake_x_spc = fake_x_spc.view(b, self.seq_len, 17, 2)

        return fake_x_vis, fake_x_spc, mu_prior, logvar_prior

    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.rand(shape)
        return -torch.log(-torch.log(u + eps))

    def gumbel_softmax_sampling(self, pi, shape, tau, eps=1e-20):
        log_pi = torch.log(pi + eps)
        g = self.sample_gumbel(shape).to(next(self.parameters()).device)
        y = F.softmax((log_pi + g) / tau, dim=1)
        return y
