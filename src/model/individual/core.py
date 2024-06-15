from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import (
    MLP,
    IndividualEmbedding,
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

        self.ff_y = MLP(config.hidden_ndim, config.n_clusters)

        self.ff_mu = MLP((config.seq_len + 1) * config.hidden_ndim, config.latent_ndim)
        self.ff_logvar = MLP(
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
        self.ff_mu = MLP(config.n_clusters, config.latent_ndim)
        self.ff_logvar = MLP(config.n_clusters, config.latent_ndim)

        # p(x|z)
        self.emb = MLP(config.latent_ndim, config.hidden_ndim)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=True)
        self.ff_z = MLP(config.latent_ndim, config.seq_len * config.hidden_ndim)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.ff = MLP(config.hidden_ndim, config.hidden_ndim * 2)

        emb_hidden_ndim = config.emb_hidden_ndim
        self.lin_vis = MLP(config.hidden_ndim, config.hidden_ndim * vis_npatchs)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose3d(emb_hidden_ndim, emb_hidden_ndim // 2, (1, 6, 6)),
            nn.SiLU(),
            nn.ConvTranspose3d(emb_hidden_ndim // 2, emb_hidden_ndim // 4, (1, 12, 8)),
            nn.SiLU(),
            nn.ConvTranspose3d(emb_hidden_ndim // 4, 5, (1, 16, 12)),
            nn.Tanh(),
        )

        self.lin_spc = MLP(config.hidden_ndim, 17 * 2)
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

        # reconstruct
        x = x.view(-1, self.hidden_ndim)
        x = self.ff(x)
        fake_x_vis, fake_x_spc = x[:, : self.hidden_ndim], x[:, self.hidden_ndim :]

        # reconstruct x_vis
        fake_x_vis = self.lin_vis(fake_x_vis)
        fake_x_vis = fake_x_vis.view(
            b * self.seq_len, self.emb_hidden_ndim, self.vis_npatchs, 1, 1
        )
        fake_x_vis = self.conv_transpose(fake_x_vis)
        h, w = self.img_size
        fake_x_vis = fake_x_vis.view(b, self.seq_len, 5, h, w)

        # reconstruct x_spc
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
