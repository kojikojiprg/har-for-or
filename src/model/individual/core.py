from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import (
    FeedForward,
    IndividualEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class IndividualTemporalTransformer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.encoder = IndividualTemporalEncoder(config)
        emb_npatchs = self.encoder.emb.npatchs
        self.decoder = IndividualTemporalDecoder(config, emb_npatchs)

    def forward(self, x, mask, bbox=None):
        z, mu, logvar, y = self.encoder(x, mask, bbox)
        fake_x, fake_bboxs, mu_prior, logvar_prior = self.decoder(z, y, mask)

        return fake_x, fake_bboxs, z, mu, logvar, mu_prior, logvar_prior, y


class IndividualTemporalEncoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.cls_token = nn.Parameter(
            torch.randn((1, 1, config.hidden_ndim)), requires_grad=True
        )
        self.cls_mask = torch.full((1, 1), False, dtype=torch.bool, requires_grad=False)

        self.emb = IndividualEmbedding(
            config.data_type,
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.add_position_patch,
            config.patch_size,
            config.img_size,
        )

        self.pe = RotaryEmbedding(config.hidden_ndim)

        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.norm = nn.LayerNorm((config.seq_len + 1, config.hidden_ndim))
        self.ff_mu = FeedForward(
            (config.seq_len + 1) * config.hidden_ndim, config.latent_ndim
        )
        self.ff_logvar = FeedForward(
            (config.seq_len + 1) * config.hidden_ndim, config.latent_ndim
        )

        self.ff_y = FeedForward(config.hidden_ndim, config.n_clusters)

    def forward(self, x, mask, bbox=None):
        # embedding
        b, seq_len = x.size()[:2]
        if bbox is not None:
            x_emb = self.emb(x[~mask], bbox[~mask])
        else:
            x_emb = self.emb(x[~mask])
        x = torch.full((b, seq_len, self.hidden_ndim), -1e10, dtype=torch.float32).to(
            next(self.parameters()).device
        )
        x[~mask] = x_emb

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
        y = F.sigmoid(y)
        y = self.gumbel_softmax_sampling(y, y.size(), 0.5)

        # q(z|x, y)
        x = x.view(b, seq_len_1 * hidden_ndim)
        mu = self.ff_mu(x)  # average
        logvar = self.ff_logvar(x)  # log(sigma^2)
        ep = torch.randn_like(logvar)
        z = mu + torch.exp(logvar / 2) * ep
        # z, mu, log_sig (b, latent_ndim)
        return z, mu, logvar, y

    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.rand(shape)
        return -torch.log(-torch.log(u + eps))

    def gumbel_softmax_sampling(self, pi, shape, tau, eps=1e-20):
        log_pi = torch.log(pi + eps)
        g = self.sample_gumbel(shape).to(next(self.parameters()).device)
        y = F.softmax((log_pi + g) / tau, dim=1)
        return y


class IndividualTemporalDecoder(nn.Module):
    def __init__(self, config: SimpleNamespace, emb_npatchs: int):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.latent_ndim = config.latent_ndim
        self.data_type = config.data_type
        self.emb_hidden_ndim = config.emb_hidden_ndim
        self.add_position_patch = config.add_position_patch
        self.img_size = config.img_size
        self.emb_npatchs = emb_npatchs

        # p(z|y)
        self.ff_mu = FeedForward(config.n_clusters, config.latent_ndim)
        self.ff_logvar = FeedForward(config.n_clusters, config.latent_ndim)

        # p(x|z)
        self.emb = FeedForward(config.latent_ndim, config.hidden_ndim)
        self.pe = RotaryEmbedding(config.hidden_ndim)
        self.ff_z = FeedForward(config.latent_ndim, config.seq_len * config.hidden_ndim)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.ff = FeedForward(config.hidden_ndim, emb_npatchs * config.emb_hidden_ndim)
        if config.add_position_patch:
            self.conv_bbox = nn.Conv2d(config.emb_hidden_ndim, 2, 1)
            self.act_bbox = nn.Sigmoid()

        if config.data_type == "keypoints":
            self.conv_kps = nn.Conv2d(config.emb_hidden_ndim, 2, 1)
            self.act = nn.Sigmoid()
        elif config.data_type == "images":
            size = config.patch_size[0] * config.patch_size[1]
            self.conv_imgs1 = nn.Conv2d(config.emb_hidden_ndim, size, 1)
            self.conv_imgs2 = nn.Conv3d(1, 5, 1)
            self.act = nn.Tanh()
        else:
            raise ValueError

    def forward(self, z, y, mask):
        # p(z|y)
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

        x = self.ff(x)
        x = x.view(b, self.seq_len, self.emb_npatchs, self.emb_hidden_ndim)
        # x (b, emb_hidden_dim, seq_len, emb_npatchs)
        if self.add_position_patch:
            fake_x, fake_bboxs = x[:, :, :-2], x[:, :, -2:]
            fake_bboxs = fake_bboxs.permute(0, 3, 1, 2)
            fake_bboxs = self.act_bbox(self.conv_bbox(fake_bboxs))
            fake_bboxs = fake_bboxs.permute(0, 2, 3, 1)
        else:
            fake_x = x
            fake_bboxs = None

        if self.data_type == "keypoints":
            fake_x = fake_x.permute(0, 3, 1, 2)  # (b, ndim, seq_len, npatchs)
            fake_x = self.act(self.conv_kps(fake_x))
            fake_x = fake_x.permute(0, 2, 3, 1)  # (b, seq_len, 17, 2)
            return fake_x, fake_bboxs, mu_prior, logvar_prior
        elif self.data_type == "images":
            fake_x = fake_x.permute(0, 3, 1, 2)  # (b, ndim, seq_len, npatchs)
            fake_x = self.conv_imgs1(fake_x)

            b, patch_sz, seq_len, np = fake_x.size()
            fake_x = fake_x.view(b, 1, patch_sz, seq_len, np)
            fake_x = self.act(self.conv_imgs2(fake_x))
            # (b, 5, patch_sz, seq_len, npatchs)

            fake_x = fake_x.permute(0, 3, 1, 4, 2).contiguous()
            # (b, seq_len, 5, patch_sz, img_size)
            h, w = self.img_size
            fake_x = fake_x.view(b, seq_len, 5, h, w)  # (b, seq_len, 5, h, w)
            return fake_x, fake_bboxs, mu_prior, logvar_prior
