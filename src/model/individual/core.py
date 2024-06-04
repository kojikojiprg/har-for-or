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
    def __init__(
        self,
        data_type: str,
        n_clusters: int,
        seq_len: int,
        hidden_ndim: int,
        latent_ndim: int,
        nheads: int,
        nlayers: int,
        emb_hidden_ndim: int,
        emb_nheads: int,
        emb_nlayers: int,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        add_position_patch: bool = True,
        patch_size=(16, 12),
        img_size=(256, 192),
    ):
        super().__init__()
        self.hidden_ndim = hidden_ndim
        self.cls_token = nn.Parameter(
            torch.randn((1, 1, hidden_ndim)), requires_grad=True
        )
        self.cls_mask = torch.full((1, 1), False, dtype=torch.bool, requires_grad=False)

        self.emb = IndividualEmbedding(
            data_type,
            emb_hidden_ndim,
            hidden_ndim,
            emb_nheads,
            emb_nlayers,
            emb_dropout,
            add_position_patch,
            patch_size,
            img_size,
        )
        emb_npatchs = self.emb.npatchs

        self.pe = RotaryEmbedding(hidden_ndim)

        self.encoder = IndividualTemporalEncoder(
            n_clusters, seq_len, hidden_ndim, latent_ndim, nheads, nlayers, dropout
        )
        self.decoder = IndividualTemporalDecoder(
            data_type,
            n_clusters,
            seq_len,
            hidden_ndim,
            latent_ndim,
            nheads,
            nlayers,
            emb_hidden_ndim,
            emb_npatchs,
            dropout,
            add_position_patch,
            patch_size,
            img_size,
        )

    def forward(self, x, mask, bbox=None):
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

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        z, mu, logvar, y = self.encoder(x, mask)
        fake_x, fake_bboxs, mu_prior, logvar_prior = self.decoder(
            x[:, 1:, :], z, y, mask[:, 1:]
        )

        return fake_x, fake_bboxs, z, mu, logvar, mu_prior, logvar_prior, y


class IndividualTemporalEncoder(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        seq_len: int,
        hidden_ndim: int,
        latent_ndim: int,
        nheads: int,
        nlayers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(hidden_ndim, nheads, dropout)
                for _ in range(nlayers)
            ]
        )

        self.norm = nn.LayerNorm((seq_len + 1, hidden_ndim))
        self.ff_mu = FeedForward((seq_len + 1) * hidden_ndim, latent_ndim)
        self.ff_logvar = FeedForward((seq_len + 1) * hidden_ndim, latent_ndim)

        self.ff_y = FeedForward(hidden_ndim, n_clusters)

    def forward(self, x, mask):
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
    def __init__(
        self,
        data_type: str,
        n_clusters: int,
        seq_len: int,
        hidden_ndim: int,
        latent_ndim: int,
        nheads: int,
        nlayers: int,
        emb_hidden_ndim: int,
        emb_npatchs: int,
        dropout: float = 0.1,
        add_position_patch: bool = True,
        patch_size: tuple = (16, 12),
        img_size: tuple = (256, 192),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_ndim = hidden_ndim
        self.data_type = data_type
        self.emb_hidden_ndim = emb_hidden_ndim
        self.emb_npatchs = emb_npatchs
        self.add_position_patch = add_position_patch
        self.img_size = img_size

        # p(z|y)
        self.ff_mu = FeedForward(n_clusters, latent_ndim)
        self.ff_logvar = FeedForward(n_clusters, latent_ndim)

        # p(x|z)
        self.ff_z = FeedForward(latent_ndim, seq_len * hidden_ndim)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_ndim, nheads, dropout)
                for _ in range(nlayers)
            ]
        )

        self.ff = FeedForward(hidden_ndim, emb_npatchs * emb_hidden_ndim)
        if add_position_patch:
            self.conv_bbox = nn.Conv2d(emb_hidden_ndim, 2, 1)
            self.act_bbox = nn.Sigmoid()

        if data_type == "keypoints":
            self.conv_kps = nn.Conv2d(emb_hidden_ndim, 2, 1)
            self.act = nn.Sigmoid()
        elif data_type == "images":
            size = patch_size[0] * patch_size[1]
            self.conv_imgs1 = nn.Conv2d(emb_hidden_ndim, size, 1)
            self.conv_imgs2 = nn.Conv3d(1, 5, 1)
            self.act = nn.Tanh()
        else:
            raise ValueError

    def forward(self, x, z, y, mask):
        # p(z|y)
        mu_prior = self.ff_mu(y)
        logvar_prior = self.ff_logvar(y)

        # p(x|z)
        # x (b, seq_len, hidden_ndim)
        # z (b, latent_ndim)
        z = self.ff_z(z)  # (b, seq_len * hidden_ndim)
        b = z.size()[0]
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
