import torch
import torch.nn as nn
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
        hidden_ndim: int,
        latent_ndim: int,
        nheads: int,
        nlayers: int,
        emb_hidden_ndim: int,
        emb_nheads: int,
        emb_nlayers: int,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        emb_add_position_patch: bool = True,
        **kwargs
    ):
        super().__init__()
        self.data_type = data_type
        self.emb_hidden_ndim = emb_hidden_ndim
        self.emb_add_position_patch = emb_add_position_patch

        self.emb = IndividualEmbedding(
            data_type,
            emb_hidden_ndim,
            hidden_ndim,
            emb_nheads,
            emb_nlayers,
            emb_dropout,
            emb_add_position_patch,
            **kwargs
        )
        self.emb_npatchs = self.emb.npatchs

        self.pe = RotaryEmbedding(hidden_ndim)

        self.encoder = IndividualTemporalEncoder(
            hidden_ndim, latent_ndim, nheads, nlayers, dropout
        )
        self.decoder = IndividualTemporalDecoder(
            hidden_ndim, latent_ndim, nheads, nlayers, dropout
        )

        self.ff = FeedForward(hidden_ndim, self.emb_npatchs * emb_hidden_ndim)

        if emb_add_position_patch:
            self.conv_bbox = nn.Conv2d(emb_hidden_ndim, 2, 1)

        if data_type == "keypoints":
            self.conv_kps = nn.Conv2d(emb_hidden_ndim, 2, 1)
        elif data_type == "images":
            patch_size = kwargs["patch_size"]
            size = patch_size[0] * patch_size[1]
            self.conv_imgs1 = nn.Conv2d(emb_hidden_ndim, size, 1)
            self.conv_imgs2 = nn.Conv3d(1, 5, 1)
        else:
            raise ValueError
        self.tanh = nn.Tanh()

    def forward(self, x, bbox=None):
        if self.data_type == "images":
            # x (b, seq_len, c, h, w)
            h, w = x.size()[3:]

        x = self.emb(x, bbox)
        x = self.pe.rotate_queries_or_keys(x)
        # x (b, seq_len, hidden_ndim)

        z, ave, log_dev = self.encoder(x)
        # z (b, seq_len, latent_ndim)

        x = self.decoder(x, z)
        # x (b, seq_len, hidden_ndim)

        b, seq_len = x.size()[:2]
        x = self.ff(x)
        x = x.view(b, seq_len, self.emb_npatchs, self.emb_hidden_ndim)
        # x (b, emb_hidden_dim, seq_len, emb_npatchs)
        if self.emb_add_position_patch:
            fake_x, fake_bbox = x[:, :, :-2], x[:, :, -2:]
            fake_bbox = fake_bbox.permute(0, 3, 1, 2)
            fake_bbox = self.tanh(self.conv_bbox(fake_bbox))
            fake_bbox = fake_bbox.permute(0, 2, 3, 1)
        else:
            fake_x = x
            fake_bbox = None

        if self.data_type == "keypoints":
            fake_x = fake_x.permute(0, 3, 1, 2)  # (b, ndim, seq_len, npatchs)
            fake_x = self.tanh(self.conv_kps(fake_x))
            fake_x = fake_x.permute(0, 2, 3, 1)  # (b, seq_len, 17, 2)

            return fake_x, z, ave, log_dev, fake_bbox
        elif self.data_type == "images":
            fake_x = fake_x.permute(0, 3, 1, 2)  # (b, ndim, seq_len, npatchs)
            fake_x = self.conv_imgs1(fake_x)

            b, patch_sz, seq_len, np = fake_x.size()
            fake_x = fake_x.view(b, 1, patch_sz, seq_len, np)
            fake_x = self.tanh(self.conv_imgs2(fake_x))  # (b, 5, patch_sz, seq_len, npatchs)

            fake_x = fake_x.permute(0, 3, 1, 4, 2)  # (b, seq_len, 5, patch_sz, img_size)
            fake_x = fake_x.view(b, seq_len, 5, h, w)  # (b, seq_len, 5, h, w)

        return fake_x, z, ave, log_dev, fake_bbox


class IndividualTemporalEncoder(nn.Module):
    def __init__(
        self,
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

        self.lr_ave = nn.Linear(hidden_ndim, latent_ndim)
        self.lr_dev = nn.Linear(hidden_ndim, latent_ndim)

    def forward(self, x):
        # x (b, seq_len, hidden_ndim)
        for layer in self.encoders:
            x = layer(x)
        # x (b, seq_len, hidden_ndim)

        ave = self.lr_ave(x)  # average
        log_dev = self.lr_dev(x)  # log(sigma^2)

        ep = torch.randn_like(ave)
        z = ave + torch.exp(log_dev / 2) * ep
        # z (b, seq_len, latent_ndim)
        return z, ave, log_dev


class IndividualTemporalDecoder(nn.Module):
    def __init__(
        self,
        hidden_ndim: int,
        latent_ndim: int,
        nheads: int,
        nlayers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff = FeedForward(latent_ndim, hidden_ndim)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_ndim, nheads, dropout)
                for _ in range(nlayers)
            ]
        )

    def forward(self, x, z):
        # x (b, seq_len, hidden_ndim)
        # z (b, seq_len, latent_ndim)
        z = self.ff(z)  # (b, seq_len, hidden_ndim)

        for layer in self.decoders:
            x = layer(x, z)
        # x (b, seq_len, hidden_ndim)

        return x
