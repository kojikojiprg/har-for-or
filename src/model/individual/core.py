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
            seq_len, hidden_ndim, latent_ndim, nheads, nlayers, dropout
        )
        self.decoder = IndividualTemporalDecoder(
            data_type,
            seq_len,
            hidden_ndim,
            latent_ndim,
            nheads,
            nlayers,
            emb_hidden_ndim,
            emb_npatchs,
            dropout,
        )

    def forward(self, x, mask, bbox=None):
        b, seq_len = x.size()[:2]
        if bbox is not None:
            x_emb = self.emb(x[~mask], bbox[~mask])
        else:
            x_emb = self.emb(x[~mask])
        x = (
            torch.full((b, seq_len, self.hidden_ndim), -1e10, dtype=torch.float32)
            .contiguous()
            .to(next(self.parameters()).device)
        )
        # print(x.shape, x_emb.shape, mask.shape, x[mask].shape)
        x[~mask] = x_emb

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        z, mu, log_sig = self.encoder(x, mask)
        fake_x, fake_bboxs = self.decoder(x, z, mask)

        return fake_x, mu, log_sig, fake_bboxs


class IndividualTemporalEncoder(nn.Module):
    def __init__(
        self,
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

        self.norm = nn.LayerNorm((seq_len, hidden_ndim))
        self.ff_mu = FeedForward(seq_len * hidden_ndim, latent_ndim)
        self.ff_log_sig = FeedForward(seq_len * hidden_ndim, latent_ndim)

    def forward(self, x, mask):
        # x (b, seq_len, hidden_ndim)
        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len, hidden_ndim)

        b, seq_len, hidden_ndim = x.size()
        x = x.view(b, seq_len * hidden_ndim)
        mu = self.ff_mu(x)  # average
        log_sig = self.ff_log_sig(x)  # log(sigma^2)

        ep = torch.randn_like(log_sig)
        z = mu + torch.exp(log_sig / 2) * ep
        # z, mu, log_sig (b, latent_ndim)
        return z, mu, log_sig


class IndividualTemporalDecoder(nn.Module):
    def __init__(
        self,
        data_type: str,
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

    def forward(self, x, z, mask):
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
            return fake_x, fake_bboxs
        elif self.data_type == "images":
            fake_x = fake_x.permute(0, 3, 1, 2)  # (b, ndim, seq_len, npatchs)
            fake_x = self.conv_imgs1(fake_x)

            b, patch_sz, seq_len, np = fake_x.size()
            fake_x = fake_x.view(b, 1, patch_sz, seq_len, np)
            fake_x = self.act(self.conv_imgs2(fake_x))
            # (b, 5, patch_sz, seq_len, npatchs)

            fake_x = fake_x.permute(0, 3, 1, 4, 2)
            # (b, seq_len, 5, patch_sz, img_size)
            h, w = self.img_size
            fake_x = fake_x.view(b, seq_len, 5, h, w)  # (b, seq_len, 5, h, w)
            return fake_x, fake_bboxs
