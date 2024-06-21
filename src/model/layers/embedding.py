import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from .feedforward import MLP, SwiGLU
from .transformer import TransformerEncoderBlock


class TransformerEmbedding(nn.Module):
    def __init__(self, in_ndim, out_ndim, nheads, nlayers, dropout):
        super().__init__()
        self.feature = nn.Parameter(torch.randn((1, 1, in_ndim)), requires_grad=True)

        self.pe = RotaryEmbedding(in_ndim + 1, learned_freq=False)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(in_ndim, nheads, dropout) for _ in range(nlayers)]
        )
        self.ff = SwiGLU(in_ndim, out_ndim)

    def forward(self, x):
        # x (b, npatch, in_ndim)
        b = x.size(0)

        x = torch.cat([self.feature.repeat(b, 1, 1), x], dim=1)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        for layer in self.layers:
            x = layer(x)[0]

        x = x[:, 0, :]
        x = self.ff(x)
        # x (b, out_ndim)
        return x


class PointEmbedding(nn.Module):
    # npatchs = 17
    npatchs = 2

    def __init__(self, hidden_ndim, out_ndim, nheads, nlayers, dropout):
        super().__init__()
        self.lin = MLP(2, hidden_ndim)
        self.transformer = TransformerEmbedding(
            hidden_ndim, out_ndim, nheads, nlayers, dropout
        )

    def forward(self, pt):
        # pt (b, npatchs, 2)
        pt = self.lin(pt)  # (b, npatchs, ndim)

        pt = self.transformer(pt)
        return pt  # (b, ndim)


class PixcelEmbedding(nn.Module):
    def __init__(
        self,
        hidden_ndim,
        out_ndim,
        nheads,
        nlayers,
        dropout,
        patch_size,
        img_size,
    ):
        super().__init__()
        self.hidden_ndim = hidden_ndim
        self.patch_size = patch_size
        self.npatchs = self.get_npatchs(img_size, patch_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, hidden_ndim // 4, 1, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_ndim // 4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_ndim // 4, hidden_ndim // 2, 5, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_ndim // 2),
            nn.MaxPool2d(5, 1, 0),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_ndim // 2, hidden_ndim, 5, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_ndim),
            nn.MaxPool2d(5, 1, 0),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_ndim, hidden_ndim, 3, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_ndim),
            nn.MaxPool2d(3, 1, 0),
        )
        self.flatten = nn.Linear(hidden_ndim * 12 * 4, hidden_ndim)

        self.transformer = TransformerEmbedding(
            hidden_ndim, out_ndim, nheads, nlayers, dropout
        )

    @staticmethod
    def get_npatchs(img_size, patch_size):
        return (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

    def patching(self, imgs):
        ph, pw = self.patch_size
        imgs = imgs.unfold(2, ph, ph).unfold(3, pw, pw).contiguous()
        n, c, nh, nw, ph, pw = imgs.size()
        return imgs.view(n, c, nh * nw, ph, pw)

    def forward(self, imgs):
        # imgs (b, 5, 256, 192)
        imgs = self.patching(imgs)
        # imgs (b, 5, nimgs, h, w)

        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous()
        b, nimgs, c, h, w = imgs.size()
        imgs = imgs.view(b * nimgs, c, h, w)
        imgs = self.conv1(imgs)
        imgs = self.conv2(imgs)
        imgs = self.conv3(imgs)
        imgs = self.conv4(imgs)
        imgs = imgs.view(b, nimgs, self.hidden_ndim * 12 * 4)
        imgs = self.flatten(imgs)
        # imgs (b, nimgs, ndim)

        imgs = self.transformer(imgs)

        return imgs  # (b, ndim)


class IndividualEmbedding(nn.Module):
    def __init__(
        self,
        hidden_ndim: int,
        out_ndim: int,
        nheads: int,
        nlayers: int,
        dropout: float,
        patch_size,
        img_size,
    ):
        super().__init__()
        self.emb_vis = PixcelEmbedding(
            hidden_ndim, out_ndim, nheads, nlayers, dropout, patch_size, img_size
        )
        self.emb_spc = PointEmbedding(
            hidden_ndim, out_ndim, nheads, nlayers, dropout
        )
        self.ff = MLP(out_ndim * 2, out_ndim)

    def forward(self, x_vis, x_spc):
        # f_vis (b, 5, h, w)
        # f_spc (b, npoints, 2)
        x_vis = self.emb_vis(x_vis)
        x_spc = self.emb_spc(x_spc)
        x = torch.cat([x_vis, x_spc], dim=1)
        x = self.ff(x)
        return x  # x (b, out_ndim)
