import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from .core import FeedForward, TransformerEncoderBlock


class KeypointsEmbedding(nn.Module):
    def __init__(self, ndim, add_position_patch=True):
        super().__init__()
        self.emb_kps = nn.Conv2d(2, ndim, kernel_size=1)
        self.npatchs = 17

        self.add_position_patch = add_position_patch
        if add_position_patch:
            self.npatchs += 2
            self.emb_bbox = nn.Conv2d(2, ndim, kernel_size=1)

    def forward(self, kps, bboxs=None):
        # kps (b, seq_len, 17, 2)
        kps = kps.permute(0, 3, 1, 2)
        kps = self.emb_kps(kps)
        kps = kps.permute(0, 2, 3, 1)  # (b, seq_len, 17, ndim)

        if self.add_position_patch:
            # bbox (b, seq_len, 2, 2)
            bboxs = bboxs.permute(0, 3, 1, 2)
            bboxs = self.emb_bbox(bboxs)
            bboxs = bboxs.permute(0, 2, 3, 1)  # (b, seq_len, 2, ndim)

            return torch.cat([kps, bboxs], dim=2)  # (b, seq_len, 19, ndim)
        else:
            return kps  # (b, seq_len, 17, ndim)


class ImageEmbedding(nn.Module):
    def __init__(
        self, ndim, add_position_patch=True, patch_size=(16, 12), img_size=(256, 192)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.npatchs = self.get_npatchs(img_size, patch_size)

        self.emb_imgs1 = nn.Conv2d(5, 1, kernel_size=1)
        size = patch_size[0] * patch_size[1]
        self.emb_imgs2 = nn.Conv2d(size, ndim, kernel_size=1)

        self.add_position_patch = add_position_patch
        if add_position_patch:
            self.npatchs += 2
            self.emb_bbox = nn.Conv2d(2, ndim, kernel_size=1)

    @staticmethod
    def get_npatchs(img_size, patch_size):
        return (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

    def patching(self, imgs):
        ph, pw = self.patch_size
        imgs = imgs.unfold(3, ph, ph).unfold(4, pw, pw).contiguous()
        b, seq_len, c, nh, nw, ph, pw = imgs.size()
        return imgs.view(b, seq_len, c, nh * nw, ph * pw)

    def forward(self, imgs, bboxs=None):
        # imgs (b, seq_len, 5, 256, 192)
        imgs = self.patching(imgs)
        b, seq_len, c, nimgs, size = imgs.size()
        # imgs (b, seq_len, 5, nimg, size)

        imgs = imgs.view(b * seq_len, c, nimgs, size)
        imgs = self.emb_imgs1(imgs)
        imgs = imgs.view(b, seq_len, nimgs, size)
        imgs = imgs.permute(0, 3, 1, 2)
        # imgs (b, size, seq_len, nimgs)
        imgs = self.emb_imgs2(imgs)
        imgs = imgs.permute(0, 2, 3, 1)
        # imgs (b, seq_len, nimgs, ndim)

        if self.add_position_patch:
            # bbox (b, seq_len, 2, 2)
            bboxs = bboxs.permute(0, 3, 1, 2)
            bboxs = self.emb_bbox(bboxs)
            bboxs = bboxs.permute(0, 2, 3, 1)
            # bboxs (b, seq_len, 2, ndim)
            return torch.cat([imgs, bboxs], dim=2)  # (b, seq_len, 2 + nimgs, ndim)
        else:
            return imgs  # (b, seq_len, nimgs, ndim)


class TransformerEmbedding(nn.Module):
    def __init__(self, in_ndim, out_ndim, npatchs, nheads, nlayers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(in_ndim, nheads, dropout) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm((npatchs, in_ndim))
        self.ff = FeedForward(npatchs * in_ndim, out_ndim)

    def forward(self, x):
        # x (b, seq_len, npatch, in_ndim)
        b, seq_len, npatch, ndim = x.size()

        x = x.view(b * seq_len, npatch, ndim)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.view(b, seq_len, npatch * ndim)
        x = self.ff(x)
        # x (b, seq_len, out_ndim)
        return x


class IndividualEmbedding(nn.Module):
    def __init__(
        self,
        data_type: str,
        hidden_ndim: int,
        out_ndim: int,
        nheads: int,
        nlayers: int,
        dropout: float = 0.1,
        add_position_patch: bool = True,
        patch_size=(16, 12),
        img_size=(256, 192),
    ):
        super().__init__()

        self.data_type = data_type
        if data_type == "keypoints":
            self.emb_kps = KeypointsEmbedding(hidden_ndim, add_position_patch)
            self.npatchs = self.emb_kps.npatchs
        elif data_type == "images":
            self.emb_imgs = ImageEmbedding(
                hidden_ndim, add_position_patch, patch_size, img_size
            )
            self.npatchs = self.emb_imgs.npatchs
        else:
            raise ValueError

        self.pe = RotaryEmbedding(hidden_ndim)

        self.emb_transformer = TransformerEmbedding(
            hidden_ndim, out_ndim, self.npatchs, nheads, nlayers, dropout
        )

    def forward(self, x, bbox=None):
        if self.data_type == "keypoints":
            x = self.emb_kps(x, bbox)
        elif self.data_type == "images":
            x = self.emb_imgs(x, bbox)
        else:
            raise ValueError

        x = self.pe.rotate_queries_or_keys(x)
        x = self.emb_transformer(x)
        return x  # x (b, seq_len, out_ndim)
