import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from .core import FeedForward, TransformerEncoderBlock


class KeypointsEmbedding(nn.Module):
    def __init__(self, ndim, add_position_patch=True):
        super().__init__()
        self.emb_kps = nn.Conv1d(2, ndim, kernel_size=1)
        self.npatchs = 17

        self.add_position_patch = add_position_patch
        if add_position_patch:
            self.npatchs += 2
            self.emb_bbox = nn.Conv1d(2, ndim, kernel_size=1)

    def forward(self, kps, bboxs=None):
        # kps (n, 17, 2)
        kps = kps.permute(0, 2, 1)  # (n, 2, 17)
        kps = self.emb_kps(kps)
        kps = kps.permute(0, 2, 1)  # (n, 17, ndim)

        if self.add_position_patch:
            # bbox (n, 2, 2)
            bboxs = bboxs.permute(0, 2, 1)
            bboxs = self.emb_bbox(bboxs)
            bboxs = bboxs.permute(0, 2, 1)  # (n, 2, ndim)

            return torch.cat([kps, bboxs], dim=1)  # (n, 19, ndim)
        else:
            return kps  # (n, 17, ndim)


class ImageEmbedding(nn.Module):
    def __init__(
        self, ndim, add_position_patch=True, patch_size=(16, 12), img_size=(256, 192)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.npatchs = self.get_npatchs(img_size, patch_size)

        self.emb_imgs1 = nn.Conv2d(5, 1, kernel_size=1)
        size = patch_size[0] * patch_size[1]
        self.emb_imgs2 = nn.Conv1d(size, ndim, kernel_size=1)

        self.add_position_patch = add_position_patch
        if add_position_patch:
            self.npatchs += 2
            self.emb_bbox = nn.Conv1d(2, ndim, kernel_size=1)

    @staticmethod
    def get_npatchs(img_size, patch_size):
        return (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

    def patching(self, imgs):
        ph, pw = self.patch_size
        imgs = imgs.unfold(2, ph, ph).unfold(3, pw, pw).contiguous()
        n, c, nh, nw, ph, pw = imgs.size()
        return imgs.view(n, c, nh * nw, ph * pw)

    def forward(self, imgs, bboxs=None):
        # imgs (n, 5, 256, 192)
        imgs = self.patching(imgs)
        n, c, nimgs, size = imgs.size()
        # imgs (n, 5, nimg, size)

        imgs = self.emb_imgs1(imgs)
        imgs = imgs.view(n, nimgs, size)
        imgs = imgs.permute(0, 2, 1)
        # imgs (n, size, nimgs)
        imgs = self.emb_imgs2(imgs)
        imgs = imgs.permute(0, 2, 1)
        # imgs (n, nimgs, ndim)

        if self.add_position_patch:
            # bbox (n, 2, 2)
            bboxs = bboxs.permute(0, 2, 1)
            bboxs = self.emb_bbox(bboxs)
            bboxs = bboxs.permute(0, 2, 1)
            # bboxs (n, 2, ndim)
            return torch.cat([imgs, bboxs], dim=1)  # (n, 2 + nimgs, ndim)
        else:
            return imgs  # (n, nimgs, ndim)


class TransformerEmbedding(nn.Module):
    def __init__(self, in_ndim, out_ndim, npatchs, nheads, nlayers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(in_ndim, nheads, dropout) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm((npatchs, in_ndim))
        self.ff = FeedForward(npatchs * in_ndim, out_ndim)

    def forward(self, x):
        # x (n, npatch, in_ndim)
        n, npatch, ndim = x.size()

        for layer in self.layers:
            x = layer(x)[0]

        x = self.norm(x)
        x = x.view(n, npatch * ndim)
        x = self.ff(x)
        # x (n, out_ndim)
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

    def forward(self, x, bboxs=None):
        # kps (n, 17, 2)
        # images (n, 5, h, w)
        # bbox (n, 2, 2)
        if self.data_type == "keypoints":
            x = self.emb_kps(x, bboxs)
        elif self.data_type == "images":
            x = self.emb_imgs(x, bboxs)
        else:
            raise ValueError
        # x (n, npatchs, ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)
        x = self.emb_transformer(x)
        return x  # x (n, out_ndim)
