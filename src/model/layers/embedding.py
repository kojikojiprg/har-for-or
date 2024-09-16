import torch
import torch.nn as nn

# from rotary_embedding_torch import RotaryEmbedding

# from .feedforward import MLP, SwiGLU
# from .transformer import TransformerEncoderBlock

# MASK_KPS = [
#     -0.5,  # nose
#     -0.5,  # lefteye
#     -0.5,  # right eye
#     -0.5,  # left ear
#     -0.5,  # right ear
#     0.0,  # left shoulder
#     0.0,  # right shoulder
#     0.0,  # left elbow
#     0.0,  # right elbow
#     0.0,  # left wrist
#     0.0,  # right wrist
#     0.0,  # left hip
#     0.0,  # right hip
#     -1e10,  # left knee
#     -1e10,  # right knee
#     -1e10,  # left ankle
#     -1e10,  # right ankle
# ]


# class PointEmbedding(nn.Module):
#     def __init__(self, emb_hidden_ndim, nheads, dropout, point_type):
#         super().__init__()
#         self.emb_hidden_ndim = emb_hidden_ndim
#         self.point_type = point_type
#         if point_type == "bbox":
#             self.npatchs = 2
#         elif point_type == "keypoints":
#             self.npatchs = 17

#         self.feature = nn.Parameter(
#             torch.randn((1, 1, emb_hidden_ndim), dtype=torch.float32),
#             requires_grad=True,
#         )
#         self.feature_mask = nn.Parameter(
#             torch.full((1, 1), -float("inf")), requires_grad=True
#         )

#         self.mlp = MLP(2, emb_hidden_ndim)
#         self.pe = RotaryEmbedding(emb_hidden_ndim, learned_freq=False)
#         self.attn = nn.MultiheadAttention(
#             emb_hidden_ndim, nheads, dropout, batch_first=True
#         )

#     def forward(self, pt):
#         b, seq_len = pt.size()[:2]

#         # emb
#         pt = self.mlp(pt)

#         # concat feature patch
#         fp = self.feature.repeat((b, seq_len, 1, 1))
#         pt = torch.concat([fp, pt], dim=2)
#         pt = pt.view(b * seq_len, self.npatchs + 1, self.emb_hidden_ndim)

#         # concat mask
#         fp_mask = self.feature_mask.repeat((b, seq_len, 1))
#         if self.point_type == "bbox":
#             pad_mask = torch.full((b, seq_len, self.npatchs), 0.0).to(pt.device)
#             pad_mask = torch.cat([fp_mask, pad_mask], dim=2)
#         elif self.point_type == "keypoints":
#             pad_mask = torch.tensor(MASK_KPS).to(pt.device)
#             pad_mask = pad_mask.view(1, self.npatchs).repeat((b, seq_len, 1))
#             pad_mask = torch.cat([fp_mask, pad_mask], dim=2)
#         pad_mask = pad_mask.view(b * seq_len, self.npatchs + 1)

#         # self attention
#         pt = self.pe.rotate_queries_or_keys(pt, seq_dim=1)
#         pt = self.attn(pt, pt, pt, key_padding_mask=pad_mask, need_weights=False)[0]
#         pt = pt.view(b, seq_len, self.npatchs + 1, self.emb_hidden_ndim)[:, :, 0]

#         return pt


# class TransformerEmbedding(nn.Module):
#     def __init__(self, in_ndim, out_ndim, nheads, nlayers, dropout):
#         super().__init__()
#         self.feature = nn.Parameter(torch.randn((1, 1, in_ndim)), requires_grad=True)

#         self.pe = RotaryEmbedding(in_ndim + 1, learned_freq=True)
#         self.layers = nn.ModuleList(
#             [
#                 TransformerEncoderBlock(in_ndim, nheads, dropout).clone()
#                 for _ in range(nlayers)
#             ]
#         )
#         self.ff = SwiGLU(in_ndim, out_ndim)

#     def forward(self, x):
#         # x (b, npatch, in_ndim)
#         b = x.size(0)

#         x = torch.cat([self.feature.repeat(b, 1, 1), x], dim=1)

#         x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

#         for layer in self.layers:
#             x = layer(x)[0]

#         x = x[:, 0, :]
#         x = self.ff(x)
#         # x (b, out_ndim)
#         return x


# class PixcelEmbedding(nn.Module):
#     def __init__(
#         self,
#         hidden_ndim,
#         out_ndim,
#         nheads,
#         nlayers,
#         dropout,
#         patch_size,
#         img_size,
#     ):
#         super().__init__()
#         self.hidden_ndim = hidden_ndim
#         self.patch_size = patch_size
#         self.npatchs = self.get_npatchs(img_size, patch_size)

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(5, hidden_ndim // 4, 1, bias=False),
#             nn.SiLU(),
#             nn.BatchNorm2d(hidden_ndim // 4),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(hidden_ndim // 4, hidden_ndim // 2, 5, bias=False),
#             nn.SiLU(),
#             nn.BatchNorm2d(hidden_ndim // 2),
#             nn.MaxPool2d(5, 1, 0),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(hidden_ndim // 2, hidden_ndim, 5, bias=False),
#             nn.SiLU(),
#             nn.BatchNorm2d(hidden_ndim),
#             nn.MaxPool2d(5, 1, 0),
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(hidden_ndim, hidden_ndim, 3, bias=False),
#             nn.SiLU(),
#             nn.BatchNorm2d(hidden_ndim),
#             nn.MaxPool2d(3, 1, 0),
#         )
#         self.flatten = nn.Linear(hidden_ndim * 12 * 4, hidden_ndim)

#         self.transformer = TransformerEmbedding(
#             hidden_ndim, out_ndim, nheads, nlayers, dropout
#         )

#     @staticmethod
#     def get_npatchs(img_size, patch_size):
#         return (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

#     def patching(self, imgs):
#         ph, pw = self.patch_size
#         imgs = imgs.unfold(2, ph, ph).unfold(3, pw, pw).contiguous()
#         n, c, nh, nw, ph, pw = imgs.size()
#         return imgs.view(n, c, nh * nw, ph, pw)

#     def forward(self, imgs):
#         # imgs (b, 5, 256, 192)
#         imgs = self.patching(imgs)
#         # imgs (b, 5, nimgs, h, w)

#         imgs = imgs.permute(0, 2, 1, 3, 4).contiguous()
#         b, nimgs, c, h, w = imgs.size()
#         imgs = imgs.view(b * nimgs, c, h, w)
#         imgs = self.conv1(imgs)
#         imgs = self.conv2(imgs)
#         imgs = self.conv3(imgs)
#         imgs = self.conv4(imgs)
#         imgs = imgs.view(b, nimgs, self.hidden_ndim * 12 * 4)
#         imgs = self.flatten(imgs)
#         # imgs (b, nimgs, ndim)

#         imgs = self.transformer(imgs)

#         return imgs  # (b, ndim)


# class IndividualEmbedding(nn.Module):
#     def __init__(
#         self,
#         emb_hidden_ndim: int,
#         hidden_ndim: int,
#         nheads: int,
#         nlayers: int,
#         dropout: float,
#         patch_size,
#         img_size,
#     ):
#         super().__init__()
#         self.emb_hidden_ndim = emb_hidden_ndim
#         # self.emb_vis = PixcelEmbedding(
#         #     hidden_ndim, out_ndim, nheads, nlayers, dropout, patch_size, img_size
#         # )
#         self.feature = nn.Parameter(
#             torch.randn((1, 1, emb_hidden_ndim), dtype=torch.float32),
#             requires_grad=True,
#         )
#         self.feature_mask = nn.Parameter(
#             torch.full((1, 1), -float("inf")), requires_grad=False
#         )
#         self.npatchs = 3
#         self.emb_vis = PointEmbedding(emb_hidden_ndim, nheads, dropout, "keypoints")
#         self.emb_spc = PointEmbedding(emb_hidden_ndim, nheads, dropout, "bbox")
#         self.emb_spc_diff = PointEmbedding(emb_hidden_ndim, nheads, dropout, "bbox")
#         self.attn = nn.MultiheadAttention(
#             emb_hidden_ndim, nheads, dropout, batch_first=True
#         )
#         self.mlp = nn.Sequential(
#             MLP(emb_hidden_ndim, hidden_ndim),
#             nn.SiLU(),
#         )

#     def forward(self, x_vis, x_spc, x_spc_diff, lmd_vis=0.3):
#         x_vis = self.emb_vis(x_vis)
#         x_spc = self.emb_spc(x_spc)
#         x_spc_diff = self.emb_spc_diff(x_spc_diff)
#         # lmd_spc = (1 - lmd_vis) / 2
#         # x = x_vis * lmd_vis + x_spc * lmd_spc + x_spc_diff * lmd_spc
#         # x = x_vis + x_spc + x_spc_diff

#         # attention
#         b, seq_len = x_vis.size()[:2]
#         x_vis = x_vis.view(b * seq_len, 1, self.emb_hidden_ndim)
#         x_spc = x_spc.view(b * seq_len, 1, self.emb_hidden_ndim)
#         x_spc_diff = x_spc_diff.view(b * seq_len, 1, self.emb_hidden_ndim)
#         fp = self.feature.repeat((b, seq_len, 1)).view(
#             b * seq_len, 1, self.emb_hidden_ndim
#         )
#         x = torch.cat([fp, x_vis, x_spc, x_spc_diff], dim=1)
#         fp_mask = self.feature_mask.repeat((b, seq_len, 1))
#         pad_mask = torch.full((b, seq_len, self.npatchs), 0.0).to(x_vis.device)
#         pad_mask = torch.cat([fp_mask, pad_mask], dim=2)
#         pad_mask = pad_mask.view(b * seq_len, self.npatchs + 1)
#         x = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)[0]
#         x = x.view(b, seq_len, self.npatchs + 1, self.emb_hidden_ndim)[:, :, 0]

#         x = self.mlp(x)
#         return x  # x (b, seq_len, out_ndim)


class Embedding(nn.Module):
    def __init__(self, seq_len, hidden_ndim, latent_ndim):
        super().__init__()
        self.hidden_ndim = hidden_ndim

        self.conv_x = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, latent_ndim, 1, bias=False),
            nn.SiLU(),
        )
        self.conv_y = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, latent_ndim, 1, bias=False),
            nn.SiLU(),
        )

    def forward(self, x):
        x, y = x[:, :, :, 0], x[:, :, :, 1]
        x = self.conv_x(x)  # (b, latent_ndim, n_pts)
        y = self.conv_y(y)  # (b, latent_ndim, n_pts)
        x = torch.cat([x, y], dim=2)
        x = x.permute(0, 2, 1)
        # x (b, n_pts * 2, latent_ndim)

        return x
