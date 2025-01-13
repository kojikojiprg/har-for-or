from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layers import MLP  # , TransformerEncoderBlock


def get_n_pts(config: SimpleNamespace):
    if not config.mask_leg:
        n_pts = 17 + 2
    else:  # mask ankles and knees
        n_pts = 17 - 4 + 2
    return n_pts


def sample_gumbel(logits, eps=1e-10):
    U = torch.rand_like(logits)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits)
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(ze, book):
    distances = (
        torch.sum(ze**2, dim=-1, keepdim=True)
        + torch.sum(book**2, dim=-1)
        - 2 * torch.matmul(ze, book.t())
    )

    return distances


class TransformerEncoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout):
        super().__init__()
        self.nheads = nheads
        self.attn = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(ndim)

        self.ff = MLP(ndim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ndim)

    def forward(self, x, need_weights=False):
        # x (b * seq_len, npatch, ndim)
        x_attn, attn_w = self.attention_block(x, need_weights)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self.feed_forward_block(x))

        return x, attn_w

    def attention_block(self, x, need_weights):
        x, attn_w = self.attn(x, x, x, need_weights=need_weights)
        return x, attn_w

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout2(x)
        return x


class Embedding(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.seq_len = config.seq_len
        self.n_pts = n_pts
        self.latent_ndim = config.latent_ndim
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, config.latent_ndim // 8, 6, 3),
            nn.GroupNorm(1, config.latent_ndim // 8),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(config.latent_ndim // 8, config.latent_ndim // 4, 5, 3),
            nn.GroupNorm(1, config.latent_ndim // 4),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(config.latent_ndim // 4, config.latent_ndim // 2, 5, 2),
            nn.GroupNorm(1, config.latent_ndim // 2),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(config.latent_ndim // 2, config.latent_ndim, 3, 1),
            nn.GroupNorm(1, config.latent_ndim),
            nn.SiLU(),
        )

    def forward(self, x):
        # x (b, seq_len, n_pts, 2)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, n_pts, 2, seq_len)
        x = x.view(-1, 2, self.seq_len)  # (b * n_pts, 2, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (b * n_pts, ndim, 1)
        x = x.view(-1, self.n_pts, self.latent_ndim)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.latent_ndim))
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers_cls)
            ]
        )
        self.mlp = MLP(config.latent_ndim, config.n_clusters)

        # log_param_q_cls = np.log(config.param_q_cls_init)
        # self.log_param_q_cls = nn.Parameter(
        #     torch.tensor(log_param_q_cls, dtype=torch.float32)
        # )

    def forward(self, zq, is_train):
        # zq (b, n_pts, latent_ndim)

        # concat cls_token
        cls_token = self.cls_token.repeat(zq.size(0), 1, 1)
        z = torch.cat([cls_token, zq], dim=1)
        # z (b, 1 + n_pts, latent_ndim)

        if is_train:
            for layer in self.encoders:
                z, attn_w = layer(z)
            attn_w_tensor = None
        else:
            attn_w_lst = []
            for layer in self.encoders:
                z, attn_w = layer(z, need_weights=True)
                attn_w_lst.append(attn_w.unsqueeze(1))
            attn_w_tensor = torch.cat(attn_w_lst, dim=1)
        # z (b, 1 + n_pts, latent_ndim)

        logits = self.mlp(z[:, 0, :])  # (b, n_clusters)

        # param_q = self.log_param_q_cls.exp()
        # precision_q_cls = 0.5 / torch.clamp(param_q, min=1e-10)
        # logits = logits * precision_q_cls

        return logits, attn_w_tensor


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_ndim = config.latent_ndim
        # self.emb = Embedding(config)
        self.emb_kps = Embedding(config, 13)
        self.emb_bbox = Embedding(config, 2)

        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

    def forward(self, kps, bbox, is_train):
        # embedding
        kps = self.emb_kps(kps)
        bbox = self.emb_bbox(bbox)
        z = torch.cat([kps, bbox], dim=1)
        # z (b, n_pts, latent_ndim)

        if is_train:
            for layer in self.encoders:
                z, attn_w = layer(z)
            attn_w_tensor = None
        else:
            attn_w_lst = []
            for layer in self.encoders:
                z, attn_w = layer(z, need_weights=True)
                attn_w_lst.append(attn_w.unsqueeze(1))
            attn_w_tensor = torch.cat(attn_w_lst, dim=1)
        # z (b, n_pts, latent_ndim)

        return z, attn_w_tensor


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.book_size = config.book_size
        self.books = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.book_size, config.latent_ndim))
                for _ in range(config.n_clusters)
            ]
        )

        log_param_q = np.log(config.param_q_init)
        self.log_param_q = nn.Parameter(torch.tensor(log_param_q, dtype=torch.float32))

    def calc_distance(self, z, book):
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(book**2, dim=1)
            - 2 * torch.matmul(z, book.t())
        )
        return distances

    def forward(self, ze, c_probs, temperature, is_train):
        # ze (b, n_pts, latent_ndim)
        b, n_pts, latent_ndim = ze.size()

        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        if is_train:
            logits = torch.empty((0, n_pts, self.book_size)).to(ze.device)
            zq = torch.empty((0, n_pts, latent_ndim)).to(ze.device)
            books = torch.cat(list(self.books.parameters()), dim=0)
            books = books.view(-1, self.book_size, latent_ndim)
            for batch_idx, c_prob in enumerate(c_probs):
                # compute logits and zq of all books
                zei = ze[batch_idx]
                zqi = torch.zeros_like(zei)
                logit = torch.zeros((n_pts, self.book_size)).to(ze.device)
                for j, book in enumerate(books):
                    logitj = -self.calc_distance(zei, book) * precision_q
                    logit = logit + logitj * c_prob[j]
                    encoding = gumbel_softmax_sample(logitj, temperature)
                    zqi = zqi + torch.matmul(encoding, book) * c_prob[j]

                logits = torch.cat(
                    [logits, logit.view(1, n_pts, self.book_size)], dim=0
                )
                zq = torch.cat([zq, zqi.view(1, n_pts, latent_ndim)], dim=0)
                # mean_prob = torch.mean(prob.detach(), dim=0)
        else:
            logits = torch.empty((0, n_pts, self.book_size)).to(ze.device)
            books = torch.empty((0, self.book_size, latent_ndim)).to(ze.device)
            for batch_idx, idx in enumerate(c_probs.argmax(dim=-1)):
                book = self.books[idx]
                logit = -self.calc_distance(ze[batch_idx], book) * precision_q

                logits = torch.cat(
                    [logits, logit.view(1, n_pts, self.book_size)], dim=0
                )
                books = torch.cat(
                    [books, book.view(1, self.book_size, latent_ndim)], dim=0
                )

            indices = torch.argmax(logits, dim=2).unsqueeze(2)
            encodings = torch.zeros(
                indices.shape[0],
                indices.shape[1],
                self.book_size,
                device=indices.device,
            )
            encodings.scatter_(2, indices, 1)
            zq = torch.matmul(encodings, books)
            # mean_prob = torch.mean(encodings, dim=0)
        # zq (b, npts, latent_ndim)

        logits = logits.view(b, -1, self.book_size)

        return zq, precision_q, logits


class Reconstruction(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.seq_len = config.seq_len
        self.n_pts = n_pts
        self.latent_ndim = config.latent_ndim
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_ndim, config.latent_ndim // 2, 3, 1),
            nn.GroupNorm(1, config.latent_ndim // 2),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_ndim // 2, config.latent_ndim // 4, 5, 2),
            nn.GroupNorm(1, config.latent_ndim // 4),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_ndim // 4, config.latent_ndim // 8, 5, 3),
            nn.GroupNorm(1, config.latent_ndim // 8),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_ndim // 8, 2, 6, 3),
            # nn.GroupNorm(1, seq_len),
            nn.Tanh(),
        )

    def forward(self, zq):
        # zq (b, n_pts, ndim)
        zq = zq.reshape(-1, self.latent_ndim, 1)  # (b * n_pts, ndim, 1)
        zq = self.conv1(zq)
        zq = self.conv2(zq)
        zq = self.conv3(zq)
        x = self.conv4(zq)
        # x (b * n_pts, 2, seq_len)
        x = x.view(-1, self.n_pts, 2, self.seq_len)
        x = x.permute(0, 3, 1, 2)
        # x (b, seq_len, n_pts, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        # self.recon = Reconstruction(config)
        self.recon_kps = Reconstruction(config, 13)
        self.recon_bbox = Reconstruction(config, 2)

    def forward(self, zq):
        # zq (b, n_pts, latent_ndim)
        for layer in self.encoders:
            zq, attn_w = layer(zq)

        # x = self.recon(zq)
        # recon_kps, recon_bbox = x[:, :, :-2], x[:, :, -2:]
        recon_kps = self.recon_kps(zq[:, :13, :])
        recon_bbox = self.recon_bbox(zq[:, 13:, :])

        return recon_kps, recon_bbox
