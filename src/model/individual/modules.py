from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import MLP, TransformerEncoderBlock


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


def calc_distance(z, book):
    distances = (
        torch.sum(z**2, dim=-1, keepdim=True)
        + torch.sum(book**2, dim=-1)
        - 2 * torch.matmul(z, book.t())
    )

    return distances


class Embedding(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.seq_len = config.seq_len
        self.n_pts = n_pts
        self.latent_dim = config.latent_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, config.latent_dim // 8, 6, 3),
            nn.GroupNorm(1, config.latent_dim // 8),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(config.latent_dim // 8, config.latent_dim // 4, 5, 3),
            nn.GroupNorm(1, config.latent_dim // 4),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(config.latent_dim // 4, config.latent_dim // 2, 5, 2),
            nn.GroupNorm(1, config.latent_dim // 2),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(config.latent_dim // 2, config.latent_dim, 3, 1),
            nn.GroupNorm(1, config.latent_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        # x (b, seq_len, n_pts, 2)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, n_pts, 2, seq_len)
        x = x.view(-1, 2, self.seq_len)  # (b * n_pts, 2, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (b * n_pts, dim, 1)
        x = x.view(-1, self.n_pts, self.latent_dim)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.latent_dim))
        self.input_type = config.input_cls
        self.pe = RotaryEmbedding(config.latent_dim, learned_freq=False)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_dim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers_cls)
            ]
        )
        self.mlp = MLP(config.latent_dim, config.n_clusters)

        log_param_q_cls = np.log(config.param_q_cls_init)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )

    def select_input(self, z):
        if self.input_type == "all":
            return z
        elif self.input_type == "bbox":
            return z[:, 2:, :]
        elif self.input_type == "kps":
            return z[:, :2, :]
        else:
            raise ValueError

    def forward(self, z, is_train):
        # z (b, n_pts, latent_dim)
        z = self.select_input(z)

        # concat cls_token
        cls_token = self.cls_token.repeat(z.size(0), 1, 1)
        z = torch.cat([cls_token, z], dim=1)
        # z (b, 1 + n_pts, latent_dim)

        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

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
        # z (b, 1 + n_pts, latent_dim)

        logits = self.mlp(z[:, 0, :])  # (b, n_clusters)

        param_q = self.log_param_q_cls.exp()
        precision_q_cls = 0.5 / torch.clamp(param_q, min=1e-10)
        logits = logits * precision_q_cls

        return logits, attn_w_tensor


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.emb_kps = Embedding(config, 13)
        self.emb_bbox = Embedding(config, 2)
        self.pe = RotaryEmbedding(config.latent_dim, learned_freq=False)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_dim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

    def forward(self, kps, bbox, is_train):
        # embedding
        kps = self.emb_kps(kps)
        bbox = self.emb_bbox(bbox)
        z = torch.cat([kps, bbox], dim=1)
        # z (b, n_pts, latent_dim)

        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

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
        # z (b, n_pts, latent_dim)

        return z, attn_w_tensor


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.book_size = config.book_size

        mu = [torch.randn(1, config.latent_dim) for _ in range(config.n_clusters)]
        self.books = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.book_size, config.latent_dim) + mu[i])
                for i in range(config.n_clusters)
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

    def forward(self, z, c_probs, temperature, is_train):
        # z (b, n_pts, latent_dim)
        b, n_pts, latent_dim = z.size()

        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        if is_train:
            logits = torch.empty((0, n_pts, self.book_size)).to(z.device)
            zq = torch.empty((0, n_pts, latent_dim)).to(z.device)
            books = torch.cat(list(self.books.parameters()), dim=0)
            books = books.view(-1, self.book_size, latent_dim)
            for batch_idx, c_prob in enumerate(c_probs):
                # compute logits and zq of all books
                zi = z[batch_idx]
                zqi = torch.zeros_like(zi)
                logit = torch.zeros((n_pts, self.book_size)).to(z.device)
                for j, book in enumerate(books):
                    logitj = -self.calc_distance(zi, book) * precision_q
                    logit = logit + logitj * c_prob[j]
                    encoding = gumbel_softmax_sample(logitj, temperature)
                    zqi = zqi + torch.matmul(encoding, book) * c_prob[j]

                logits = torch.cat(
                    [logits, logit.view(1, n_pts, self.book_size)], dim=0
                )
                zq = torch.cat([zq, zqi.view(1, n_pts, latent_dim)], dim=0)
        else:
            logits = torch.empty((0, n_pts, self.book_size)).to(z.device)
            books = torch.empty((0, self.book_size, latent_dim)).to(z.device)
            for batch_idx, idx in enumerate(c_probs.argmax(dim=-1)):
                book = self.books[idx]
                logit = -self.calc_distance(z[batch_idx], book) * precision_q

                logits = torch.cat(
                    [logits, logit.view(1, n_pts, self.book_size)], dim=0
                )
                books = torch.cat(
                    [books, book.view(1, self.book_size, latent_dim)], dim=0
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
            # zq (b, npts, latent_dim)

        logits = logits.view(b, -1, self.book_size)

        return zq, precision_q, logits


class Reconstruction(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.seq_len = config.seq_len
        self.n_pts = n_pts
        self.latent_dim = config.latent_dim
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim, config.latent_dim // 2, 3, 1),
            nn.GroupNorm(1, config.latent_dim // 2),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim // 2, config.latent_dim // 4, 5, 2),
            nn.GroupNorm(1, config.latent_dim // 4),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim // 4, config.latent_dim // 8, 5, 3),
            nn.GroupNorm(1, config.latent_dim // 8),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim // 8, 2, 6, 3),
            # nn.GroupNorm(1, seq_len),
            nn.Tanh(),
        )

    def forward(self, zq):
        # zq (b, n_pts, dim)
        zq = zq.reshape(-1, self.latent_dim, 1)  # (b * n_pts, dim, 1)
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
        self.pe = RotaryEmbedding(config.latent_dim, learned_freq=False)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_dim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.recon_kps = Reconstruction(config, 13)
        self.recon_bbox = Reconstruction(config, 2)

    def forward(self, zq):
        zq = self.pe.rotate_queries_or_keys(zq, seq_dim=1)

        # zq (b, n_pts, latent_dim)
        for layer in self.encoders:
            zq, attn_w = layer(zq)

        recon_kps = self.recon_kps(zq[:, :13, :])
        recon_bbox = self.recon_bbox(zq[:, 13:, :])

        return recon_kps, recon_bbox
