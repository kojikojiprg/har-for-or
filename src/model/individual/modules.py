from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import MLP, TransformerDecoderBlock, TransformerEncoderBlock


def get_n_pts(config: SimpleNamespace):
    if not config.mask_leg:
        n_pts = 17 + 2
    else:  # mask ankles and knees
        n_pts = 17 - 4 + 2
    return n_pts


def gumbel_softmax_relaxation(logits, temperature, eps=1e-10):
    U = torch.rand(logits.shape, device=logits.device)
    g = -torch.log(-torch.log(U + eps) + eps)
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


class Embedding(nn.Module):
    def __init__(self, seq_len, latent_ndim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(seq_len, latent_ndim * 8, (1, 2)),
            nn.GroupNorm(8, latent_ndim * 8),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(latent_ndim * 8, latent_ndim * 4, 1),
            nn.GroupNorm(4, latent_ndim * 4),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(latent_ndim * 4, latent_ndim * 2, 1),
            nn.GroupNorm(2, latent_ndim * 2),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(latent_ndim * 2, latent_ndim, 1),
            nn.GroupNorm(1, latent_ndim),
            nn.SiLU(),
        )

    def forward(self, x):
        # x (b, seq_len, n_pts, 2)
        x = self.conv1(x).squeeze(dim=-1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)  # (b, n_pts, ndim)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.latent_ndim))
        self.emb = Embedding(config.seq_len, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
                # for _ in range(1)
            ]
        )
        self.mlp = MLP(config.latent_ndim, config.n_clusters)

        log_param_q_cls = np.log(config.param_q_cls_init)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )

    def forward(self, kps, bbox, is_train):
        # embedding
        x = torch.cat([kps, bbox], dim=2)
        z = self.emb(x)
        # z (b, n_pts, latent_ndim)

        # concat cls_token
        cls_token = self.cls_token.repeat(z.size(0), 1, 1)
        z = torch.cat([cls_token, z], dim=1)
        # z (b, 1 + n_pts, latent_ndim)

        # positional embedding
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
        # z (b, 1 + n_pts, latent_ndim)

        logits = self.mlp(z[:, 0, :])  # (b, n_clusters)

        param_q = 1 + self.log_param_q_cls.exp()
        precision_q_cls = 0.5 / torch.clamp(param_q, min=1e-10)
        logits = logits * precision_q_cls

        return logits, attn_w_tensor


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_ndim = config.latent_ndim
        self.n_pts = get_n_pts(config)
        self.emb = Embedding(config.seq_len, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.emb_c = MLP(config.n_clusters, config.latent_ndim)

    def forward(self, kps, bbox, c, is_train):
        # kps (b, seq_len, n_pts, 2)
        # bbox (b, seq_len, n_pts, 2)

        # embedding
        b = kps.size(0)
        x = torch.cat([kps, bbox], dim=2)
        z = self.emb(x)
        # z (b, n_pts, latent_ndim)

        # positional embedding
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        c = self.emb_c(c)
        z = z + c.view(b, 1, self.latent_ndim)

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
                    encoding = gumbel_softmax_relaxation(logitj, temperature)
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

        prob = torch.softmax(logits, dim=-1)
        prob = prob.view(b, n_pts, self.book_size)
        log_prob = torch.log_softmax(logits, dim=-1)
        log_prob = log_prob.view(b, n_pts, self.book_size)

        return zq, precision_q, prob, log_prob


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.n_pts = get_n_pts(config)
        self.decoder = DecoderModule(config)

    def forward(self, kps, bbox, zq):
        b, seq_len = kps.size()[:2]
        recon_kps = torch.empty((b, seq_len, 0, 2)).to(kps.device)
        for i in range((self.n_pts - 2)):
            recon_x = self.decoder(kps[:, :, i], zq[:, i, :])
            recon_kps = torch.cat([recon_kps, recon_x], dim=2)

        recon_bbox = torch.empty((b, seq_len, 0, 2)).to(bbox.device)
        for i in range(2):
            recon_x = self.decoder(bbox[:, :, i], zq[:, (self.n_pts - 2) + i, :])
            recon_bbox = torch.cat([recon_bbox, recon_x], dim=2)

        return recon_kps, recon_bbox

    def sample_next_step(self, kps, bbox, zq):
        zq = zq.unsqueeze(0)

        recon_kps = torch.empty((1, 1, 0, 2)).to(kps.device)
        for i in range((self.n_pts - 2)):
            recon_x = self.decoder.sample_next_step(kps[:, :, i], zq[:, i, :])
            recon_kps = torch.cat([recon_kps, recon_x], dim=2)
        kps = torch.cat([kps, recon_kps], dim=1)

        recon_bbox = torch.empty((1, 1, 0, 2)).to(bbox.device)
        for i in range(2):
            recon_x = self.decoder.sample_next_step(bbox[:, :, i], zq[:, (self.n_pts - 2) + i, :])
            recon_bbox = torch.cat([recon_bbox, recon_x], dim=2)
        bbox = torch.cat([bbox, recon_bbox], dim=1)

        return kps, bbox


class DecoderModule(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_ndim = config.latent_ndim

        self.x_start = nn.Parameter(
            torch.randn((1, 1, config.latent_ndim), dtype=torch.float32),
            requires_grad=True,
        )

        self.emb = MLP(2, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.mlp = nn.Sequential(
            MLP(config.latent_ndim, 2),
            nn.Tanh(),
        )

    def forward(self, x, zq, mask=None):
        # x (b, seq_len, 2)
        # zq (b, latent_ndim)

        b, seq_len = x.size()[:2]
        x = self.emb(x)  # (b, seq_len, latent_ndim)

        # concat start token
        x = torch.cat([self.x_start.repeat((b, 1, 1)), x], dim=1)
        x = x[:, :-1]  # (b, seq_len, latent_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        zq = zq.unsqueeze(1).repeat(1, seq_len, 1)
        for layer in self.decoders:
            x = layer(x, zq, mask)
        # x (b, seq_len, latent_ndim)

        recon_x = self.mlp(x).view(b, seq_len, 1, 2)

        return recon_x

    def sample_next_step(self, x, zq):
        if x.size(1) == 0:  # seq_len == 0
            x = self.x_start
        else:
            x = self.emb(x)
            x = torch.cat([self.x_start, x], dim=1)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        zq = zq.unsqueeze(1).repeat(1, x.size(1), 1)
        for layer in self.decoders:
            x = layer(x, zq, is_sampling=True)
        # x (b, seq_len, latent_ndim)

        recon_x = self.mlp(x).view(1, -1, 1, 2)

        return recon_x[:, -1:]
