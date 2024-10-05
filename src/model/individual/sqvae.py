from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import (
    MLP,
    Embedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class SQVAE(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        annotation_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters

        if not config.mask_leg:
            self.n_pts = 17 + 2
            self.n_pts_kps = 17
        else:  # mask ankles and knees
            self.n_pts = 17 - 4 + 2
            self.n_pts_kps = 17 - 4

        self.encoder = None
        self.decoder = None
        self.quantizer = None

        self.annotation_path = annotation_path
        self.annotations = None

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(self.config)
        self.decoders = nn.ModuleList(
            [Decoder(self.config) for _ in range(self.n_pts * 2)]
        )
        self.quantizer = GaussianVectorQuantizer(self.config)

        if self.annotation_path is not None:
            anns = np.loadtxt(self.annotation_path, str, delimiter=" ", skiprows=1)
            self.annotations = anns

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda epoch: self.config.lr_lmd**epoch
        )
        return [opt], [sch]

    def forward(self, kps, bbox, is_train):
        # kps (b, seq_len, n_pts, 2)
        # bbox (b, seq_len, n_pts, 2)

        # encoding
        ze, c_logits, attn_w = self.encoder(kps, bbox, is_train)
        c_prob = F.softmax(c_logits, dim=-1)
        # ze (b, npts, latent_ndim)
        # c_prob (b, n_clusters)

        # quantization
        zq, precision_q, prob, log_prob = self.quantizer(ze, c_logits, is_train)
        # zq (b, npts, latent_ndim)
        # prob (b, npts, book_size)

        # reconstruction
        b, seq_len = kps.size()[:2]
        kps = kps.view(b, seq_len, self.n_pts_kps * 2)
        recon_kps = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.decoders[: self.n_pts_kps * 2]):
            recon_x = decoder(kps[:, :, i], zq[:, i, :])
            recon_kps = torch.cat([recon_kps, recon_x], dim=2)
        recon_kps = recon_kps.view(b, seq_len, self.n_pts_kps, 2)

        bbox = bbox.view(b, seq_len, 2 * 2)
        recon_bbox = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.decoders[self.n_pts_kps * 2 :]):
            recon_x = decoder(bbox[:, :, i], zq[:, i, :])
            recon_bbox = torch.cat([recon_bbox, recon_x], dim=2)
        recon_bbox = recon_bbox.view(b, seq_len, 2, 2)

        return (
            ze,
            zq,
            attn_w,
            precision_q,
            prob,
            log_prob,
            recon_kps,
            recon_bbox,
            c_prob,
        )

    def calc_temperature(self):
        return np.max(
            [
                self.temp_init * np.exp(-self.temp_decay * self.global_step),
                self.temp_min,
            ]
        )

    def mse_x(self, x, recon_x, mask=None):
        if x.ndim == 4:
            return F.mse_loss(recon_x, x, reduction="none").mean(dim=(1, 2, 3))  # (b,)
        elif x.ndim == 3:
            return F.mse_loss(recon_x, x, reduction="none").mean(dim=(1, 2))  # (b,)

    def loss_x(self, x, recon_x, mask=None):
        mses = self.mse_x(x, recon_x, mask)
        return mses.mean()

    def loss_kl_continuous(self, ze, zq, precision_q):
        return torch.sum(((ze - zq) ** 2) * precision_q, dim=(1, 2)).mean()

    def loss_kl_discrete(self, prob, log_prob):
        return torch.sum(prob * log_prob, dim=(0, 1)).mean()

    def training_step(self, batch, batch_idx):
        keys, ids, kps, bbox, mask = batch
        keys = np.array(keys).T[0]
        ids = ids[0]
        kps = kps[0]
        bbox = bbox[0]
        # mask = mask[0]

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature()
        self.quantizer.temperature = temp_cur

        # forward
        (
            ze,
            zq,
            attn_w,
            precision_q,
            prob,
            log_prob,
            recon_kps,
            recon_bbox,
            c_prob,
        ) = self(kps, bbox, True)

        # ELBO loss
        lrc_kps = self.loss_x(kps, recon_kps)
        lrc_bbox = self.loss_x(bbox, recon_bbox)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(prob, log_prob)
        loss_dict = dict(
            kps=lrc_kps.item(),
            bbox=lrc_bbox.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            log_param_q=self.quantizer.log_param_q.item(),
            log_param_q_cls=self.quantizer.log_param_q_cls.item(),
        )

        # clustering loss
        c_prior = torch.full_like(c_prob, 1 / self.n_clusters)
        c_prob = torch.clamp(c_prob, min=1e-10)
        lc_psuedo = (c_prior * (c_prior.log() - c_prob.log())).mean()
        loss_dict["c_psuedo"] = lc_psuedo.item()

        if self.annotations is not None:
            keys = np.array(["{}_{}".format(*key.split("_")[0::2]) for key in keys])
            mask_supervised = np.isin(keys, self.annotations.T[0]).ravel()
            keys = keys[mask_supervised]
            mask_supervised = torch.tensor(mask_supervised).to(self.device)
            if torch.any(mask_supervised):
                labels = []
                for key in keys:
                    if key in self.annotations.T[0]:
                        label = self.annotations.T[1][key == self.annotations.T[0]]
                        labels.append(int(label))
                labels = torch.tensor(labels).to(self.device, torch.long)
                lc_real = F.cross_entropy(
                    c_prob[mask_supervised], labels, reduction="sum"
                )
                lc_real = lc_real / ids.size(0)
                loss_dict["c_real"] = lc_real.item()
            else:
                lc_real = 0.0
                loss_dict["c_real"] = 0.0
        else:
            lc_real = 0.0

        lc = lc_real * 10 + lc_psuedo * 0.1

        loss_total = (
            (lrc_kps + lrc_bbox) * self.config.lmd_lrc
            + kl_continuous * self.config.lmd_klc
            + kl_discrete * self.config.lmd_kld
            + lc * self.config.lmd_c
        )
        loss_dict["total"] = loss_total.item()

        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    @torch.no_grad()
    def predict_step(self, batch):
        keys, ids, kps, bbox, mask = batch
        keys = np.array(keys).T[0]
        kps = kps.to(next(self.parameters()).device)
        bbox = bbox.to(next(self.parameters()).device)
        if kps.ndim == 5:
            ids = ids[0]
            kps = kps[0]
            bbox = bbox[0]

        # forward
        (
            ze,
            zq,
            attn_w,
            precision_q,
            prob,
            log_prob,
            recon_kps,
            recon_bbox,
            c_prob,
        ) = self(kps, bbox, False)

        mse_kps = self.mse_x(kps, recon_kps)
        mse_bbox = self.mse_x(bbox, recon_bbox)

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i].item(),
                "id": ids[i].cpu().numpy().item(),
                "kps": kps[i].cpu().numpy(),
                "recon_kps": recon_kps[i].cpu().numpy(),
                "mse_kps": mse_kps[i].item(),
                "bbox": bbox[i].cpu().numpy(),
                "recon_bbox": recon_bbox[i].cpu().numpy(),
                "mse_bbox": mse_bbox[i].item(),
                "ze": ze[i].cpu().numpy(),
                "zq": zq[i].cpu().numpy(),
                "attn_w": attn_w[i].cpu().numpy(),
                "book_prob": prob[i].cpu().numpy(),
                "book_idx": prob[i].cpu().numpy().argmax(axis=1),
                "label_prob": c_prob[i].cpu().numpy(),
                "label": c_prob[i].cpu().numpy().argmax(),
            }
            results.append(data)

        return results


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        ndim = config.latent_ndim
        self.conv = nn.Sequential(
            nn.Conv1d(ndim, ndim * 2, 1, bias=False),
            nn.SiLU(),
            nn.AvgPool1d(2),  # 38 -> 19 (mask_leg: 30 -> 15)
            nn.Conv1d(ndim * 2, ndim * 4, 1, bias=False),
            nn.SiLU(),
            nn.AvgPool1d(3, 2),  # 19 -> 9 (mask_leg: 15 -> 7)
            nn.Conv1d(ndim * 4, ndim * 8, 1, bias=False),
            nn.SiLU(),
            nn.AvgPool1d(3, 2),  # 9 -> 4 (mask_leg: 7 -> 3)
        )
        if not config.mask_leg:
            self.mlp = MLP(4 * ndim * 8, config.n_clusters)
        else:
            self.mlp = MLP(3 * ndim * 8, config.n_clusters)

    def forward(self, x):
        # x (b, n_pts, latent_ndim)
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # (b, ndim, 4 or 3)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)  # (b, n_clusters)
        return x


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.emb_kps = Embedding(config.seq_len, config.hidden_ndim, config.latent_ndim)
        self.emb_bbox = Embedding(
            config.seq_len, config.hidden_ndim, config.latent_ndim
        )
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.cls_head = ClassificationHead(config)

    def forward(self, kps, bbox, is_train):
        # kps (b, seq_len, n_pts, 2)
        # bbox (b, seq_len, n_pts, 2)

        # embedding
        kps = self.emb_kps(kps)  # (b, n_pts * 2, latent_ndim)
        bbox = self.emb_bbox(bbox)  # (b, n_pts * 2, latent_ndim)
        z = torch.cat([kps, bbox], dim=1)
        # z (b, n_pts * 2, latent_ndim)

        # positional embedding
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        if is_train:
            for layer in self.encoders:
                z, attn_w = layer(z)
            attn_w_tensor = None
        else:
            attn_w_lst = []
            for layer in self.encoders:
                z, attn_w = layer(z, need_weights=~is_train)
                attn_w_lst.append(attn_w.unsqueeze(1))
            attn_w_tensor = torch.cat(attn_w_lst, dim=1)
        # z (b, n_pts * 2, latent_ndim)

        c_logits = self.cls_head(z)  # (b, n_clusters)

        return z, c_logits, attn_w_tensor


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

        self.temperature = None
        log_param_q = np.log(config.param_q_init)
        self.log_param_q = nn.Parameter(torch.tensor(log_param_q, dtype=torch.float32))
        log_param_q_cls = np.log(config.param_q_init)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )

    def calc_distance(self, z, book):
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(book**2, dim=1)
            - 2 * torch.matmul(z, book.t())
        )
        return distances

    def gumbel_softmax_relaxation(self, logits, eps=1e-10):
        U = torch.rand(logits.shape, device=logits.device)
        g = -torch.log(-torch.log(U + eps) + eps)
        y = logits + g
        return F.softmax(y / self.temperature, dim=-1)

    def forward(self, ze, c_logits, is_train):
        # ze (b, n_pts, latent_ndim)
        b, n_pts, latent_ndim = ze.size()

        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        if is_train:
            param_q = 1 + self.log_param_q_cls.exp()
            precision_q_cls = 0.5 / torch.clamp(param_q, min=1e-10)

            logits = torch.empty((0, n_pts, self.book_size)).to(ze.device)
            zq = torch.empty((0, n_pts, latent_ndim)).to(ze.device)
            for i, c_logit in enumerate(c_logits):
                c_prob = self.gumbel_softmax_relaxation(c_logit * precision_q_cls)

                # compute logits and zq of all books
                zei = ze[i]
                zqi = torch.zeros_like(zei)
                logit = torch.zeros((n_pts, self.book_size)).to(ze.device)
                books = torch.cat(list(self.books.parameters()), dim=0)
                books = books.view(-1, self.book_size, latent_ndim)
                for j, book in enumerate(books):
                    logitj = -self.calc_distance(zei, book) * precision_q
                    logit = logit + logitj * c_prob[j]
                    encoding = self.gumbel_softmax_relaxation(logitj)
                    zqi = zqi + torch.matmul(encoding, book) * c_prob[j]

                logits = torch.cat(
                    [logits, logit.view(1, n_pts, self.book_size)], dim=0
                )
                books = torch.cat(
                    [books, book.view(1, self.book_size, latent_ndim)], dim=0
                )
                zq = torch.cat([zq, zqi.view(1, n_pts, latent_ndim)], dim=0)
                # mean_prob = torch.mean(prob.detach(), dim=0)
        else:
            logits = torch.empty((0, n_pts, self.book_size)).to(ze.device)
            books = torch.empty((0, self.book_size, latent_ndim)).to(ze.device)
            for i, idx in enumerate(c_logits.argmax(dim=-1)):
                book = self.books[idx]
                logit = -self.calc_distance(ze[i], book) * precision_q

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
        log_prob = torch.log_softmax(logits, dim=-1)

        logits = logits.view(b, n_pts, self.book_size)
        prob = prob.view(b, n_pts, self.book_size)
        log_prob = log_prob.view(b, n_pts, self.book_size)

        return zq, precision_q, prob, log_prob


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_ndim = config.latent_ndim

        self.x_start = nn.Parameter(
            torch.randn((1, 1, config.latent_ndim), dtype=torch.float32),
            requires_grad=True,
        )

        self.emb = MLP(1, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.mlp_z = MLP(config.latent_ndim, config.latent_ndim * config.seq_len)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.mlp = nn.Sequential(
            MLP(config.latent_ndim, config.hidden_ndim),
            nn.SiLU(),
            MLP(config.hidden_ndim, 1),
            nn.Tanh(),
        )

    def forward(self, x, zq, mask=None):
        # x (b, seq_len)
        # zq (b, latent_ndim)

        b, seq_len = x.size()
        x = x.view(b, seq_len, 1)
        x = self.emb(x)  # (b, seq_len, latent_ndim)

        # concat start token
        x = torch.cat([self.x_start.repeat((b, 1, 1)), x], dim=1)
        x = x[:, :-1]  # (b, seq_len, latent_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        zq = self.mlp_z(zq)
        zq = zq.view(b, seq_len, self.latent_ndim)
        for layer in self.decoders:
            x = layer(x, zq, mask)
        # x (b, seq_len, latent_ndim)

        recon_x = self.mlp(x).view(b, seq_len, 1)

        return recon_x
