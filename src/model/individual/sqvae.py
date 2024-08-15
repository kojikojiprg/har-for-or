from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import MLP  # IndividualEmbedding,
from src.model.layers import TransformerDecoderBlock, TransformerEncoderBlock


class SQVAE(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        n_batches: Optional[int] = None,
        annotation_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.annotation_path = annotation_path
        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min
        self.latent_ndim = config.latent_ndim
        self.book_size = config.book_size

        self.encoder = None
        self.decoder = None
        self.quantizers = None

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantizers = nn.ModuleList(
            [
                GaussianVectorQuantizer(self.config)
                for _ in range(len(self.body_indices))
            ]
        )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return opt

    body_indices = [
        list(range(5)),  # face
        [5, 6, 11, 12],  # body
        [7, 9],  # left-arm
        [8, 10],  # right-arm
        [13, 15],  # left-leg
        [14, 16],  # right-leg
        [17, 18],  # bbox
    ]

    def forward(self, x_vis, x_spc, mask, is_train):
        ze = self.encoder(x_vis, x_spc)
        # ze (b, npts, latent_ndim)

        ze_lst = [ze[:, indices, :] for indices in self.body_indices]

        b, n_pts = ze.size()[:2]
        zq = torch.zeros_like(ze)
        precision_q = torch.zeros((b, n_pts, 1)).to(self.device, torch.float32)
        prob = torch.empty((b, n_pts, self.book_size)).to(self.device, torch.float32)
        log_prob = torch.empty((b, n_pts, self.book_size)).to(
            self.device, torch.float32
        )
        for i, indices in enumerate(self.body_indices):
            zq_part, precision_q_part, prob_part, log_prob_part = self.quantizers[i](
                ze_lst[i], is_train
            )
            zq[:, indices, :] = zq_part
            precision_q[:, indices, 0] = precision_q_part
            prob[:, indices, :] = prob_part.view(b, len(indices), -1)
            log_prob[:, indices, :] = log_prob_part.view(b, len(indices), -1)

        recon_x_vis, recon_x_spc = self.decoder(x_vis, x_spc, zq, mask)

        return (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
        )

    def calc_temperature(self):
        return np.max(
            [
                self.temp_init * np.exp(-self.temp_decay * self.global_step),
                self.temp_min,
            ]
        )

    def mse_x(self, x, recon_x, mask=None):
        b = x.size(0)
        mses = torch.empty((0,)).to(self.device)
        for i in range(b):
            # mse = F.mse_loss(recon_x[i][~mask[i]], x[i][~mask[i]])
            mse = F.mse_loss(recon_x[i], x[i])
            mses = torch.cat([mses, mse.view(1, 1)])
        return mses.ravel()  # (b,)

    def loss_x(self, x, recon_x, mask=None):
        b = x.size(0)
        mses = self.mse_x(x, recon_x, mask)
        mses = mses.sum() / b * (17 * 2)
        return mses

    def loss_kl_continuous(self, ze, zq, precision_q):
        return torch.sum(((ze - zq) ** 2) * precision_q, dim=(1, 2)).mean()

    def loss_kl_discrete(self, prob, log_prob):
        return torch.sum(prob * log_prob, dim=(0, 1)).mean()

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        keys = np.array(keys).T[0]
        ids = ids[0]
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        # mask = mask[0]
        mask = None

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature()
        for quantizer in self.quantizers:
            quantizer.temperature = temp_cur

        # forward
        (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
        ) = self(x_vis, x_spc, mask, is_train=True)

        # loss
        lrc_x_vis = self.loss_x(x_vis, recon_x_vis)
        lrc_x_spc = self.loss_x(x_spc, recon_x_spc)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(prob, log_prob)
        loss_total = lrc_x_vis + lrc_x_spc + kl_continuous + kl_discrete * 0.01

        # mean of log_param_q
        log_param_q_lst = []
        for quantizer in self.quantizers:
            log_param_q_lst.append(quantizer.log_param_q.detach())
        log_param_q = torch.tensor(log_param_q_lst).mean()

        loss_dict = dict(
            x_vis=lrc_x_vis.item(),
            x_spc=lrc_x_spc.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            total=loss_total.item(),
            log_param_q=log_param_q.item(),
        )

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
        )

        return loss_total

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        mask = mask.to(next(self.parameters()).device)
        if x_vis.ndim == 5:
            ids = ids[0]
            x_vis = x_vis[0]
            x_spc = x_spc[0]
            mask = mask[0]
        mask = None

        # forward
        (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
        ) = self(x_vis, x_spc, mask, is_train=False)

        mse_x_vis = self.mse_x(x_vis, recon_x_vis, mask)
        mse_x_spc = self.mse_x(x_spc, recon_x_spc, mask)

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i],
                "id": ids[i].cpu().numpy().item(),
                "x_vis": x_vis[i].cpu().numpy(),
                "recon_x_vis": recon_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "recon_x_spc": recon_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "ze": ze[i].cpu().numpy(),
                "zq": zq[i].cpu().numpy(),
                "label_prob": prob[i].cpu().numpy(),
                "label": prob[i].cpu().numpy().argmax().item(),
                # "mask": mask[i].cpu().numpy(),
            }
            results.append(data)

        return results


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp_x = MLP(config.seq_len, config.hidden_ndim // 2)
        self.mlp_y = MLP(config.seq_len, config.hidden_ndim // 2)

    def forward(self, x):
        # x (b, seq_len, n_pts, 2)
        x, y = x[:, :, :, 0], x[:, :, :, 1]

        x = self.mlp_x(x.permute(0, 2, 1))  # (b, n_pts, hidden_ndim / 2)
        y = self.mlp_y(y.permute(0, 2, 1))  # (b, n_pts, hidden_ndim / 2)

        x = torch.cat([x, y], dim=2)  # (b, n_pts, hidden_ndim)
        return x


class Reconstruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim // 2
        self.mlp_x = nn.Sequential(
            MLP(config.hidden_ndim // 2, config.hidden_ndim // 2),
            nn.Linear(config.hidden_ndim // 2, config.seq_len),
            nn.Tanh(),
        )
        self.mlp_y = nn.Sequential(
            MLP(config.hidden_ndim // 2, config.hidden_ndim // 2),
            nn.Linear(config.hidden_ndim // 2, config.seq_len),
            nn.Tanh(),
        )

    def forward(self, x):
        # x (b, n_pts, hidden_ndim)
        x, y = x[:, :, : self.hidden_ndim], x[:, :, self.hidden_ndim :]

        x = self.mlp_x(x).permute(0, 2, 1)  # (b, seq_len, n_pts)
        y = self.mlp_y(y).permute(0, 2, 1)  # (b, seq_len, n_pts)

        b, seq_len, n_pts = x.size()
        recon_x = torch.cat(
            [x.view(b, seq_len, n_pts, 1), y.view(b, seq_len, n_pts, 1)], dim=3
        )  # (b, seq_len, n_pts, 2)
        return recon_x


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.emb_vis = Embedding(config)
        self.emb_spc = Embedding(config)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.mlp = MLP(config.hidden_ndim, config.latent_ndim)

    def forward(self, x_vis, x_spc, mask=None):
        # x_vis (b, seq_len, 17, 2)
        # x_spc (b, seq_len, 2, 2)

        # embedding
        x_vis = self.emb_vis(x_vis)
        x_spc = self.emb_spc(x_spc)
        x = torch.cat([x_vis, x_spc], dim=1)
        # x (b, 17 + 2, hidden_ndim)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, 17 + 2, hidden_ndim)

        z = self.mlp(x)
        # z (b, 17 + 2, latent_ndim)

        return z


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace, vis_npatchs: int = None):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        # self.emb_hidden_ndim = config.emb_hidden_ndim

        # start tokens
        self.x_start = nn.Parameter(
            torch.randn((1, 1, config.hidden_ndim), dtype=torch.float32),
            requires_grad=True,
        )

        self.emb_vis = Embedding(config)
        self.emb_spc = Embedding(config)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=True)
        self.mlp_z = MLP(config.latent_ndim, config.hidden_ndim)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp_rec_x_vis = Reconstruction(config)
        self.mlp_rec_x_spc = Reconstruction(config)

    def forward(self, x_vis, x_spc, zq, mask=None):
        b, seq_len = x_vis.size()[:2]

        # embedding
        x_vis = self.emb_vis(x_vis)
        x_spc = self.emb_spc(x_spc)
        x = torch.cat([x_vis, x_spc], dim=1)
        # x (b, 17 + 2, hidden_ndim)

        # concat start token
        x = torch.cat([self.x_start.repeat((b, 1, 1)), x], dim=1)
        x = x[:, :-1]
        # x (b, 17 + 2, hidden_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        zq = self.mlp_z(zq)
        for layer in self.decoders:
            x = layer(x, zq, mask)
        # x (b, 17 + 2, hidden_ndim)

        # reconstruct
        recon_x_vis = self.mlp_rec_x_vis(x[:, :17, :])
        recon_x_spc = self.mlp_rec_x_spc(x[:, 17:, :])

        return recon_x_vis, recon_x_spc


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.book_size = config.book_size
        self.latent_ndim = config.latent_ndim
        self.book = nn.Parameter(torch.randn(self.book_size, self.latent_ndim))

        self.temperature = None
        log_param_q = np.log(config.param_q_init)
        self.log_param_q = nn.Parameter(torch.tensor(log_param_q, dtype=torch.float32))

    def calc_distance(self, z, book):
        z = z.view(-1, self.latent_ndim)
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

    def forward(self, ze, is_train):
        # z (b, n_vals, latent_ndim)
        b, n_vals, latent_ndim = ze.size()

        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        ze = ze.permute(0, 2, 1).contiguous()
        logits = -self.calc_distance(ze, self.book) * precision_q
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)

        if is_train:
            encodings = self.gumbel_softmax_relaxation(logits)
            zq = torch.mm(encodings, self.book).view(b, latent_ndim, n_vals)
            # mean_prob = torch.mean(prob.detach(), dim=0)
        else:
            indices = torch.argmax(logits, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                indices.shape[0], self.book_size, device=indices.device
            )
            encodings.scatter_(1, indices, 1)
            zq = torch.mm(encodings, self.book).view(b, latent_ndim, n_vals)
            # mean_prob = torch.mean(encodings, dim=0)

        zq = zq.permute(0, 2, 1).contiguous()

        return zq, precision_q, prob, log_prob
