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
    IndividualEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


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

        self.encoder = None
        self.decoder = None
        self.quantizer = None

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(self.config)
        vis_npatchs = self.encoder.emb.emb_vis.npatchs
        self.decoder = Decoder(self.config, vis_npatchs)
        self.quantizer = GaussianVectorQuantizer(self.config)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return opt

    def forward(self, x_vis, x_spc, x_spc_diff, mask, is_train):
        z_e = self.encoder(x_vis, x_spc, x_spc_diff, mask)
        z_q, precision_q, prob, log_prob, mean_prob = self.quantizer(z_e, is_train)
        recon_x_vis, recon_x_spc, recon_x_spc_diff = self.decoder(
            x_vis, x_spc, x_spc_diff, z_q, mask
        )

        return (
            z_e,
            z_q,
            precision_q,
            prob,
            log_prob,
            mean_prob,
            recon_x_vis,
            recon_x_spc,
            recon_x_spc_diff,
        )

    def _calc_temperature(self):
        temperature = np.max(
            [
                self.temp_init * np.exp(-self.temp_decay * self.global_step),
                self.temp_min,
            ]
        )
        return temperature

    def _calc_reconstruction_loss(self, reconstructed_x, x):
        mse = F.mse_loss(reconstructed_x, x, reduction="sum") / x.shape[0]
        reconstruction_loss = (
            self.input_volume * torch.log(mse) / 2
        )  # https://arxiv.org/abs/2102.08663
        return reconstruction_loss, mse

    def loss_x(self, x, fake_x, mask):
        b = x.size(0)
        mses = torch.empty((0,)).to(self.device)
        for i in range(b):
            mse = F.mse_loss(x[i][~mask[i]], fake_x[i][~mask[i]])
            mses = torch.cat([mses, mse.view(1, 1)])
        return mses.ravel()  # (b,)

    def loss_latent_and_perplexity(
        self, z_e, z_q, precision_q, prob, log_prob, mean_prob
    ):
        kl_discrete = torch.sum(prob * log_prob, dim=(0, 1)) / z_e.shape[0]
        kl_continuous = torch.sum(
            ((z_e - z_q) ** 2) * precision_q,
            dim=list(range(1, z_e.dim())),
        ).mean()

        perplex = torch.exp(-torch.sum(mean_prob * torch.log(mean_prob + 1e-7)))

        return kl_discrete, kl_continuous, perplex

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, x_spc_diff, mask = batch
        keys = np.array(keys).T[0]
        ids = ids[0]
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        x_spc_diff = x_spc_diff[0]
        mask = mask[0]

        temp_cur = self._calc_temperature()
        self.quantizer.temperature = temp_cur

        # forward
        (
            z_e,
            z_q,
            precision_q,
            prob,
            log_prob,
            mean_prob,
            recon_x_vis,
            recon_x_spc,
            recon_x_spc_diff,
        ) = self(x_vis, x_spc, x_spc_diff, mask, is_train=True)

        # reconstruct loss of vis
        lrc_x_vis = self.loss_x(x_vis, recon_x_vis, mask)
        lrc_x_vis *= self.config.lrc_x_vis
        # reconstruct loss of spc
        lrc_x_spc = self.loss_x(x_spc, recon_x_spc, mask)
        lrc_x_spc *= self.config.lrc_x_spc
        # reconstruct loss of spc diff
        lrc_x_spc_diff = self.loss_x(x_spc_diff, recon_x_spc_diff, mask)
        lrc_x_spc_diff *= self.config.lrc_x_spc

        lrc = lrc_x_vis + lrc_x_spc + lrc_x_spc_diff
        lrc = lrc.sum()

        kl_discrete, kl_continuous, perplex = self.loss_latent_and_perplexity(
            z_e,
            z_q,
            precision_q,
            prob,
            log_prob,
            mean_prob,
        )

        loss_dict = dict(
            rc=lrc.item(),
            vis=lrc_x_vis.mean().item(),
            spc=lrc_x_spc.mean().item(),
            spc_diff=lrc_x_spc_diff.mean().item(),
            kl_discrete=kl_discrete,
            kl_continuous=kl_continuous,
            perplex=perplex,
            total=kl_discrete + kl_continuous + lrc,
            log_param_q=self.quantizer.log_param_q,
        )

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
        )

        return loss_dict["total"]

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, x_spc_diff, mask = batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        x_spc_diff = x_spc_diff.to(next(self.parameters()).device)
        mask = mask.to(next(self.parameters()).device)
        if x_vis.ndim == 5:
            ids = ids[0]
            x_vis = x_vis[0]
            x_spc = x_spc[0]
            x_spc_diff = x_spc_diff[0]
            mask = mask[0]

        # forward
        (
            z_e,
            z_q,
            precision_q,
            prob,
            log_prob,
            mean_prob,
            recon_x_vis,
            recon_x_spc,
            recon_x_spc_diff,
        ) = self(x_vis, x_spc, x_spc_diff, mask, is_train=False)

        mse_x_vis = self.loss_x(x_vis, recon_x_vis, mask)
        mse_x_spc = self.loss_x(x_spc, recon_x_spc, mask)
        mse_x_spc_diff = self.loss_x(x_spc_diff, recon_x_spc_diff, mask)

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i],
                "id": ids[i].cpu().numpy().item(),
                "x_vis": x_vis[i].cpu().numpy(),
                "fake_x_vis": recon_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "fake_x_spc": recon_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "fake_x_spc_diff": recon_x_spc_diff[i].cpu().numpy(),
                "mse_x_spc_diff": mse_x_spc_diff.item(),
                "z_e": z_e[i].cpu().numpy(),
                "z_q": z_q[i].cpu().numpy(),
                "label_prob": prob[i].cpu().numpy(),
                "label": prob[i].cpu().numpy().argmax().item(),
                "mask": mask[i].cpu().numpy(),
            }
            results.append(data)

        return results


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.emb = IndividualEmbedding(
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.patch_size,
            config.img_size,
        )
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.mlp = MLP(config.hidden_ndim, config.latent_ndim)

    def forward(self, x_vis, x_spc, x_spc_diff, mask):
        # embedding
        z = self.emb(x_vis, x_spc, x_spc_diff)

        # positional embedding
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        for layer in self.encoders:
            z, attn_w = layer(z, mask)
        # z (b, seq_len, hidden_ndim)

        z = self.mlp(z)
        # z (b, seq_len, latent_ndim)

        return z


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace, vis_npatchs: int):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.emb_hidden_ndim = config.emb_hidden_ndim
        self.x_vis_start = nn.Parameter(
            torch.zeros((1, 1, 17, 2), dtype=torch.float32), requires_grad=False
        )
        self.x_spc_start = nn.Parameter(
            torch.zeros((1, 1, 2, 2), dtype=torch.float32), requires_grad=False
        )

        self.emb = IndividualEmbedding(
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.patch_size,
            config.img_size,
        )
        # self.emb_y = MLP(config.n_clusters, config.hidden_ndim)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)
        self.emb_z = MLP(config.latent_ndim, config.hidden_ndim)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp = MLP(config.hidden_ndim, config.emb_hidden_ndim * 3)
        self.rec_vis = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 17 * 2),
            nn.Tanh(),
        )
        self.rec_spc = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 2 * 2),
            nn.Tanh(),
        )
        self.rec_spc_diff = nn.Sequential(
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 2 * 2),
            nn.Tanh(),
        )

    def forward(self, x_vis, x_spc, x_spc_diff, z, mask=None):
        b, seq_len = x_vis.size()[:2]
        x_vis = torch.cat([self.x_vis_start.repeat((b, 1, 1, 1)), x_vis], dim=1)
        x_vis = x_vis[:, :-1]
        x_spc = torch.cat([self.x_spc_start.repeat((b, 1, 1, 1)), x_spc], dim=1)
        x_spc = x_spc[:, :-1]
        x_spc_diff = torch.cat(
            [self.x_spc_start.repeat((b, 1, 1, 1)), x_spc_diff], dim=1
        )
        x_spc_diff = x_spc_diff[:, :-1]
        x = self.emb(x_vis, x_spc, x_spc_diff)
        mask = torch.cat([torch.full((b, 1), False).to(mask.device), mask], dim=1)
        mask = mask[:, :-1]
        # x (b, seq_len, hidden_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        z = self.emb_z(z)
        for layer in self.decoders:
            x = layer(x, z, mask)
        # x (b, seq_len, hidden_ndim)

        # reconstruct
        x = self.mlp(x)
        fake_x_vis, fake_x_spc, fake_x_spc_diff = (
            x[:, :, : self.emb_hidden_ndim],
            x[:, :, self.emb_hidden_ndim : self.emb_hidden_ndim * 2],
            x[:, :, self.emb_hidden_ndim * 2 :],
        )
        # fake_x_vis, fake_x_spc (b, seq_len, emb_hidden_ndim)

        # reconstruct x_vis
        fake_x_vis = self.rec_vis(fake_x_vis)
        fake_x_vis = fake_x_vis.view(b, seq_len, 17, 2)

        # reconstruct x_spc
        fake_x_spc = self.rec_spc(fake_x_spc)
        fake_x_spc = fake_x_spc.view(b, seq_len, 2, 2)

        # reconstruct x_spc
        fake_x_spc_diff = self.rec_spc_diff(fake_x_spc_diff)
        fake_x_spc_diff = fake_x_spc_diff.view(b, seq_len, 2, 2)

        return fake_x_vis, fake_x_spc, fake_x_spc_diff


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace, log_param_q=np.log(10.0)):
        super().__init__()
        self.book_size = config.book_size
        self.book_dim = config.latent_ndim
        self.book = nn.Parameter(torch.randn(config.book_size, config.latent_ndim))
        self.temperature = None
        self.log_param_q = nn.Parameter(torch.tensor(log_param_q))

    def calc_distance(self, z, book):
        z = z.view(-1, self.book_dim)
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

    def forward(self, z, is_train):
        shape, dims = torch.tensor(z.shape), z.dim()
        permute_dims = [0, *range(2, dims), 1]
        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        z = z.permute(permute_dims).contiguous()
        logits = -self.calc_distance(z, self.book) * precision_q
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)

        if is_train:
            encodings = self.gumbel_softmax_relaxation(logits)
            z_q = torch.mm(encodings, self.book).view(list(shape[permute_dims]))
            mean_prob = torch.mean(prob.detach(), dim=0)
        else:
            indices = torch.argmax(logits, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                indices.shape[0], self.book_size, device=indices.device
            )
            encodings.scatter_(1, indices, 1)
            z_q = torch.mm(encodings, self.book).view(list(shape[permute_dims]))
            mean_prob = torch.mean(encodings, dim=0)

        z_q = z_q.permute(0, -1, *range(1, dims - 1)).contiguous()

        return z_q, precision_q, prob, log_prob, mean_prob
