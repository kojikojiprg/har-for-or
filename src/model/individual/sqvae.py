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

EDGE_INDEX = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),  # Body
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]


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
        self.n_clusters = config.n_clusters

        if not config.mask_leg:
            self.n_pts = (17 + 2) * 2
        else:  # mask ankles and knees
            self.n_pts = (17 - 4 + 2) * 2

        self.encoder = None
        self.decoder = None
        self.quantizer = None
        self.clustering = None

        self.annotation_path = annotation_path

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(self.config)
        self.decoders = nn.ModuleList([Decoder(self.config) for _ in range(self.n_pts)])
        self.quantizer = GaussianVectorQuantizer(self.config)
        self.clustering = ClusteringModule(self.config, self.n_pts)

        if self.annotation_path is not None:
            anns = np.loadtxt(self.annotation_path, str, delimiter=" ", skiprows=1)
            self.annotations = anns

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda epoch: self.config.lr_lmd**epoch
        )
        return [opt], [sch]

    def forward(self, x_vis, x_spc, quantizer_is_train):
        # x_vis (b, seq_len, 17, 2)
        # x_spc (b, seq_len, 2, 2)

        # encoding
        ze = self.encoder(x_vis, x_spc)
        # ze (b, npts, latent_ndim)

        # quantization
        zq, precision_q, prob, log_prob = self.quantizer(ze, quantizer_is_train)
        # zq (b, npts, latent_ndim)

        # reconstruction
        b, seq_len = x_vis.size()[:2]
        x_vis = x_vis.view(b, seq_len, 17 * 2)
        recon_x_vis = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.decoders[: 17 * 2]):
            recon_x = decoder(x_vis[:, :, i], zq[:, i, :])
            recon_x_vis = torch.cat([recon_x_vis, recon_x], dim=2)
        recon_x_vis = recon_x_vis.view(b, seq_len, 17, 2)

        x_spc = x_spc.view(b, seq_len, 2 * 2)
        recon_x_spc = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.decoders[17 * 2 :]):
            recon_x = decoder(x_spc[:, :, i], zq[:, i, :])
            recon_x_spc = torch.cat([recon_x_spc, recon_x], dim=2)
        recon_x_spc = recon_x_spc.view(b, seq_len, 2, 2)

        # clustering
        c_prob = self.clustering.sample_c_prob(prob)
        zq_prior_prob = self.clustering.sample_zq_prob(c_prob)
        # psuedo_labels_prob = self.clustering.sample_psuedo_labels(prob)
        psuedo_labels_prob = torch.full_like(c_prob, 1 / self.n_clusters)

        return (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
            zq_prior_prob,
            c_prob,
            psuedo_labels_prob,
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

        # mse loss of edge
        npt = x.size()[2]
        if npt == 2:  # bbox
            edge = x[:, :, 1] - x[:, :, 0]
            recon_edge = recon_x[:, :, 1] - recon_x[:, :, 0]
            mses = mses + self.mse_x(edge, recon_edge)
        else:  # kps
            for i, j in EDGE_INDEX:
                edge = x[:, :, j] - x[:, :, i]
                recon_edge = recon_x[:, :, j] - recon_x[:, :, i]
                mses = mses + self.mse_x(edge, recon_edge)

        # mse loss of seq
        diff = x[:, 1:] - x[:, :-1]
        recon_diff = recon_x[:, 1:] - recon_x[:, :-1]
        mses = mses + self.mse_x(diff, recon_diff)

        return mses.mean()

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

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature()
        self.quantizer.temperature = temp_cur

        # forward
        (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
            zq_prior_prob,
            c_prob,
            psuedo_labels_prob,
        ) = self(x_vis, x_spc, self.current_epoch < self.config.pre_epochs)

        # ELBO loss
        lrc_x_vis = self.loss_x(x_vis, recon_x_vis)
        lrc_x_spc = self.loss_x(x_spc, recon_x_spc)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(prob, log_prob)

        # clustering loss
        keys = ["{}_{}".format(*key.split("_")[0::2]) for key in keys]
        for i, key in enumerate(keys):
            if key in self.annotations.T[0]:
                label = self.annotations.T[1][key == self.annotations.T[0]]
                psuedo_labels_prob[i] = F.one_hot(
                    torch.tensor(int(label)), self.n_clusters
                ).to(self.device, torch.float32)

        mask_supervised = np.isin(keys, self.annotations.T[0]).ravel()
        mask_supervised = torch.tensor(mask_supervised).to(self.device)
        lc = F.cross_entropy(c_prob, psuedo_labels_prob, reduction="none")
        lc = lc * (mask_supervised * 1.0) + lc * (~mask_supervised * 0.01)
        lc = lc.mean()

        lzq_prior = F.cross_entropy(zq_prior_prob, prob)

        if self.current_epoch < self.config.pre_epochs:
            loss_total = (
                (lrc_x_vis + lrc_x_spc) * self.config.lmd_lrc_pre
                + kl_continuous * self.config.lmd_klc_pre
                + kl_discrete * self.config.lmd_kld_pre
                + (lc + lzq_prior) * self.config.lmd_c_pre
            )
        else:
            loss_total = (
                (lrc_x_vis + lrc_x_spc) * self.config.lmd_lrc
                + kl_continuous * self.config.lmd_klc
                + kl_discrete * self.config.lmd_kld
                + (lc + lzq_prior) * self.config.lmd_c
            )

        loss_dict = dict(
            x_vis=lrc_x_vis.item(),
            x_spc=lrc_x_spc.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            log_param_q=self.quantizer.log_param_q.item(),
            c=lc.item(),
            lzq=lzq_prior.item(),
            total=loss_total.item(),
        )

        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        # mask = mask.to(next(self.parameters()).device)
        if x_vis.ndim == 5:
            ids = ids[0]
            x_vis = x_vis[0]
            x_spc = x_spc[0]
            # mask = mask[0]

        # forward
        (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
            zq_prior_prob,
            c_prob,
            psuedo_label_prob,
        ) = self(x_vis, x_spc, False)

        mse_x_vis = self.mse_x(x_vis, recon_x_vis)
        mse_x_spc = self.mse_x(x_spc, recon_x_spc)

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
                "book_prob": prob[i].cpu().numpy(),
                "book_idx": prob[i].cpu().numpy().argmax(axis=1),
                "label_prob": c_prob[i].cpu().numpy(),
                "label": c_prob[i].cpu().numpy().argmax(),
            }
            results.append(data)

        return results


class Embedding(nn.Module):
    def __init__(self, seq_len, hidden_ndim, latent_ndim):
        super().__init__()
        self.hidden_ndim = hidden_ndim
        # self.mlp_x = MLP(config.seq_len, config.hidden_ndim)
        # self.mlp_y = MLP(config.seq_len, config.hidden_ndim)

        self.conv_x = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ndim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, hidden_ndim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, latent_ndim, 1),
            nn.SiLU(),
        )
        self.conv_y = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ndim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, hidden_ndim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, latent_ndim, 1),
            nn.SiLU(),
        )

    def forward(self, x):
        # x (b, seq_len, n_pts, 2)
        # x, y = x[:, :, :, 0], x[:, :, :, 1]
        # x = self.mlp_x(x.permute(0, 2, 1))  # (b, n_pts, hidden_ndim)
        # y = self.mlp_y(y.permute(0, 2, 1))  # (b, n_pts, hidden_ndim)
        # x = torch.cat([x, y], dim=1)  # (b, n_pts * 2, hidden_ndim)

        b, _, n_pts = x.size()[:3]
        x, y = x[:, :, :, 0], x[:, :, :, 1]
        x = self.conv_x(x)  # (b, latent_ndim, n_pts)
        y = self.conv_y(y)  # (b, latent_ndim, n_pts)
        x = torch.cat([x, y], dim=2)
        x = x.permute(0, 2, 1)
        # x (b, n_pts * 2, latent_ndim)

        return x


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.emb_vis = Embedding(config.seq_len, config.hidden_ndim, config.latent_ndim)
        self.emb_spc = Embedding(config.seq_len, config.hidden_ndim, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

    def forward(self, x_vis, x_spc, mask=None):
        # x_vis (b, seq_len, 17, 2)
        # x_spc (b, seq_len, 2, 2)

        # embedding
        x_vis = self.emb_vis(x_vis)  # (b, 17 * 2, latent_ndim)
        x_spc = self.emb_spc(x_spc)  # (b, 2 * 2, latent_ndim)
        z = torch.cat([x_vis, x_spc], dim=1)
        # z (b, (17 + 2) * 2, latent_ndim)

        # positional embedding
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        for layer in self.encoders:
            z, attn_w = layer(z, mask)
        # z (b, (17 + 2) * 2, latent_ndim)

        return z


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace, vis_npatchs: int = None):
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
        # z (b, n_pts, latent_ndim)
        b, n_pts, latent_ndim = ze.size()

        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        ze = ze.permute(0, 2, 1).contiguous()
        logits = -self.calc_distance(ze, self.book) * precision_q
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)

        if is_train:
            encodings = self.gumbel_softmax_relaxation(logits)
            zq = torch.mm(encodings, self.book).view(b, latent_ndim, n_pts)
            # mean_prob = torch.mean(prob.detach(), dim=0)
        else:
            indices = torch.argmax(logits, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                indices.shape[0], self.book_size, device=indices.device
            )
            encodings.scatter_(1, indices, 1)
            zq = torch.mm(encodings, self.book).view(b, latent_ndim, n_pts)
            # mean_prob = torch.mean(encodings, dim=0)

        zq = zq.permute(0, 2, 1).contiguous()
        logits = logits.view(b, n_pts, self.book_size)
        prob = prob.view(b, n_pts, self.book_size)
        log_prob = log_prob.view(b, n_pts, self.book_size)

        return zq, precision_q, prob, log_prob


class ClusteringModule(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.n_clusters = config.n_clusters
        self.latent_ndim = config.latent_ndim
        self.book_size = config.book_size
        self.n_pts = n_pts

        # self.emb_ze = nn.Embedding(config.book_size, config.hidden_ndim)
        # self.ze_to_c = nn.Sequential(
        #     MLP(config.hidden_ndim, config.hidden_ndim),
        #     nn.SiLU(),
        #     MLP(config.hidden_ndim, config.n_clusters),
        # )
        self.ze_to_c = ClusteringTransformer(config)
        self.c_to_zq = nn.Sequential(
            MLP(config.n_clusters, config.book_size * n_pts),
            nn.SiLU(),
            MLP(config.book_size * n_pts, config.book_size * n_pts),
        )

    def sample_c_prob(self, ze_prob):
        # ze_prob (b, npts, book_size)
        # indices = ze_prob.argmax(dim=-1)  # (b, npts)
        # emb = self.emb_ze(indices)  # (b, npts, hidden_ndim)
        # logits = self.ze_to_c(emb)  # (b, n_clusters)
        # prob = F.softmax(logits, dim=-1)
        prob = self.ze_to_c(ze_prob)
        return prob

    def sample_zq_prob(self, c_prob):
        logits = self.c_to_zq(c_prob)
        logits = logits.view(-1, self.n_pts, self.book_size)  # (b, n_pts, book_size)
        prob = F.softmax(logits, dim=-1)
        return prob

    def sample_zq(self, c_prob, book):
        # c_prob (b, n_clusters)
        # book (book_size, latent_ndim)
        prob = self.sample_zq_prob(c_prob)

        # select zq
        b = c_prob.size(0)
        book = book.detach()
        indices = torch.argmax(prob, dim=1).unsqueeze(1)
        encodings = torch.zeros(b, self.book_size, device=indices.device)
        encodings.scatter_(1, indices, 1)
        zq = torch.mm(encodings, book).view(b, self.latent_ndim, self.n_pts)

        return zq

    def sample_psuedo_labels(self, ze_prob):
        # ze_prob (b, npts, book_size)
        b, n_pts, book_size = ze_prob.size()

        # sample logits of zq prior
        labels = torch.arange(self.n_clusters)
        labels = F.one_hot(labels, self.n_clusters).to(ze_prob.device, torch.float32)
        zq_prior_prob = self.sample_zq_prob(labels)
        zq_prior_prob = zq_prior_prob.view(1, self.n_clusters, n_pts * book_size)
        zq_prior_prob = zq_prior_prob.repeat(b, 1, 1)

        # calc cos sim between ze_prob and zq_prior_prob
        ze_prob = ze_prob.view(b, n_pts, book_size)
        ze_prob = ze_prob.view(b, 1, n_pts * book_size)
        ze_prob = ze_prob.repeat(1, self.n_clusters, 1)

        cos_sim = F.cosine_similarity(ze_prob, zq_prior_prob, dim=2)

        return cos_sim  # (b,)


class ClusteringTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn((1, 1, config.latent_ndim)))

        self.emb_ze = nn.Embedding(config.book_size, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.encoder = TransformerEncoderBlock(
            config.latent_ndim, config.nheads, config.dropout
        )
        self.out = nn.Sequential(
            MLP(config.latent_ndim, config.latent_ndim),
            nn.SiLU(),
            MLP(config.latent_ndim, config.n_clusters),
            nn.Softmax(dim=-1),
        )

    def forward(self, ze_prob):
        # ze_prob (b, npts, book_size)
        indices = ze_prob.argmax(dim=-1)  # (b, npts)
        emb = self.emb_ze(indices)  # (b, npts, latent_ndim)

        cls_token = self.cls_token.repeat(emb.size(0), 1, 1)
        emb = torch.concat([cls_token, emb], dim=1)

        emb = self.pe.rotate_queries_or_keys(emb, seq_dim=1)
        z, attn_w = self.encoder(emb)

        c_prob = self.out(z[:, 0, :])
        return c_prob
