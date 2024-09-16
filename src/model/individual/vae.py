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

# from sklearn.cluster import KMeans


class VAE(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        n_batches: Optional[int] = None,
        annotation_path: Optional[str] = None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.seq_len = config.seq_len
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters
        self.accumulate_grad_batches = config.accumulate_grad_batches
        self.update_prior_interval = config.update_prior_interval
        self.tau = config.tau
        self.alpha = config.alpha
        self.batch_size = config.batch_size
        self.n_batches = n_batches

        self.Qy_x = None
        self.Qz_xy = None
        self.Py = None
        self.Pz_y = None
        self.Px_z = None

        self.mu_all = torch.empty((0, self.latent_ndim)).cpu()
        self.logvar_all = torch.empty((0, self.latent_ndim)).cpu()
        self.ids_all = []
        self.supervised_ids = []
        self.annotation_path = annotation_path

    def configure_model(self):
        if self.Qy_x is not None:
            return
        self.Qy_x = Qy_x(self.config)
        self.Qz_xy = Qz_xy(self.config)
        # vis_npatchs = self.Qy_x.emb.emb.npatchs
        self.Py = Py(self.config)
        self.Pz_y = Pz_y(self.config)
        self.Px_z = nn.ModuleList([Px_z(self.config) for _ in range((17 + 2) * 2)])

        if self.annotation_path is not None:
            anns = np.loadtxt(self.annotation_path, int, delimiter=" ", skiprows=1)
            self.annotations = torch.tensor(anns, dtype=torch.long)

    def loss_x(self, x, recon_x, mask=None):
        b = x.size(0)
        mses = torch.empty((0,)).to(self.device)
        for i in range(b):
            # mse = F.mse_loss(recon_x[i][~mask[i]], x[i][~mask[i]])
            mse = F.mse_loss(recon_x[i], x[i])
            mses = torch.cat([mses, mse.view(1, 1)])
        return mses.ravel()  # (b,)

    def loss_kl(self, q, p, weights=None, eps=1e-10):
        kl = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)
        if weights is None:
            kl = kl.mean()
        else:
            kl = torch.sum(kl * weights) / weights.sum()
        return kl

    def loss_kl_gaussian(self, mu1, logv1, mu2, logv2, weights=None):
        # mu, log (b, latent_ndim)
        lg = -0.5 * (
            1
            + logv1
            - logv2
            - logv1.exp() / logv2.exp()
            - (mu1 - mu2) ** 2 / logv2.exp()
        ).sum(dim=-1)
        if weights is None:
            lg = lg.mean()
        else:
            lg = torch.sum(lg * weights) / weights.sum()
        return lg

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        keys = np.array(keys).T[0]
        ids = ids[0]
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        # mask = mask[0]

        opt_pz_y, opt = self.optimizers()

        self.toggle_optimizer(opt)
        # VAE
        logits = self.Qy_x(x_vis, x_spc)
        y = F.gumbel_softmax(logits, self.tau, dim=1)
        pi = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, y)
        z_prior, mu_prior, logvar_prior = self.Pz_y(pi)

        b, seq_len = x_vis.size()[:2]
        x_vis = x_vis.view(b, seq_len, 17 * 2)
        recon_x_vis = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[: 17 * 2]):
            recon_x = decoder(x_vis[:, :, i], z[:, i, :])
            recon_x_vis = torch.cat([recon_x_vis, recon_x], dim=2)

        x_spc = x_spc.view(b, seq_len, 2 * 2)
        recon_x_spc = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[17 * 2 :]):
            recon_x = decoder(x_spc[:, :, i], z[:, i, :])
            recon_x_spc = torch.cat([recon_x_spc, recon_x], dim=2)

        x_vis = x_vis.view(b, seq_len, 17, 2)
        x_spc = x_spc.view(b, seq_len, 2, 2)
        recon_x_vis = recon_x_vis.view(b, seq_len, 17, 2)
        recon_x_spc = recon_x_spc.view(b, seq_len, 2, 2)

        # get augumented label
        logits_aug = self.Qy_x(recon_x_vis, recon_x_spc)
        y_aug = F.gumbel_softmax(logits_aug, self.tau, dim=1)
        pi_aug = F.softmax(logits_aug, dim=1)
        z_aug, mu_aug, logvar_aug = self.Qz_xy(recon_x_vis, recon_x_spc, y_aug)

        # calc joint probabilities of the pairwise similarities
        q_sim = self.pairwise_sim(mu, logvar)
        # p_sim = self.pairwise_sim(mu_prior, logvar_prior)

        # create psuedo labels
        idx = q_sim.argsort(descending=True)
        b = q_sim.size(0)
        weights_pl = (q_sim / q_sim.max(dim=1).values.view(1, -1)).view(b, b, 1)
        labels = (pi[idx] * weights_pl).sum(dim=1) / weights_pl.sum(dim=1)
        labels = labels.argmax(dim=1).to(torch.long)

        # obtain true labels if id ids are annotated
        supervised_mask = torch.isin(ids, self.annotations.T[0]).ravel()
        for i, _id in enumerate(ids):
            if _id in self.annotations.T[0]:
                label = self.annotations.T[1][_id == self.annotations.T[0]]
                labels[i] = label

        # loss
        logs = {}

        # ELBO
        # reconstruct loss of vis
        lrc_x_vis = self.loss_x(x_vis, recon_x_vis)
        logs["vis"] = lrc_x_vis.mean().item()

        # reconstruct loss of spc
        lrc_x_spc = self.loss_x(x_spc, recon_x_spc)
        logs["spc"] = lrc_x_spc.mean().item()

        lrc = lrc_x_vis * self.config.lrc_x_vis + lrc_x_spc * self.config.lrc_x_spc
        lrc = lrc.mean()

        # clustering loss
        lc = self.loss_kl(pi, self.Py.pi)
        logs["c"] = lc.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior)
        lg = lg
        logs["g"] = lg.item()

        loss_elbo = lrc + lc * self.config.lc + lg * self.config.lg

        # augumentation loss
        lcp = F.cross_entropy(pi, labels, reduction="none")
        lcpa = F.cross_entropy(pi_aug, labels, reduction="none")
        lc_aug = lcp + lcpa
        lc_aug[supervised_mask] = lc_aug[supervised_mask] * 1.0
        lc_aug[~supervised_mask] = lc_aug[~supervised_mask] * 0.01
        lc_aug = lc_aug.mean()
        logs["caug"] = lc_aug.item()

        lg_aug = self.loss_kl_gaussian(mu, logvar, mu_aug, logvar_aug)
        # lg_sim = self.loss_kl(q_sim, p_sim)
        # lg_aug = lg_sim + lg_aug
        logs["gaug"] = lg_aug.item()

        loss = loss_elbo + lc_aug + lg_aug
        # logs["l"] = loss.item()
        self.manual_backward(loss)
        self.log_dict(logs, prog_bar=True)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.untoggle_optimizer(opt)

        # if (
        #     self.current_epoch >= 19
        #     and self.current_epoch % self.update_prior_interval == 0
        # ):
        #     # self.discreate_pz_y(opt_pz_y)
        #     self.mu_all = torch.cat([self.mu_all, mu.cpu()], dim=0)
        #     self.logvar_all = torch.cat(
        #         [self.logvar_all, logvar.cpu()], dim=0
        #     )
        #     self.ids_all.extend(ids)
        #     if (batch_idx + 1) == self.n_batches:
        #         # kmeans
        #         # z = self.z_all_samples.detach().cpu().numpy()
        #         # self.kmeans = KMeans(self.n_clusters, random_state=True)
        #         # self.kmeans = self.kmeans.fit(z)
        #         # labels = self.kmeans.labels_
        #         # _, counts = np.unique(labels, return_counts=True)
        #         # sorted_idx = sorted(np.argsort(counts))
        #         # sort_map = {s: i for i, s in enumerate(sorted_idx)}
        #         # ids = np.array(self.all_ids).ravel()
        #         # self.psuedo_labels = [(k, sort_map[l]) for k, l in zip(keys, labels)]

        #         mu = self.mu_all.detach().cpu().numpy()
        #         logvar = self.logvar_all.detach().cpu().numpy()
        #         ids = np.array(self.ids_all).ravel()
        #         unique_ids = np.unique(ids)
        #         mu_ids = torch.empty((0, self.latent_ndim)).cpu()
        #         logvar_ids = torch.empty((0, self.latent_ndim)).cpu()
        #         for _id in unique_ids:
        #             mu_id = mu[ids == _id].mean(dim=0)
        #             n = logvar[ids == _id].size(0)
        #             logvar_id = logvar[ids == _id].sum(dim=0) - np.log(n)
        #             mu_ids = torch.cat([mu_ids, mu_id.view(1, -1)], dim=0)
        #             logvar_ids = torch.cat([logvar_ids, logvar_id.view(1, -1)], dim=0)

        #         sim_ids = self.pairwise_sim(mu_ids, logvar_ids)
        #         p_sim = self.pairwise_sim(mu_prior, logvar_prior)

        #         # init z_all_samples
        #         self.mu_all = torch.empty((0, self.latent_ndim)).cpu()
        #         self.logvar_all = torch.empty((0, self.latent_ndim)).cpu()
        #         self.ids_all = []

        del z, mu, logvar, z_prior, mu_prior, logvar_prior
        del keys, ids, x_vis, x_spc, mask

    @staticmethod
    def log_normal(z, mu, logvar):
        return -0.5 * (logvar + (z - mu) ** 2 / logvar.exp())  # + np.log(2.0 * np.pi)

    def pairwise_sim(self, mu, logvar):
        b = mu.size(0)
        pij = torch.empty((0, b, self.latent_ndim * 19 * 2), dtype=torch.float32).to(
            self.device
        )
        for i in range(b):
            pdfs = self.log_normal(mu, mu[i], logvar[i]).exp()
            pdfs = pdfs / pdfs.sum(dim=0)
            pij = torch.cat([pij, pdfs.view(1, b, self.latent_ndim * 19 * 2)], dim=0)

        # diag to zero
        pij = pij * ((1 - torch.eye(b)) + 1e-30).view(b, b, 1).to(self.device)

        pij = (pij + pij.permute(1, 0, 2)) / (2 * b)
        pij = torch.clamp_min(pij, 1e-30).sum(dim=2)
        # pij (b, b)

        return pij

    # def discreate_pz_y(self, opt_pz_y):
    #     self.toggle_optimizer(opt_pz_y)

    #     b = self.batch_size
    #     y = np.random.choice(self.n_clusters, b, p=self.Py.pi.detach().cpu().numpy())
    #     y = torch.tensor(y, dtype=torch.long)
    #     y = F.one_hot(y, self.n_clusters).to(self.device, torch.float32)
    #     # y = torch.eye(self.n_clusters, dtype=torch.float32).to(self.device)
    #     z, mu, logvar = self.Pz_y(y)

    #     mu = mu.repeat((1, self.n_clusters)).view(self.n_clusters, b, self.latent_ndim)
    #     norm = torch.linalg.norm((z - mu).permute(1, 0, 2), dim=2)
    #     pdfs = (1 + norm / self.alpha) ** (-(self.alpha + 1) / 2)
    #     pdfs = pdfs / pdfs.sum(dim=1).view(-1, 1)

    #     tij = pdfs**2 / pdfs.sum(dim=0)
    #     tij = tij / tij.sum(dim=-1).view(-1, 1)
    #     loss = -torch.clamp((pdfs * (torch.log(pdfs) - torch.log(tij))).mean(), 0, 5e-4)
    #     self.manual_backward(loss)
    #     opt_pz_y.step()
    #     opt_pz_y.zero_grad(set_to_none=True)

    #     self.log("lpzy", loss.item(), prog_bar=True)
    #     self.untoggle_optimizer(opt_pz_y)

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

        logits = self.Qy_x(x_vis, x_spc)
        y = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, y)

        b, seq_len = x_vis.size()[:2]
        x_vis = x_vis.view(b, seq_len, 17 * 2)
        recon_x_vis = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[: 17 * 2]):
            recon_x = decoder(x_vis[:, :, i], z[:, i, :])
            recon_x_vis = torch.cat([recon_x_vis, recon_x], dim=2)

        x_spc = x_spc.view(b, seq_len, 2 * 2)
        recon_x_spc = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[17 * 2 :]):
            recon_x = decoder(x_spc[:, :, i], z[:, i, :])
            recon_x_spc = torch.cat([recon_x_spc, recon_x], dim=2)

        mse_x_vis = self.loss_x(x_vis, recon_x_vis, mask)
        mse_x_spc = self.loss_x(x_spc, recon_x_spc, mask)

        x_vis = x_vis.view(b, seq_len, 17, 2)
        recon_x_vis = recon_x_vis.view(b, seq_len, 17, 2)
        x_spc = x_spc.view(b, seq_len, 2, 2)
        recon_x_spc = recon_x_spc.view(b, seq_len, 2, 2)

        results = []
        for i in range(len(keys)):
            # label_prob = self.clustering_prob(z[i], mu[i], logvar[i], mask[i])
            data = {
                "key": keys[i],
                "id": ids[i].cpu().numpy().item(),
                # "x_vis": x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                # "fake_x_vis": fake_x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                "x_vis": x_vis[i].cpu().numpy(),
                "recon_x_vis": recon_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "recon_x_spc": recon_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "z": z[i].cpu().numpy(),
                "mu": mu[i].cpu().numpy(),
                "logvar": logvar[i].cpu().numpy(),
                # "label_prob": label_prob[i].cpu().numpy(),
                # "label": label_prob[i].cpu().numpy().argmax().item(),
                "label_prob": y[i].cpu().numpy(),
                "label": y[i].cpu().numpy().argmax().item(),
                # "mask": mask[i].cpu().numpy(),
            }
            results.append(data)
        return results

    def configure_optimizers(self):
        opt_pz_y = torch.optim.Adam(self.Pz_y.parameters(), lr=self.config.lr_pz_y)  # type: ignore
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return [opt_pz_y, opt], []


class Qy_x(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.cls = nn.Parameter(torch.randn((1, 1, config.latent_ndim)))

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
        self.mlp = nn.Sequential(
            MLP(config.latent_ndim, config.n_clusters), nn.Softmax(dim=-1)
        )

    def forward(self, x_vis, x_spc, mask=None):
        # embedding
        b, seq_len = x_vis.size()[:2]

        # embedding
        x_vis = self.emb_vis(x_vis)  # (b, 17 * 2, latent_ndim)
        x_spc = self.emb_spc(x_spc)  # (b, 2 * 2, latent_ndim)
        z = torch.cat([self.cls.repeat(b, 1, 1), x_vis, x_spc], dim=1)
        # z (b, (17 + 2) * 2 + 1, latent_ndim)

        # positional embedding
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        for layer in self.encoders:
            z, attn_w = layer(z, mask)

        logits = self.mlp(z[:, 0, :])
        # (b, n_clusters)
        return logits


class Qz_xy(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.latent_ndim = config.latent_ndim

        self.emb_vis = Embedding(config.seq_len, config.hidden_ndim, config.latent_ndim)
        self.emb_spc = Embedding(config.seq_len, config.hidden_ndim, config.latent_ndim)
        self.emb_y = MLP(config.n_clusters, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.lin_mu = nn.Linear(config.latent_ndim, config.latent_ndim)
        self.lin_logvar = nn.Linear(config.latent_ndim, config.latent_ndim)

    def forward(self, x_vis, x_spc, y, mask=None):
        # embedding
        b, seq_len = x_vis.size()[:2]

        y = self.emb_y(y).view(b, 1, self.latent_ndim)
        x_vis = self.emb_vis(x_vis) + y  # (b, 17 * 2, latent_ndim)
        x_spc = self.emb_spc(x_spc) + y  # (b, 2 * 2, latent_ndim)
        z = torch.cat([x_vis, x_spc], dim=1)
        # z (b, (17 + 2) * 2, latent_ndim)

        # positional embedding
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        for layer in self.encoders:
            z, attn_w = layer(z, mask)
        # z (b, (17 + 2) * 2, latent_ndim)

        mu = self.lin_mu(z)
        logvar = self.lin_logvar(z)
        ep = torch.randn_like(logvar)
        z = mu + logvar.mul(0.5).exp() * ep
        # z, mu, log_sig (b, (17 + 2) * 2, latent_ndim)

        return z, mu, logvar


class Py(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        pi = torch.ones(config.n_clusters) / config.n_clusters
        self.pi = nn.Parameter(pi, requires_grad=False)


class Pz_y(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.seq_len = config.seq_len
        self.latent_ndim = config.latent_ndim
        self.mlp = MLP(config.n_clusters, config.latent_ndim * (17 + 2) * 2)
        self.lin_mu = nn.Linear(config.latent_ndim, config.latent_ndim)
        self.lin_logvar = nn.Linear(config.latent_ndim, config.latent_ndim)

    def forward(self, y):
        b = y.size(0)
        z = self.mlp(y).view(b, (17 + 2) * 2, self.latent_ndim)
        mu_prior = self.lin_mu(z)
        logvar_prior = self.lin_logvar(z)
        ep = torch.randn_like(logvar_prior)
        z_prior = mu_prior + logvar_prior.mul(0.5).exp() * ep

        return z_prior, mu_prior, logvar_prior


class Px_z(nn.Module):
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

    def forward(self, x, z, mask=None):
        # x (b, seq_len)
        # z (b, latent_ndim)

        b, seq_len = x.size()
        x = x.view(b, seq_len, 1)
        x = self.emb(x)  # (b, seq_len, latent_ndim)

        # concat start token
        x = torch.cat([self.x_start.repeat((b, 1, 1)), x], dim=1)
        x = x[:, :-1]  # (b, seq_len, latent_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        z = self.mlp_z(z)
        z = z.view(b, seq_len, self.latent_ndim)
        for layer in self.decoders:
            x = layer(x, z, mask)
        # x (b, seq_len, latent_ndim)

        recon_x = self.mlp(x).view(b, seq_len, 1)

        return recon_x
