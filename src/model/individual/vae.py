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
        vis_npatchs = self.Qy_x.emb.emb_vis.npatchs
        self.Py = Py(self.config)
        self.Pz_y = Pz_y(self.config)
        self.Px_z = Px_z(self.config, vis_npatchs)

        if self.annotation_path is not None:
            anns = np.loadtxt(self.annotation_path, int, delimiter=" ", skiprows=1)
            self.annotations = torch.tensor(anns, dtype=torch.long)

    def loss_x(self, x, fake_x, mask):
        b = x.size(0)
        mses = torch.empty((0,)).to(self.device)
        for i in range(b):
            mse = F.mse_loss(x[i][~mask[i]], fake_x[i][~mask[i]])
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
        keys, ids, x_vis, x_spc, x_spc_diff, mask = batch
        keys = np.array(keys).T[0]
        ids = ids[0]
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        x_spc_diff = x_spc_diff[0]
        mask = mask[0]

        opt_pz_y, opt = self.optimizers()

        self.toggle_optimizer(opt)
        # VAE
        logits = self.Qy_x(x_vis, x_spc, x_spc_diff, mask)
        y = F.gumbel_softmax(logits, self.tau, dim=1)
        pi = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, x_spc_diff, y, mask)
        z_prior, mu_prior, logvar_prior = self.Pz_y(pi)
        recon_x_vis, recon_x_spc, recon_x_spc_diff = self.Px_z(
            x_vis, x_spc, x_spc_diff, z, mask
        )

        # get augumented label
        logits_aug = self.Qy_x(recon_x_vis, recon_x_spc, recon_x_spc_diff, mask)
        y_aug = F.gumbel_softmax(logits_aug, self.tau, dim=1)
        # pi_aug = F.softmax(logits_aug, dim=1)
        z_aug, mu_aug, logvar_aug = self.Qz_xy(
            recon_x_vis, recon_x_spc, recon_x_spc_diff, y_aug, mask
        )

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
        lrc_x_vis = self.loss_x(x_vis, recon_x_vis, mask)
        lrc_x_vis *= self.config.lrc_x_vis
        logs["vis"] = lrc_x_vis.mean().item()

        # reconstruct loss of spc
        lrc_x_spc = self.loss_x(x_spc, recon_x_spc, mask)
        lrc_x_spc *= self.config.lrc_x_spc
        logs["spc"] = lrc_x_spc.mean().item()

        # reconstruct loss of spc diff
        lrc_x_spc_diff = self.loss_x(x_spc_diff, recon_x_spc_diff, mask)
        lrc_x_spc_diff *= self.config.lrc_x_spc
        logs["spcd"] = lrc_x_spc_diff.mean().item()

        lrc = lrc_x_vis + lrc_x_spc + lrc_x_spc_diff
        lrc = lrc.mean()

        # clustering loss
        lc = self.loss_kl(pi, self.Py.pi)
        lc *= self.config.lc
        logs["c"] = lc.item()

        # Gaussian loss
        lg = self.loss_kl_gaussian(mu, logvar, mu_prior, logvar_prior)
        lg = lg
        lg *= self.config.lg
        logs["g"] = lg.item()

        # loss_elbo = lrc + lc + lg
        loss_elbo = lrc + lc + lg

        # augumentation loss
        lcp = F.cross_entropy(pi, labels, reduction="none")
        # lcpa = F.cross_entropy(pi_aug, labels, reduction="none")
        # lc_aug = lcp + lcpa
        lc_aug = lcp
        # if self.current_epoch < 30:
        #     lc_aug[supervised_mask] = lc_aug[supervised_mask] * 0.99
        #     lc_aug[~supervised_mask] = lc_aug[~supervised_mask] * 0.01
        # else:
        #     lc_aug[supervised_mask] = lc_aug[supervised_mask] * 0.9
        #     lc_aug[~supervised_mask] = lc_aug[~supervised_mask] * 0.1
        lc_aug[supervised_mask] = lc_aug[supervised_mask] * 1.0
        lc_aug[~supervised_mask] = lc_aug[~supervised_mask] * 0.05
        lc_aug = lc_aug.mean()
        logs["caug"] = lc_aug.item()

        # lg_sim = self.loss_kl(q_sim, p_sim)
        # weights = torch.ones(b, dtype=torch.float32).to(self.device)
        # if self.current_epoch < 30:
        #     weights[supervised_mask] = weights[supervised_mask] * 0.99
        #     weights[~supervised_mask] = weights[~supervised_mask] * 0.01
        # else:
        #     weights[supervised_mask] = weights[supervised_mask] * 0.9
        #     weights[~supervised_mask] = weights[~supervised_mask] * 0.1
        # weights[supervised_mask] = weights[supervised_mask] * 0.99
        # weights[~supervised_mask] = weights[~supervised_mask] * 0.01
        lg_aug = self.loss_kl_gaussian(mu, logvar, mu_aug, logvar_aug)
        # lg_aug = lg_sim + lg_aug
        logs["gaug"] = lg_aug.item()

        # if self.current_epoch < 20:
        #     lmd = 0.01
        # else:
        #     lmd = 1.0
        lmd = 1.0

        loss = loss_elbo + lmd * (lc_aug + lg_aug)
        # loss = loss_elbo
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
        del keys, ids, x_vis, x_spc, x_spc_diff, mask

    @staticmethod
    def log_normal(z, mu, logvar):
        return -0.5 * (logvar + (z - mu) ** 2 / logvar.exp())  # + np.log(2.0 * np.pi)

    def pairwise_sim(self, mu, logvar):
        b = mu.size(0)
        pij = torch.empty((0, b, self.latent_ndim), dtype=torch.float32).to(self.device)
        for i in range(b):
            pdfs = self.log_normal(mu, mu[i], logvar[i]).exp()
            pdfs = pdfs / pdfs.sum(dim=0)
            pij = torch.cat([pij, pdfs.view(1, b, self.latent_ndim)], dim=0)

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

        logits = self.Qy_x(x_vis, x_spc, x_spc_diff, mask)
        y = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xy(x_vis, x_spc, x_spc_diff, y, mask)
        recon_x_vis, recon_x_spc, recon_x_spc_diff = self.Px_z(
            x_vis, x_spc, x_spc_diff, z, mask
        )
        mse_x_vis = self.loss_x(x_vis, recon_x_vis, mask)
        mse_x_spc = self.loss_x(x_spc, recon_x_spc, mask)
        mse_x_spc_diff = self.loss_x(x_spc_diff, recon_x_spc_diff, mask)

        # label_prob = self.calc_responsibility(z)

        results = []
        for i in range(len(keys)):
            # label_prob = self.clustering_prob(z[i], mu[i], logvar[i], mask[i])
            data = {
                "key": keys[i],
                "id": ids[i].cpu().numpy().item(),
                # "x_vis": x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                # "fake_x_vis": fake_x_vis[0].cpu().numpy().transpose(0, 2, 3, 1),
                "x_vis": x_vis[i].cpu().numpy(),
                "fake_x_vis": recon_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "fake_x_spc": recon_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "fake_x_spc_diff": recon_x_spc_diff[i].cpu().numpy(),
                "mse_x_spc_diff": mse_x_spc_diff.item(),
                "z": z[i].cpu().numpy(),
                "mu": mu[i].cpu().numpy(),
                "logvar": logvar[i].cpu().numpy(),
                # "label_prob": label_prob[i].cpu().numpy(),
                # "label": label_prob[i].cpu().numpy().argmax().item(),
                "label_prob": y[i].cpu().numpy(),
                "label": y[i].cpu().numpy().argmax().item(),
                "mask": mask[i].cpu().numpy(),
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

        self.cls = nn.Parameter(torch.randn((1, 1, self.hidden_ndim)))
        self.cls_mask = nn.Parameter(torch.full((1, 1), True), requires_grad=False)

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

        self.mlp = nn.Sequential(
            MLP(config.hidden_ndim, config.n_clusters),
            # nn.Softmax(dim=1),
        )

    def forward(self, x_vis, x_spc, x_spc_diff, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc, x_spc_diff)

        # concat y to x
        x = torch.cat([self.cls.repeat((b, 1, 1)), x], dim=1)
        mask = torch.cat([self.cls_mask.repeat((b, 1)), mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        # x (b, seq_len+1, hidden_ndim)
        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        logits = self.mlp(x[:, 0, :])
        # (b, n_clusters)
        return logits


class Qz_xy(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.cls_mask = nn.Parameter(torch.full((1, 1), False), requires_grad=False)

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
        self.emb_y = MLP(config.n_clusters, config.hidden_ndim)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.lin_mu = nn.Linear(config.hidden_ndim, config.latent_ndim)
        self.lin_logvar = nn.Linear(config.hidden_ndim, config.latent_ndim)

    def forward(self, x_vis, x_spc, x_spc_diff, y, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc, x_spc_diff)
        y = self.emb_y(y).view(b, 1, self.hidden_ndim)

        # concat y to x
        x = torch.cat([y, x], dim=1)
        mask = torch.cat([self.cls_mask.repeat((b, 1)), mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        mu = self.lin_mu(x[:, 0, :])
        logvar = self.lin_logvar(x[:, 0, :])
        ep = torch.randn_like(logvar)
        z = mu + logvar.mul(0.5).exp() * ep
        # z, mu, log_sig (b, latent_ndim)

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
        self.hidden_ndim = config.hidden_ndim
        self.mlp = MLP(config.n_clusters, config.hidden_ndim)
        self.lin_mu = nn.Linear(config.hidden_ndim, config.latent_ndim)
        self.lin_logvar = nn.Linear(config.hidden_ndim, config.latent_ndim)

    def forward(self, y):
        z = self.mlp(y)
        mu_prior = self.lin_mu(z)
        logvar_prior = self.lin_logvar(z)
        ep = torch.randn_like(logvar_prior)
        z_prior = mu_prior + logvar_prior.mul(0.5).exp() * ep

        return z_prior, mu_prior, logvar_prior


class Px_z(nn.Module):
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
        z = z.repeat((1, seq_len)).view(b, seq_len, self.hidden_ndim)
        z = z.view(b, self.seq_len, self.hidden_ndim)
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
