from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .modules import MLP, ClassificationHead, Decoder, Encoder, get_n_pts


class VAE(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        annotation_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters
        self.tau = config.tau
        self.alpha = config.alpha
        self.batch_size = config.batch_size

        self.Qc_x = None
        self.Qz_xc = None
        self.Pc = None
        self.Pz_c = None
        self.Px_z = None

        self.mu_all = torch.empty((0, self.latent_ndim)).cpu()
        self.logvar_all = torch.empty((0, self.latent_ndim)).cpu()
        self.ids_all = []
        self.supervised_ids = []
        self.annotation_path = annotation_path

    def configure_model(self):
        if self.Qc_x is not None:
            return
        self.Qc_x = Qc_x(self.config)
        self.Qz_xc = Qz_xc(self.config)
        # vis_npatchs = self.Qy_x.emb.emb.npatchs
        self.Pz_c = Pz_c(self.config)
        self.Px_z = Decoder(self.config)

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
        keys, ids, kps, bbox, mask = batch
        keys = np.array(keys).T[0]
        ids = ids[0]
        kps = kps[0]
        bbox = bbox[0]
        # mask = mask[0]

        # VAE
        logits = self.Qc_x(kps, bbox)
        c = F.gumbel_softmax(logits, self.tau, dim=1)
        pi = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xc(kps, bbox, c)
        z_prior, mu_prior, logvar_prior = self.Pz_c(pi)

        b, seq_len = kps.size()[:2]
        kps = kps.view(b, seq_len, 17 * 2)
        recon_x_kps = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[: 17 * 2]):
            recon_x = decoder(kps[:, :, i], z[:, i, :])
            recon_x_kps = torch.cat([recon_x_kps, recon_x], dim=2)

        bbox = bbox.view(b, seq_len, 2 * 2)
        recon_x_bbox = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[17 * 2 :]):
            recon_x = decoder(bbox[:, :, i], z[:, i, :])
            recon_x_bbox = torch.cat([recon_x_bbox, recon_x], dim=2)

        kps = kps.view(b, seq_len, 17, 2)
        bbox = bbox.view(b, seq_len, 2, 2)
        recon_x_kps = recon_x_kps.view(b, seq_len, 17, 2)
        recon_x_bbox = recon_x_bbox.view(b, seq_len, 2, 2)

        # get augumented label
        logits_aug = self.Qc_x(recon_x_kps, recon_x_bbox)
        y_aug = F.gumbel_softmax(logits_aug, self.tau, dim=1)
        pi_aug = F.softmax(logits_aug, dim=1)
        z_aug, mu_aug, logvar_aug = self.Qz_xc(recon_x_kps, recon_x_bbox, y_aug)

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
        lrc_x_kps = self.loss_x(kps, recon_x_kps)
        logs["vis"] = lrc_x_kps.mean().item()

        # reconstruct loss of spc
        lrc_x_bbox = self.loss_x(bbox, recon_x_bbox)
        logs["spc"] = lrc_x_bbox.mean().item()

        lrc = lrc_x_kps * self.config.lrc_x_kps + lrc_x_bbox * self.config.lrc_x_bbox
        lrc = lrc.mean()

        # clustering loss
        lc = self.loss_kl(pi, self.Pc.pi)
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

        del z, mu, logvar, z_prior, mu_prior, logvar_prior
        del keys, ids, kps, bbox, mask

        return loss

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

    @torch.no_grad()
    def predict_step(self, batch):
        keys, ids, kps, bbox, mask = batch
        kps = kps.to(next(self.parameters()).device)
        bbox = bbox.to(next(self.parameters()).device)
        if kps.ndim == 5:
            ids = ids[0]
            kps = kps[0]
            bbox = bbox[0]

        logits = self.Qc_x(kps, bbox)
        y = F.softmax(logits, dim=1)
        z, mu, logvar = self.Qz_xc(kps, bbox, y)

        b, seq_len = kps.size()[:2]
        kps = kps.view(b, seq_len, 17 * 2)
        recon_x_kps = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[: 17 * 2]):
            recon_x = decoder(kps[:, :, i], z[:, i, :])
            recon_x_kps = torch.cat([recon_x_kps, recon_x], dim=2)

        bbox = bbox.view(b, seq_len, 2 * 2)
        recon_x_bbox = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.Px_z[17 * 2 :]):
            recon_x = decoder(bbox[:, :, i], z[:, i, :])
            recon_x_bbox = torch.cat([recon_x_bbox, recon_x], dim=2)

        mse_x_kps = self.loss_x(kps, recon_x_kps, mask)
        mse_x_bbox = self.loss_x(bbox, recon_x_bbox, mask)

        kps = kps.view(b, seq_len, 17, 2)
        recon_x_kps = recon_x_kps.view(b, seq_len, 17, 2)
        bbox = bbox.view(b, seq_len, 2, 2)
        recon_x_bbox = recon_x_bbox.view(b, seq_len, 2, 2)

        results = []
        for i in range(len(keys)):
            # label_prob = self.clustering_prob(z[i], mu[i], logvar[i], mask[i])
            data = {
                "key": keys[i],
                "id": ids[i].cpu().numpy().item(),
                # "x_kps": x_kps[0].cpu().numpy().transpose(0, 2, 3, 1),
                # "fake_x_kps": fake_x_kps[0].cpu().numpy().transpose(0, 2, 3, 1),
                "x_kps": kps[i].cpu().numpy(),
                "recon_x_kps": recon_x_kps[i].cpu().numpy(),
                "mse_x_kps": mse_x_kps.item(),
                "x_bbox": bbox[i].cpu().numpy(),
                "recon_x_bbox": recon_x_bbox[i].cpu().numpy(),
                "mse_x_bbox": mse_x_bbox.item(),
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
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return opt


class Qc_x(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.encoder = Encoder(config)
        self.cls_head = ClassificationHead(config)

    def forward(self, kps, bbox, is_train):
        z, attn_w = self.encoder(kps, bbox, is_train)
        c_logits = self.cls_head(z)
        return c_logits


class Qz_xc(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_ndim = config.latent_ndim
        self.emb_c = MLP(config.n_clusters, config.latent_ndim)
        self.encoder = Encoder(config)
        self.mlp_mu = MLP(config.latent_ndim, config.latent_ndim)
        self.mlp_logvar = MLP(config.latent_ndim, config.latent_ndim)

    def forward(self, kps, bbox, c, is_train):
        z, attn_w = self.encoder(kps, bbox, is_train)
        c = self.emb_c(c).view(c.size(0), 1, self.latent_ndim)
        z = z + c
        # z (b, (17 + 2) * 2, latent_ndim)

        mu = self.mlp_mu(z)
        logvar = self.mlp_logvar(z)
        ep = torch.randn_like(logvar)
        z = mu + logvar.mul(0.5).exp() * ep
        # z, mu, log_sig (b, (17 + 2) * 2, latent_ndim)

        return z, mu, logvar


class Pz_c(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.n_pts = get_n_pts(config)
        self.seq_len = config.seq_len
        self.latent_ndim = config.latent_ndim
        self.mlp = MLP(config.n_clusters, config.latent_ndim * self.n_pts * 2)
        self.mlp_mu = MLP(config.latent_ndim, config.latent_ndim)
        self.mlp_logvar = MLP(config.latent_ndim, config.latent_ndim)

    def forward(self, c):
        b = c.size(0)
        z = self.mlp(c).view(b, self.n_pts * 2, self.latent_ndim)

        mu_prior = self.mlp_mu(z)
        logvar_prior = self.mlp_logvar(z)
        ep = torch.randn_like(logvar_prior)
        z_prior = mu_prior + logvar_prior.mul(0.5).exp() * ep

        return z_prior, mu_prior, logvar_prior
