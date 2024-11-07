from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from numpy.typing import NDArray

from src.model.individual.modules import (
    ClassificationHead,
    Decoder,
    Encoder,
    GaussianVectorQuantizer,
    gumbel_softmax_relaxation,
)


class SQVAE(LightningModule):
    def __init__(self, config: SimpleNamespace, annotations: Optional[NDArray] = None):
        super().__init__()
        self.config = config
        self.latent_ndim = config.latent_ndim

        self.n_clusters = config.n_clusters
        self.temp_cls_init = config.temp_cls_init
        self.temp_cls_decay = config.temp_cls_decay
        self.temp_cls_min = config.temp_cls_min
        self.temperature_cls = None

        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min
        self.temperature = None

        self.cls_head = None
        self.encoder = None
        self.decoder = None
        self.quantizer = None
        self.annotations = annotations

    def configure_model(self):
        if self.encoder is not None:
            return
        self.cls_head = ClassificationHead(self.config)
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantizer = GaussianVectorQuantizer(self.config)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.config.lr, self.config.betas)
        # sch = torch.optim.lr_scheduler.ExponentialLR(opt, self.config.lr_gamma)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, self.config.t_max, self.config.lr_min
        )
        return [opt], [sch]

    def forward(self, kps, bbox, is_train):
        # kps (b, seq_len, n_pts, 2)
        # bbox (b, seq_len, n_pts, 2)

        # classification
        c_logits, attn_w_cls = self.cls_head(kps, bbox, is_train)
        if is_train:
            c_probs = torch.zeros_like(c_logits)
            for i, c_logit in enumerate(c_logits):
                c_prob = gumbel_softmax_relaxation(c_logit, self.temperature_cls)
                c_probs[i] = c_prob
        else:
            c_probs = F.softmax(c_logits, dim=-1)
        # c_prob (b, n_clusters)

        # encoding
        ze, attn_w = self.encoder(kps, bbox, c_probs, is_train)
        # ze (b, npts, latent_ndim)

        # quantization
        zq, precision_q, prob, log_prob = self.quantizer(
            ze, c_probs, self.temperature, is_train
        )
        # zq (b, npts, latent_ndim)
        # prob (b, npts, book_size)

        # reconstruction
        recon_kps, recon_bbox = self.decoder(kps, bbox, zq)

        if is_train:
            c_probs = F.softmax(c_logits, dim=-1)
        return (
            ze,
            zq,
            attn_w,
            attn_w_cls,
            precision_q,
            prob,
            log_prob,
            recon_kps,
            recon_bbox,
            c_probs,
        )

    def calc_temperature(self, temp_init, temp_decay, temp_min):
        return np.max(
            [
                temp_init * np.exp(-temp_decay * self.global_step),
                temp_min,
            ]
        )

    def mse_x(self, x, recon_x, reduction):
        mse = F.mse_loss(recon_x, x, reduction="none")
        if reduction == "mean":
            return mse.mean(dim=(1, 2, 3))  # (b,)
        elif reduction == "sum":
            return mse.sum(dim=(1, 2, 3))  # (b,)
        else:
            return mse  # (b,)

    def loss_x(self, x, recon_x):
        mses = self.mse_x(x, recon_x, "sum")
        return mses.mean()

    def loss_kl_continuous(self, ze, zq, precision_q):
        return torch.sum(((ze - zq) ** 2) * precision_q, dim=(1, 2)).mean()

    def loss_kl_discrete(self, prob, log_prob):
        return torch.sum(prob * log_prob, dim=(0, 1)).mean()

    def process_batch(self, batch):
        keys, ids, kps, bbox, mask = batch
        keys = np.array(keys).ravel()
        if kps.device != self.device:
            kps = kps.to(self.device)
            bbox = bbox.to(self.device)
        if kps.ndim == 5:
            ids = ids[0]
            kps = kps[0]
            bbox = bbox[0]
            # mask = mask[0]

        return keys, ids, kps, bbox, mask

    def training_step(self, batch, batch_idx):
        keys, ids, kps, bbox, mask = self.process_batch(batch)

        # update temperature of gumbel softmax
        self.temperature_cls = self.calc_temperature(
            self.temp_cls_init, self.temp_cls_decay, self.temp_cls_min
        )
        self.temperature = self.calc_temperature(
            self.temp_init, self.temp_decay, self.temp_min
        )

        # forward
        (
            ze,
            zq,
            attn_w,
            attn_w_cls,
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
            log_param_q_cls=self.cls_head.log_param_q_cls.item(),
        )

        # clustering loss
        c_prior = torch.full_like(c_prob, 1 / self.n_clusters)
        c_prob = torch.clamp(c_prob, min=1e-10)
        # lc_psuedo = (c_prob * (c_prob.log() - c_prior.log())).mean()
        lc_psuedo = F.kl_div(c_prob.log(), c_prior)
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
        keys, ids, kps, bbox, mask = self.process_batch(batch)

        # forward
        (
            ze,
            zq,
            attn_w,
            attn_w_cls,
            precision_q,
            prob,
            log_prob,
            recon_kps,
            recon_bbox,
            c_prob,
        ) = self(kps, bbox, False)

        mse_kps = self.mse_x(kps, recon_kps, "mean")
        mse_bbox = self.mse_x(bbox, recon_bbox, "mean")

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i],
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
                "attn_w_cls": attn_w_cls[i].cpu().numpy(),
                "book_prob": prob[i].cpu().numpy(),
                "book_idx": prob[i].cpu().numpy().argmax(axis=1),
                "label_prob": c_prob[i].cpu().numpy(),
                "label": c_prob[i].cpu().numpy().argmax(),
            }
            results.append(data)

        return results
