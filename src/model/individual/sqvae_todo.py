from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from numpy.typing import NDArray

# TODO
from src.model.individual.modules_todo import (
    ClassificationHead,
    Decoder,
    Encoder,
    GaussianVectorQuantizer,
    gumbel_softmax_sample,
)


class CSQVAE(LightningModule):
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
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, self.config.lr_gamma)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt, self.config.t_max, self.config.lr_min
        # )
        return [opt], [sch]
        # return opt

    def forward(self, kps, bbox, is_train):
        # kps (b, seq_len, n_pts, 2)
        # bbox (b, seq_len, n_pts, 2)

        # encoding
        ze, attn_w = self.encoder(kps, bbox, is_train)
        # ze (b, npts, latent_ndim)

        # quantization
        zq, precision_q, logits = self.quantizer(ze, self.temperature, is_train)
        # zq (b, npts, latent_ndim)
        # prob (b, npts, book_size)

        # classification
        c_logits, attn_w_cls = self.cls_head(zq, is_train)
        if is_train:
            c_probs = gumbel_softmax_sample(c_logits, self.temperature_cls)
        else:
            c_probs = F.softmax(c_logits, dim=-1)
        # c_prob (b, n_clusters)

        # reconstruction
        recon_kps, recon_bbox = self.decoder(zq)

        # sampling
        logits_sampled = self.quantizer.sample_logits_from_c(c_probs)

        return (
            recon_kps,
            recon_bbox,
            ze,
            zq,
            precision_q,
            logits,
            logits_sampled,
            c_logits,
            attn_w,
            attn_w_cls,
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

    def loss_kl_discrete(self, logits, logits_sampled):
        kl_discrete = F.kl_div(
            logits.log_softmax(dim=-1),
            logits_sampled.softmax(dim=-1),
            reduction="batchmean",
        )
        return kl_discrete

    def loss_c_elbo(self, c_logits):
        prob = F.softmax(c_logits, dim=-1)
        log_prob = F.log_softmax(c_logits, dim=-1)
        lc_elbo = torch.sum(prob * log_prob, dim=1).mean()
        return lc_elbo

    def loss_c_real(self, c_logits, keys, ids):
        c_prob = F.softmax(c_logits, dim=-1)
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
            else:
                lc_real = torch.Tensor([0.0]).to(self.device)
        else:
            lc_real = torch.Tensor([0.0]).to(self.device)

        return lc_real

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
            recon_kps,
            recon_bbox,
            ze,
            zq,
            precision_q,
            logits,
            logits_sampled,
            c_logits,
            attn_w,
            attn_w_cls,
        ) = self(kps, bbox, True)

        # ELBO loss
        lrc_kps = self.loss_x(kps, recon_kps)
        lrc_bbox = self.loss_x(bbox, recon_bbox)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(logits, logits_sampled)

        # clustering loss
        lc_elbo = self.loss_c_elbo(c_logits)
        lc_real = self.loss_c_real(c_logits, keys, ids)

        loss_total = (
            (lrc_kps + lrc_bbox) * self.config.lmd_lrc
            + kl_continuous * self.config.lmd_klc
            + kl_discrete * self.config.lmd_kld
            + lc_elbo * self.config.lmd_c_elbo
            + lc_real * self.config.lmd_c_real
        )

        loss_dict = dict(
            kps=lrc_kps.item(),
            bbox=lrc_bbox.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            log_param_q=self.quantizer.log_param_q.item(),
            log_param_q_cls=self.cls_head.log_param_q_cls.item(),
            c_elbo=lc_elbo.item(),
            c_real=lc_real.item(),
            total=loss_total.item(),
        )
        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    @torch.no_grad()
    def predict_step(self, batch):
        keys, ids, kps, bbox, mask = self.process_batch(batch)

        # forward
        (
            recon_kps,
            recon_bbox,
            ze,
            zq,
            precision_q,
            logits,
            logits_sampled,
            c_logits,
            attn_w,
            attn_w_cls,
        ) = self(kps, bbox, False)

        prob = logits.softmax(dim=-1)
        c_prob = c_logits.softmax(dim=-1)

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
