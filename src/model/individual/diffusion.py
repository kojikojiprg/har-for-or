from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from rotary_embedding_torch import RotaryEmbedding
from tqdm import tqdm

from src.model.individual.modules import get_n_pts
from src.model.layers.dit import DiTBlock, FinalLayer

from .csqvae import CSQVAE


class Diffusion(LightningModule):
    def __init__(self, config: SimpleNamespace, csqvae_checkpoint_path: str):
        super().__init__()
        self.config = config
        self.npts = get_n_pts(config.csqvae)
        self.csqvae_chkp = csqvae_checkpoint_path

        self.latent_dim = config.csqvae.latent_dim
        self.noise_steps = config.noise_steps
        self.beta = torch.linspace(config.beta_start, config.beta_end, self.noise_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.model = None
        self.csqvae = None

    def init_weights(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.apply(_weights_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.model.emb_c.weight, 0)
        nn.init.constant_(self.model.emb_c.bias, 0)
        for block in self.model.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.model.fin.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.model.fin.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.model.fin.linear.weight, 0)
        nn.init.constant_(self.model.fin.linear.bias, 0)

    def configure_model(self):
        if self.model is not None:
            return
        self.model = DiffusionModel(self.config)
        self.init_weights()

        # load csqvae params
        self.csqvae = CSQVAE(self.config.csqvae)
        self.csqvae.configure_model()
        ckpt = torch.load(self.csqvae_chkp, map_location="cpu")
        self.csqvae.load_state_dict(ckpt["state_dict"])
        self.csqvae.requires_grad_(False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.config.lr, self.config.betas)
        return opt

    def send_sigma_to_device(self, device):
        if device != self.beta.device:
            self.beta = self.beta.to(device)
            self.alpha = self.alpha.to(device)
            self.alpha_hat = self.alpha_hat.to(device)
            # self.gamma_hat = self.gamma_hat.to(device)

    def to(self, device):
        self.send_sigma_to_device(device)
        return super().to(device)

    def forward(self, z, t, c):
        return self.model(z, t, c)

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,))

    def sample_noise(self, z, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(z)
        return sqrt_alpha_hat * z + sqrt_one_minus_alpha_hat * eps, eps

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

        return keys, ids, kps, bbox, mask

    def training_step(self, batch, batch_idx):
        keys, ids, kps, bbox, mask = self.process_batch(batch)
        z, attn_w = self.csqvae.encoder(kps, bbox, True)
        c_logits, attn_w_cls = self.csqvae.cls_head(z, True)

        c_probs = c_logits.softmax(-1)
        # zq, precision_q, logits = self.csqvae.quantizer(
        #     z, c_probs.detach(), None, False
        # )

        # training step of diffusion
        t = self.sample_timesteps(z.size(0)).to(self.device)
        z_t, noise = self.sample_noise(z, t)
        pred_noise = self(z_t, t, c_probs)

        # samplig z_prior
        t1 = torch.ones_like(t)
        z_1, noise_1 = self.sample_noise(z, t1)
        pred_noise_1 = self(z_1, t1, c_probs)
        beta = self.beta[t1][:, None, None]
        alpha = self.alpha[t1][:, None, None]
        alpha_hat = self.alpha_hat[t1][:, None, None]
        z_prior = (z_1 - beta / torch.sqrt(1 - alpha_hat) * pred_noise_1) / torch.sqrt(
            alpha
        )

        b = len(keys)
        # z_prior = z_prior.view(b, -1, self.latent_dim)
        # logits_prior = self.csqvae.quantizer(
        #     z_prior.view(-1, self.latent_dim), c_probs, None, False
        # )

        # loss
        # kl_q = self.loss_kl_logits(logits, logits_prior)
        lrc_z = F.mse_loss(z, z_prior, reduction="sum") / b
        ldt = F.mse_loss(pred_noise, noise, reduction="sum") / b

        # loss_total = kl_q + lrc_z + ldt
        loss_total = lrc_z + ldt
        loss_dict = dict(
            # klq=kl_q.item(),
            z=lrc_z.item(),
            ldt=ldt.item(),
            total=loss_total.item(),
        )
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss_total

    def loss_kl_logits(self, logits, logits_prior, eps=1e-10):
        q = logits.softmax(dim=-1)
        q_log = logits.log_softmax(dim=-1)
        p_log = logits_prior.log_softmax(dim=-1)
        kl = torch.sum(q * (q_log - p_log), dim=(1, 2)).mean()

        return kl

    @torch.no_grad()
    def sample(self, c_probs):
        self.send_sigma_to_device(c_probs.device)

        b = c_probs.size(0)

        z = torch.randn((b, self.npts[0] * self.npts[1], self.latent_dim))
        z = z.to(c_probs.device)

        for i in tqdm(list(reversed(range(1, self.noise_steps)))):
            t = torch.full((b,), i).long().to(c_probs.device)
            pred_noise = self(z, t, c_probs)

            beta = self.beta[t][:, None, None]
            alpha = self.alpha[t][:, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None]
            if i > 1:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = (z - beta / torch.sqrt(1 - alpha_hat) * pred_noise) / torch.sqrt(
                alpha
            ) + torch.sqrt(beta) * noise

        return z


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.latent_dim_diffusion
        self.latent_dim = config.csqvae.latent_dim

        self.emb_x = nn.Linear(self.latent_dim, self.dim)
        self.rotary_emb = RotaryEmbedding(self.dim)
        self.emb_c = nn.Linear(config.csqvae.n_clusters, self.dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(self.dim, config.n_heads_dit) for _ in range(config.n_blocks_dit)]
        )
        self.fin = FinalLayer(self.dim, self.latent_dim)

        self.emb_c.requires_grad_(False)

    def pos_encoding(self, t):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.to(t.device)

    def forward(self, x, t, c):
        x = self.emb_x(x)
        x = self.rotary_emb.rotate_queries_or_keys(x, seq_dim=1)

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t)

        c = self.emb_c(c)
        c = c + t

        for block in self.blocks:
            x = block(x, c)
        x = self.fin(x, c)

        return x
