from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import (
    MLP,
    ClusteringModule,
    IndividualEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class GAN(LightningModule):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.accumulate_grad_batches = config.accumulate_grad_batches
        self.n_clusters = config.n_clusters
        self.latent_ndim = config.latent_ndim

        self.E = Encoder(config)
        self.G = Generator(config)
        self.D = Discriminator(config)
        self.C = ClusteringModule(config)

    @staticmethod
    def adversarial_loss(y, fake_y):
        return F.cross_entropy(y, fake_y)

    @staticmethod
    def loss_x_vis(x_vis, fake_x_vis, mask):
        diff = torch.diff(x_vis[~mask], dim=0)
        fake_diff = torch.diff(fake_x_vis[~mask], dim=0)
        return F.mse_loss(x_vis[~mask], fake_x_vis[~mask]) + F.mse_loss(diff, fake_diff)

    @staticmethod
    def loss_x_spc(x_spc, fake_x_spc, mask):
        diff = torch.diff(x_spc[~mask], dim=0)
        fake_diff = torch.diff(fake_x_spc[~mask], dim=0)
        return F.mse_loss(x_spc[~mask], fake_x_spc[~mask]) + F.mse_loss(diff, fake_diff)

    @staticmethod
    def loss_kl_clustering(q, p, eps=1e-10):
        lc = (q * (torch.log(q + eps) - torch.log(p + eps))).sum()
        return lc

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis[0].detach()
        x_spc = x_spc[0].detach()
        mask = mask[0].detach()

        opt_c, opt_ae, opt_g, opt_d = self.optimizers()

        # train clustering module and encoder
        self.toggle_optimizer(opt_c)
        z, y = self.E(x_vis, x_spc, mask)
        s, _ = self.C(y)
        t = self.C.update_target_distribution(s)
        lc = self.loss_kl_clustering(s, t)
        lc = lc / self.accumulate_grad_batches
        self.log("c", lc, prog_bar=True)
        self.manual_backward(lc)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt_c.step()
            opt_c.zero_grad()
        self.untoggle_optimizer(opt_c)

        # train autoencoder (encoder + generator)
        self.toggle_optimizer(opt_ae)
        z, y = self.E(x_vis, x_spc, mask)
        fake_x_vis, fake_x_spc = self.G(z, y, mask)
        lr = self.loss_x_vis(x_vis, fake_x_vis, mask) + self.loss_x_spc(
            x_spc, fake_x_spc, mask
        )
        lr = lr / self.accumulate_grad_batches
        self.log("r", lr, prog_bar=True)
        self.manual_backward(lr)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt_ae.step()
            opt_ae.zero_grad()
        self.untoggle_optimizer(opt_ae)

        # create noise from clustering centroids
        b, seq_len = x_vis.size()[:2]
        c_random_sample = torch.randint(0, self.n_clusters, (b,))
        y_noise = torch.randn((b, self.latent_ndim)).to(x_vis.device)
        for i in c_random_sample:
            y_noise[i] = y_noise[i] + self.C.centroids[i].detach()
        z_noise = torch.randn((b, seq_len, self.latent_ndim)).to(x_vis.device)
        c_noise = (
            F.one_hot(c_random_sample, self.n_clusters + 1)
            .to(torch.float32)
            .to(x_vis.device)
        )

        # train generator
        self.toggle_optimizer(opt_g)
        fake_x_vis, fake_x_spc = self.G(z_noise, y_noise)
        c_fake = self.D(fake_x_vis, fake_x_spc)
        lg = self.adversarial_loss(c_fake, c_noise)
        lg = lg / self.accumulate_grad_batches
        self.log("g", lg, prog_bar=True)
        self.manual_backward(lg)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt_g.step()
            opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # train discriminator
        self.toggle_optimizer(opt_d)
        # fake sample
        fake_x_vis, fake_x_spc = self.G(z_noise, y_noise)
        c_fake = self.D(fake_x_vis.detach(), fake_x_spc.detach())
        label_fake = (
            F.one_hot(torch.full((b,), self.n_clusters))
            .to(torch.float32)
            .to(x_vis.device)
        )
        ld_fake = self.adversarial_loss(c_fake, label_fake)

        # true sample
        z, y = self.E(x_vis, x_spc, mask)
        _, c = self.C(y)
        c_true = self.D(x_vis, x_spc, mask)
        label_true = (
            F.one_hot(c, self.n_clusters + 1).to(torch.float32).to(x_vis.device)
        )
        ld_true = self.adversarial_loss(c_true, label_true)

        ld = (ld_fake + ld_true) / 2
        ld = ld / self.accumulate_grad_batches
        self.log("d", ld, prog_bar=True)
        self.manual_backward(ld)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt_d.step()
            opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

    def configure_optimizers(self):
        opt_c = torch.optim.Adam(
            list(self.E.parameters()) + list(self.C.parameters()), lr=self.config.lr_c
        )
        opt_ae = torch.optim.Adam(
            list(self.E.parameters()) + list(self.G.parameters()), lr=self.config.lr_ae
        )
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.config.lr_g)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_d)

        return [opt_c, opt_ae, opt_g, opt_d], []


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters

        self.cls = nn.Parameter(torch.randn((1, 1, self.hidden_ndim)))
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

        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp_cls = nn.Sequential(
            nn.LayerNorm(config.hidden_ndim),
            MLP(config.hidden_ndim, config.latent_ndim),
        )
        self.mlp_z = nn.Sequential(
            nn.LayerNorm(config.hidden_ndim),
            MLP(config.hidden_ndim, config.latent_ndim),
        )

    def forward(self, x_vis, x_spc, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc)

        # concat y to x
        x = torch.cat([self.cls.repeat((b, 1, 1)), x], dim=1)
        mask = torch.cat([self.cls_mask.repeat((b, 1)), mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        # x (b, seq_len+1, hidden_ndim)
        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        y = x[:, 0, :]
        y = self.mlp_cls(y).view(b, self.latent_ndim)
        # y (b, latent_ndim)

        z = self.mlp_z(x[:, 1:, :])
        # z (b, seq_len, latent_ndim)

        return z, y


class Generator(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.emb_hidden_ndim = config.emb_hidden_ndim

        self.emb_y = nn.Sequential(
            MLP(config.latent_ndim, config.hidden_ndim),
            MLP(config.hidden_ndim, config.hidden_ndim * config.seq_len),
        )
        self.emb_z = MLP(config.latent_ndim, config.hidden_ndim)

        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=False)

        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_ndim),
            MLP(config.hidden_ndim, config.emb_hidden_ndim * 2),
        )

        self.rec_vis = nn.Sequential(
            nn.LayerNorm(config.emb_hidden_ndim),
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 17 * 2),
            nn.Tanh(),
        )

        self.rec_spc = nn.Sequential(
            nn.LayerNorm(config.emb_hidden_ndim),
            MLP(config.emb_hidden_ndim),
            nn.Linear(config.emb_hidden_ndim, 2 * 2),
            nn.Tanh(),
        )

    def forward(self, z, y, mask=None):
        b = z.size()[0]

        # embedding fake x
        y = self.emb_y(y)
        y = y.view(b, self.seq_len, self.hidden_ndim)
        y = self.pe.rotate_queries_or_keys(y, seq_dim=1, offset=1)

        z = self.emb_z(z)

        for layer in self.decoders:
            y = layer(y, z, mask)
        # fake_x (b, seq_len, hidden_ndim)

        # reconstruct
        y = self.mlp(y)
        rec_x_vis, rec_x_spc = (
            y[:, :, : self.emb_hidden_ndim],
            y[:, :, self.emb_hidden_ndim :],
        )
        # fake_x_vis, fake_x_spc (b, seq_len, emb_hidden_ndim)

        # reconstruct x_vis
        rec_x_vis = self.rec_vis(rec_x_vis)
        rec_x_vis = rec_x_vis.view(b, self.seq_len, 17, 2)

        # reconstruct x_spc
        rec_x_spc = self.rec_spc(rec_x_spc)
        rec_x_spc = rec_x_spc.view(b, self.seq_len, 2, 2)

        return rec_x_vis, rec_x_spc


class Discriminator(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.cls = nn.Parameter(torch.randn((1, 1, self.hidden_ndim)))
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

        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp_cls = nn.Sequential(
            nn.LayerNorm(config.hidden_ndim),
            MLP(config.hidden_ndim, config.hidden_ndim),
            nn.Linear(config.hidden_ndim, config.n_clusters + 1),  # n_clusters + fake
        )

    def forward(self, x_vis, x_spc, mask=None):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc)

        # concat y to x
        x = torch.cat([self.cls.repeat((b, 1, 1)), x], dim=1)
        if mask is not None:
            mask = torch.cat([self.cls_mask.repeat((b, 1)), mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        # x (b, seq_len+1, hidden_ndim)
        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        y = x[:, 0, :]
        y = self.mlp_cls(y)
        # y (b, n_clusters)

        return y
