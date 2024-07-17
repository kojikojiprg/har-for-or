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
    def __init__(
        self, config: SimpleNamespace, clustering_init_batch=None, pretrain=False
    ):
        super().__init__()
        self.automatic_optimization = False
        self.pretrain = pretrain
        self.config = config
        self.accumulate_grad_batches = config.accumulate_grad_batches
        self.update_discriminator = config.update_discriminator
        self.update_clustering = config.update_clustering
        self.update_kmeans = config.update_kmeans
        self.n_clusters = config.n_clusters
        self.latent_ndim = config.latent_ndim

        self.Ge = Encoder(config)
        self.Gd = Decoder(config)
        self.D = Discriminator(config)
        if clustering_init_batch is None:
            self.C = ClusteringModule(config)
        else:
            self.C = ClusteringModule(config, clustering_init_batch, self.Ge)
        if pretrain:
            self.C.requires_grad_(False)

    def load_state_dict_without_clustering(self, state_dict):
        ge_params = {
            name.replace("Ge.", ""): params
            for name, params in state_dict.items()
            if "Ge" in name
        }
        self.Ge.load_state_dict(ge_params)
        gd_params = {
            name.replace("Gd.", ""): params
            for name, params in state_dict.items()
            if "Gd" in name
        }
        self.Gd.load_state_dict(gd_params)
        d_params = {
            name.replace("D.", ""): params
            for name, params in state_dict.items()
            if "D" in name
        }
        self.D.load_state_dict(d_params)

    @staticmethod
    def loss_adv(pred, label):
        return F.binary_cross_entropy_with_logits(pred, label)

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

    def loss_cross_entropy(self, s, z):
        psuedo_labels = self.C.predict_kmeans(z).to(s.device)
        lce = F.cross_entropy(s, psuedo_labels)
        return lce

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis[0].detach()
        x_spc = x_spc[0].detach()
        mask = mask[0].detach()

        opt_g, opt_d, opt_c = self.optimizers()

        # train generator (autoencoder)
        self.toggle_optimizer(opt_g)
        z = self.Ge(x_vis, x_spc, mask)
        fake_x_vis, fake_x_spc = self.Gd(x_vis, x_spc, z, mask)
        c_fake = self.D(fake_x_vis, fake_x_spc)

        # loss generator
        l_vis = self.loss_x_vis(x_vis, fake_x_vis, mask)
        l_vis = l_vis / self.accumulate_grad_batches
        self.log("vis", l_vis, prog_bar=True, on_step=True, on_epoch=True)
        l_spc = self.loss_x_spc(x_spc, fake_x_spc, mask)
        l_spc = l_spc / self.accumulate_grad_batches
        self.log("spc", l_spc, prog_bar=True, on_step=True, on_epoch=True)
        lr = l_vis + l_spc
        label = torch.ones_like(c_fake)
        lg = self.loss_adv(c_fake, label)
        lg = lg / self.accumulate_grad_batches
        self.log("g", lg, prog_bar=True, on_step=True, on_epoch=True)
        self.manual_backward(lr + lg)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt_g.step()
            opt_g.zero_grad(set_to_none=True)
        self.untoggle_optimizer(opt_g)

        # train discriminator
        if (self.current_epoch + 1) % self.update_discriminator == 0:
            self.toggle_optimizer(opt_d)
            # fake sample
            z = self.Ge(x_vis, x_spc, mask)
            fake_x_vis, fake_x_spc = self.Gd(x_vis, x_spc, z, mask)
            c_fake = self.D(fake_x_vis.detach(), fake_x_spc.detach())
            label_fake = torch.zeros_like(c_fake)
            # true sample
            c_true = self.D(x_vis, x_spc, mask)
            label_true = torch.ones_like(c_true)

            # loss discriminator
            ld_fake = self.loss_adv(c_fake, label_fake)
            ld_true = self.loss_adv(c_true, label_true)
            ld = (ld_fake + ld_true) / 2
            ld = ld / self.accumulate_grad_batches
            self.log("d", ld, prog_bar=True, on_step=True, on_epoch=True)
            self.manual_backward(ld)
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                opt_d.step()
                opt_d.zero_grad(set_to_none=True)
            self.untoggle_optimizer(opt_d)

        # train clustering module and encoder
        if not self.pretrain and (self.current_epoch + 1) % self.update_clustering == 0:
            self.toggle_optimizer(opt_c)
            z = self.Ge(x_vis, x_spc, mask)
            s, _ = self.C(z)
            t = self.C.update_target_distribution(s)
            lc = self.loss_kl_clustering(s, t)
            # lc = lc + self.loss_cross_entropy(s, z)
            lc = lc / self.accumulate_grad_batches
            self.log("c", lc, prog_bar=True, on_step=True, on_epoch=True)
            self.manual_backward(lc)
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                opt_c.step()
                opt_c.zero_grad(set_to_none=True)
            self.untoggle_optimizer(opt_c)
        # if not self.pretrain and (self.current_epoch + 1) % self.update_kmeans == 0:
        #     self.C.update_kmeans(self.Ge)

        del keys, ids, x_vis, x_spc, mask

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

        z = self.Ge(x_vis, x_spc, mask)
        fake_x_vis, fake_x_spc = self.Gd(x_vis, x_spc, z, mask)
        s, c = self.C(z)

        mse_x_vis = self.loss_x_vis(x_vis, fake_x_vis, mask)
        mse_x_spc = self.loss_x_spc(x_spc, fake_x_spc, mask)

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i][0],
                "id": ids[i].cpu().numpy().item(),
                "x_vis": x_vis[i].cpu().numpy(),
                "fake_x_vis": fake_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "fake_x_spc": fake_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "z": z[i].cpu().numpy(),
                "label_prob": s[i].cpu().numpy(),
                "label": c[i].cpu().numpy(),
                "mask": mask[i].cpu().numpy(),
            }
            results.append(data)
        return results

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.Ge.parameters()) + list(self.Gd.parameters()), lr=self.config.lr_g
        )
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_d)
        opt_c = torch.optim.Adam(
            list(self.Ge.parameters()) + list(self.C.parameters()), lr=self.config.lr_c
        )
        return [opt_g, opt_d, opt_c], []


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters

        self.cls = nn.Parameter(torch.randn((1, 1, self.hidden_ndim)))
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
        self.lin_cls = nn.Linear(config.hidden_ndim, config.latent_ndim)

    def forward(self, x_vis, x_spc, mask):
        # embedding
        b, seq_len = x_vis.size()[:2]
        x = self.emb(x_vis, x_spc)

        # concat y to x
        x = torch.cat([self.cls.repeat((b, 1, 1)), x], dim=1)
        mask = torch.cat([torch.full((b, 1), False).to(mask.device), mask], dim=1)

        # positional embedding
        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        # x (b, seq_len+1, hidden_ndim)
        for layer in self.encoders:
            x, attn_w = layer(x, mask)
        # x (b, seq_len+1, hidden_ndim)

        z = x[:, 0, :]
        z = self.lin_cls(z).view(b, self.latent_ndim)
        # y (b, latent_ndim)

        return z


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.seq_len = config.seq_len
        self.hidden_ndim = config.hidden_ndim
        self.emb_hidden_ndim = config.emb_hidden_ndim
        self.x_vis_start = nn.Parameter(torch.zeros((1, 1, 17, 2), dtype=torch.float32), requires_grad=False)
        self.x_spc_start = nn.Parameter(torch.zeros((1, 1, 2, 2), dtype=torch.float32), requires_grad=False)

        self.emb = IndividualEmbedding(
            config.emb_hidden_ndim,
            config.hidden_ndim,
            config.emb_nheads,
            config.emb_nlayers,
            config.emb_dropout,
            config.patch_size,
            config.img_size,
        )
        self.emb_z = MLP(config.latent_ndim, config.seq_len * config.hidden_ndim)
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

    def forward(self, x_vis, x_spc, z, mask=None):
        b, seq_len = x_vis.size()[:2]
        x_vis = torch.cat([self.x_vis_start.repeat((b, 1, 1, 1)), x_vis], dim=1)
        x_vis = x_vis[:, :-1]
        x_spc = torch.cat([self.x_spc_start.repeat((b, 1, 1, 1)), x_spc], dim=1)
        x_spc = x_spc[:, :-1]
        x = self.emb(x_vis, x_spc)
        mask = torch.cat([torch.full((b, 1), False).to(mask.device), mask], dim=1)
        mask = mask[:, :-1]
        # x (b, seq_len + 1, hidden_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        z = self.emb_z(z)
        z = z.view(b, self.seq_len, self.hidden_ndim)
        for layer in self.decoders:
            x = layer(x, z, mask)
        # x (b, seq_len, hidden_ndim)

        # reconstruct
        x = self.mlp(x)
        fake_x_vis, fake_x_spc = (
            x[:, :, : self.emb_hidden_ndim],
            x[:, :, self.emb_hidden_ndim :],
        )
        # fake_x_vis, fake_x_spc (b, seq_len, emb_hidden_ndim)

        # reconstruct x_vis
        fake_x_vis = self.rec_vis(fake_x_vis)
        fake_x_vis = fake_x_vis.view(b, seq_len, 17, 2)

        # reconstruct x_spc
        fake_x_spc = self.rec_spc(fake_x_spc)
        fake_x_spc = fake_x_spc.view(b, seq_len, 2, 2)

        return fake_x_vis, fake_x_spc


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
            MLP(config.hidden_ndim, config.hidden_ndim),
            nn.Linear(config.hidden_ndim, 1),
            nn.Sigmoid(),  # real, fake
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
