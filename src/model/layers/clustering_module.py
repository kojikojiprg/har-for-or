import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign


class ClusteringModule(nn.Module):
    def __init__(self, cfg, n_samples, n_samples_batch):
        super().__init__()
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch

        # get configs
        self._img_w = cfg.img_size.w
        self._img_h = cfg.img_size.h
        i3d_nch = cfg.autoencoder.i3d.nch
        i3d_seq_len = cfg.autoencoder.i3d.seq_len
        cfg = cfg.clustering
        self._n_clusters = cfg.n_clusters
        self._t_alpha = cfg.alpha
        os = cfg.roialign.output_size

        # layers for visual feature
        self._avg_pool = nn.AvgPool3d((i3d_seq_len, 1, 1))
        self._roi_align = RoIAlign(
            os,
            cfg.roialign.spatial_scale,
            1,
            cfg.roialign.aligned,
        )
        self._emb_visual = nn.Linear(i3d_nch * os * os, cfg.ndf)

        # layer for spatial feature
        self._add_spc_feature = cfg.add_spacial_feature
        if self._add_spc_feature:
            self._emb_spacial = nn.Linear(1, cfg.ndf)
            self._attn_vis_spc = nn.MultiheadAttention(cfg.ndf, 4, 0.1, batch_first=True)
            self._attn_spc_vis = nn.MultiheadAttention(cfg.ndf, 4, 0.1, batch_first=True)

        # centroids
        # z_vis = torch.normal(0, 0.1, (cfg.n_clusters, 480 * os * os))
        z_vis = torch.randn((cfg.n_clusters, i3d_nch * os * os))
        z_vis = self._emb_visual(z_vis)
        if self._add_spc_feature:
            z_spc = torch.rand((cfg.n_clusters, 1))
            z_spc = self._emb_spacial(z_spc)
            z = z_vis + z_spc
            # z = self._feature_fusion(z_vis, z_spc)
        else:
            z = z_vis
        self._centroids = nn.ParameterList(
            [
                nn.Parameter(z[i], requires_grad=True)
                for i in range(self._n_clusters)
            ]
        )
        self._target_distribution = torch.zeros((self._n_samples, self._n_clusters))

    @property
    def centroids(self):
        return self._centroids

    @property
    def target_distribution(self):
        return self._target_distribution

    def _feature_fusion(self, vis, spc):
        attn_vis = self._attn_vis_spc(vis, spc, spc, need_weights=False)[0]
        # attn_spc = self._attn_vis_spc(spc, vis, vis, need_weights=False)[0]
        # return attn_vis + attn_spc
        return attn_vis

    def forward(self, z_vis, bboxs, norms):
        # visual feature
        fy, fx = z_vis.shape[3:5]
        bn, sn = bboxs.shape[0:2]
        bboxs_vis = bboxs.view(-1, 2, 2).clone()
        bboxs_vis *= torch.Tensor((fx / self._img_w, fy / self._img_h)).to(
            next(self.parameters()).device
        )
        bboxs_vis = bboxs_vis.view(bn, -1, 4)
        rois = self._convert_bboxes_to_roi_format(bboxs_vis)

        z_vis = self._avg_pool(z_vis)[:, :, 0]
        z_vis = self._roi_align(z_vis, rois)
        z_vis = z_vis.view(bn * sn, -1)
        z_vis = self._emb_visual(z_vis)
        z_vis = z_vis.view(bn, sn, -1)

        # clustering
        s = torch.zeros((bn, sn, self._n_clusters))
        bboxs = bboxs.cpu().numpy()
        z_all = torch.zeros_like(z_vis)
        for b in range(bn):
            mask_not_nan = ~np.isnan(bboxs[b]).any(axis=1)

            # spacial feature
            if self._add_spc_feature:
                norms = norms.view(bn, sn, 1)
                norms = norms[b][mask_not_nan]
                z_spc = self._emb_spacial(norms)

                # merge feature
                z = z_vis[b][mask_not_nan] * 4 + z_spc
                # z = self._feature_fusion(z_vis[b][mask_not_nan], z_spc)
            else:
                z = z_vis[b][mask_not_nan]

            # student T
            s[b, : z.shape[0]] = self._student_t(z)

            z_all[b, : z.shape[0]] = z.detach()

        # calc cluster
        c = s.argmax(dim=2)

        return z_all, s, c

    def _convert_bboxes_to_roi_format(self, boxes: torch.Tensor) -> torch.Tensor:
        concat_boxes = torch.cat([b for b in boxes], dim=0)
        temp = []
        for i, b in enumerate(boxes):
            temp.append(torch.full_like(b[:, :1], i))
        ids = torch.cat(temp, dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        rois = rois.nan_to_num(0)
        return rois

    def _student_t(self, z):
        sn = z.shape[0]
        norm = torch.zeros((sn, self._n_clusters), dtype=torch.float32)
        for j in range(self._n_clusters):
            diff = (z - self._centroids[j]).clone()
            norm[:, j] = torch.linalg.vector_norm(diff, dim=1)

        s = torch.zeros((sn, self._n_clusters), dtype=torch.float32)
        s_tmp = torch.zeros_like(s)
        for i in range(sn):
            s_tmp[i] = (1 + norm[i] / self._t_alpha) ** -((self._t_alpha + 1) / 2)
        s_tmp_sum = s_tmp.sum(dim=1)
        s = s_tmp / s_tmp_sum.view(-1, 1)

        return s

    def update_target_distribution(self, s, batch_idxs):
        sn = s.shape[1]

        for b, batch_idx in enumerate(batch_idxs):
            s_sums = s[b].sum(dim=0)
            tmp = torch.zeros((sn, self._n_clusters))
            for i in range(sn):
                for j in range(self._n_clusters):
                    sij = s[b, i, j]
                    tmp[i, j] = sij**2 / s_sums[j]

            tmp_sums = tmp.sum(dim=1)
            for i in range(sn):
                targets = tmp[i] / tmp_sums[i]

                ti = batch_idx * self._n_samples_batch + i  # target idx
                self._target_distribution[ti] = targets
