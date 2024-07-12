import numpy as np
import torch
import torch.nn as nn


class ClusteringModule(nn.Module):
    def __init__(self, config, init_batch=None, init_model=None):
        super().__init__()
        self.n_clusters = config.n_clusters
        self.latent_ndim = config.latent_ndim
        self._t_alpha = config.alpha

        # centroids
        if init_model is not None:
            self._centroids = nn.ParameterList()
            self.init_centroids(init_batch, init_model)
            del init_batch  # release memory
        else:
            self._centroids = nn.ParameterList(
                [torch.zeros(self.latent_ndim) for i in range(self.n_clusters)]
            )

    @property
    def centroids(self):
        return self._centroids

    def init_centroids(self, init_batch, model, seed=42):
        np.random.seed(seed)
        x_vis, x_spc, mask = init_batch
        b = x_vis.size(0)
        selected_idxs = []
        with torch.no_grad():
            z = model(x_vis, x_spc, mask)

            # first centroid
            r = np.random.choice(np.arange(z.size()[0]))
            selected_idxs.append(r)
            self._centroids.append(z[r].detach())

            # other centroids
            for i in range(1, self.n_clusters):
                c = torch.cat(list(self._centroids))
                dist = (
                    z.view(b, self.latent_ndim, 1) - c.T.view(1, self.latent_ndim, i)
                ) ** 2
                dist = dist.sum(dim=1)
                min_idx = dist.argmin(dim=1)
                for j in range(dist.shape[1]):
                    dist.T[j][min_idx != j] = 0
                prob = (dist.sum(dim=1) / dist.sum()).cpu().numpy()
                r = np.random.choice(np.arange(z.size()[0]), p=prob)
                self._centroids.append(z[r].detach())

    def forward(self, z):
        s = self._student_t(z)
        c = s.argmax(dim=1)

        return s, c

    def _student_t(self, z):
        # y (b, latent_ndim)
        b = z.shape[0]
        norm = torch.zeros((b, self.n_clusters), dtype=torch.float32)
        for j in range(self.n_clusters):
            diff = (z - self._centroids[j]).clone()
            norm[:, j] = torch.linalg.vector_norm(diff, dim=1)

        s = torch.zeros((b, self.n_clusters), dtype=torch.float32)
        s_tmp = torch.zeros_like(s)
        for i in range(b):
            s_tmp[i] = (1 + norm[i] / self._t_alpha) ** -((self._t_alpha + 1) / 2)
        s_tmp_sum = s_tmp.sum(dim=1)
        s = s_tmp / s_tmp_sum.view(-1, 1)
        # s (b, n_clusters)

        return s

    def update_target_distribution(self, s):
        # s (b, n_clusters)
        b = s.shape[0]
        target_distribution = torch.zeros(s.size()).to(s.device)

        s_sums = s.sum(dim=0)
        tmp = torch.zeros((b, self.n_clusters))
        for i in range(b):
            for j in range(self.n_clusters):
                sij = s[i, j]
                tmp[i, j] = sij**2 / s_sums[j]

        tmp_sums = tmp.sum(dim=1)
        for i in range(b):
            targets = tmp[i] / tmp_sums[i]
            target_distribution[i] = targets

        return target_distribution
