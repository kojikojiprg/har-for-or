import torch
import torch.nn as nn


class ClusteringModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        # get configs
        self._n_clusters = config.n_clusters
        self.latent_ndim = config.latent_ndim
        self._t_alpha = config.alpha

        # centroids
        y = torch.randn((config.n_clusters, config.latent_ndim))
        self._centroids = nn.ParameterList(
            [nn.Parameter(y[i], requires_grad=True) for i in range(self._n_clusters)]
        )

    @property
    def centroids(self):
        return self._centroids

    def forward(self, y):
        s = self._student_t(y)
        c = s.argmax(dim=1)

        return s, c

    def _student_t(self, y):
        # y (b, latent_ndim)
        b = y.shape[0]
        norm = torch.zeros((b, self._n_clusters), dtype=torch.float32)
        for j in range(self._n_clusters):
            diff = (y - self._centroids[j]).clone()
            norm[:, j] = torch.linalg.vector_norm(diff, dim=1)

        s = torch.zeros((b, self._n_clusters), dtype=torch.float32)
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
        tmp = torch.zeros((b, self._n_clusters))
        for i in range(b):
            for j in range(self._n_clusters):
                sij = s[i, j]
                tmp[i, j] = sij**2 / s_sums[j]

        tmp_sums = tmp.sum(dim=1)
        for i in range(b):
            targets = tmp[i] / tmp_sums[i]
            target_distribution[i] = targets

        return target_distribution
