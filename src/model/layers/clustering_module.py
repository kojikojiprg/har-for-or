import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class ClusteringModule(nn.Module):
    def __init__(self, config, init_batch=None, init_model=None, seed=42):
        super().__init__()
        self.n_clusters = config.n_clusters
        self.latent_ndim = config.latent_ndim
        self.alpha = config.alpha

        # centroids
        if init_batch is not None and init_model is not None:
            self.centroids = nn.ParameterList()
            # kmeans++ for psuedo label
            self.kmeans = KMeans(self.n_clusters, random_state=seed)
            self.init_batch = init_batch
            self.init_centroids(init_model, seed)
        else:
            self.centroids = nn.ParameterList(
                [torch.zeros(self.latent_ndim) for i in range(self.n_clusters)]
            )

    def init_centroids(self, model, seed=42):
        np.random.seed(seed)
        x_vis, x_spc, mask = self.init_batch
        b = x_vis.size(0)
        with torch.no_grad():
            z = model(x_vis, x_spc, mask)

            # first centroid
            idx = np.random.choice(np.arange(b))
            self.centroids.append(z[idx].detach())

            # other centroids
            for i in range(1, self.n_clusters):
                c = torch.cat(list(self.centroids))
                dist = (
                    z.view(b, self.latent_ndim, 1) - c.T.view(1, self.latent_ndim, i)
                ) ** 2
                dist = dist.sum(dim=1)
                min_idx = dist.argmin(dim=1)
                for j in range(dist.shape[1]):
                    dist.T[j][min_idx != j] = 0
                prob = (dist.sum(dim=1) / dist.sum()).cpu().numpy()
                idx = np.random.choice(np.arange(b), p=prob)
                self.centroids.append(z[idx].detach())

        self.kmeans = self.kmeans.fit(z.detach().cpu().numpy())

        del x_vis, x_spc, mask, z

    def update_kmeans(self, model, seed=42):
        np.random.seed(seed)
        x_vis, x_spc, mask = self.init_batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        mask = mask.to(next(self.parameters()).device)
        with torch.no_grad():
            z = model(x_vis, x_spc, mask)
        self.kmeans = self.kmeans.fit(z.detach().cpu().numpy())

    def predict_kmeans(self, z):
        labels = self.kmeans.predict(z.detach().cpu().numpy())
        return torch.tensor(labels, dtype=torch.long)

    def forward(self, z):
        s = self._student_t(z)
        c = s.argmax(dim=1)

        return s, c

    def _student_t(self, z):
        # y (b, latent_ndim)
        b = z.shape[0]
        norm = torch.zeros((b, self.n_clusters), dtype=torch.float32)
        for j in range(self.n_clusters):
            diff = (z - self.centroids[j]).clone()
            norm[:, j] = torch.linalg.vector_norm(diff, dim=1)

        s = torch.zeros((b, self.n_clusters), dtype=torch.float32)
        s_tmp = torch.zeros_like(s)
        for i in range(b):
            s_tmp[i] = (1 + norm[i] / self.alpha) ** -((self.alpha + 1) / 2)
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
