import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, in_ndim: int, out_ndim: int = None):
        super().__init__()

        if out_ndim is None:
            out_ndim = in_ndim
        hdim = int(in_ndim * 4 * (2 / 3))
        self.w1 = nn.Linear(in_ndim, hdim)
        self.w2 = nn.Linear(hdim, out_ndim)
        self.w3 = nn.Linear(in_ndim, hdim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        hidden_dim: int = None,
        act_layer: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = int(in_dim * 4 * (2 / 3))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)
