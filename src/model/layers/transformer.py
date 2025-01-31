import torch
import torch.nn as nn
import torch.nn.functional as F

from .feedforward import MLP


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        nheads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % nheads == 0, "dim should be divisible by num_heads"
        self.num_heads = nheads
        self.head_dim = dim // nheads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )

        x = x.transpose(1, 2).reshape(b, n, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, nheads, dropout):
        super().__init__()
        self.nheads = nheads
        self.attn = nn.MultiheadAttention(
            dim, nheads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        self.ff = MLP(dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, need_weights=False):
        # x (b * seq_len, npatch, dim)
        x_attn, attn_w = self.attention_block(x, need_weights)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self.feed_forward_block(x))

        return x, attn_w

    def attention_block(self, x, need_weights):
        x, attn_w = self.attn(x, x, x, need_weights=need_weights)
        return x, attn_w

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout2(x)
        return x


def create_src_mask(mask, b, seq_len, nheads):
    mask = (
        mask.view(b, 1, seq_len)
        .repeat((1, nheads, seq_len))
        .view(b * nheads, seq_len, seq_len)
    ).detach()
    return mask


def create_tgt_mask(mask, b, seq_len, nheads, device=None):
    if mask is not None:
        mask = create_src_mask(mask, b, seq_len, nheads)
        subsequent_mask = ~torch.tril(torch.full((seq_len, seq_len), True)).to(
            mask.device
        )
        mask = mask + subsequent_mask
    else:
        mask = ~torch.tril(torch.full((seq_len, seq_len), True)).to(device)
    return mask
