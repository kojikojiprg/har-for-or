import torch
import torch.nn as nn

from .feedforward import MLP


class TransformerEncoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout):
        super().__init__()
        self.nheads = nheads
        self.attn = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(ndim)

        self.ff = MLP(ndim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ndim)

    def forward(self, x, need_weights=False):
        # x (b * seq_len, npatch, ndim)
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
