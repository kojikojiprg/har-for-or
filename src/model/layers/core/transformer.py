import torch.nn as nn

from .feadfoward import FeedForward


class TransformerEncoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(ndim, eps=1e-5)
        self.self_attn = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(ndim, eps=1e-5)
        self.ff = FeedForward(ndim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x (b * seq_len, npatch, ndim)
        x = self.norm1(x)
        x = x + self.self_attention_block(x)

        x = self.norm2(x)
        x = x + self.feed_forward_block(x)

        return x

    def self_attention_block(self, x):
        x = self.self_attn(x, x, x)[0]
        x = self.dropout1(x)
        return x

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout2(x)
        return x
