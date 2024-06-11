import torch.nn as nn

from .feedforward import FeedForward


class TransformerEncoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(ndim, eps=1e-5)
        self.attn = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(ndim, eps=1e-5)
        self.ff = FeedForward(ndim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, need_weights=False):
        # x (b * seq_len, npatch, ndim)
        x = self.norm1(x)
        x_attn, attn_w = self.attention_block(x, mask, need_weights)
        x = x + x_attn

        x = self.norm2(x)
        x = x + self.feed_forward_block(x)

        return x, attn_w

    def attention_block(self, x, mask, need_weights):
        x, attn_w = self.attn(x, x, x, mask, need_weights)
        x = self.dropout1(x)
        return x, attn_w

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout2(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(ndim, eps=1e-5)
        self.attn1 = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2x = nn.LayerNorm(ndim, eps=1e-5)
        self.norm2z = nn.LayerNorm(ndim, eps=1e-5)
        self.attn2 = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(ndim, eps=1e-5)
        self.ff = FeedForward(ndim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, z, mask=None):
        # x (b * seq_len, npatch, ndim)
        x = self.norm1(x)
        x = x + self.attention_block1(x, mask)

        z = self.norm2z(z)
        x = self.norm2x(x)
        x = x + self.attention_block2(x, z)

        x = self.norm3(x)
        x = x + self.feed_forward_block(x)

        return x

    def attention_block1(self, x, mask):
        x = self.attn1(x, x, x, mask, False)[0]
        x = self.dropout1(x)
        return x

    def attention_block2(self, x, z):
        x = self.attn2(x, z, z, need_weights=False)[0]
        x = self.dropout2(x)
        return x

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout3(x)
        return x
