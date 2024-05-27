import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from src.model.layers import IndividualEmbedding, TransformerEncoderBlock


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        data_type: str,
        ndim: int,
        nheads: int,
        nlayers: int,
        emb_hidden_ndim: int,
        emb_nheads: int,
        emb_nlayers: int,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        emb_add_position_patch: bool = True,
        **kwargs
    ):
        super().__init__()
        self.emb = IndividualEmbedding(
            data_type,
            emb_hidden_ndim,
            ndim,
            emb_nheads,
            emb_nlayers,
            emb_dropout,
            emb_add_position_patch,
            **kwargs
        )
        self.pe = RotaryEmbedding(ndim)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(ndim, nheads, dropout) for _ in range(nlayers)]
        )

    def forward(self, x, bbox=None):
        x = self.emb(x, bbox)
        # x (b, seq_len, ndim)
        x = self.pe.rotate_queries_or_keys(x)
        for layer in self.layers:
            x = layer(x)
        # x (b, seq_len, ndim)
        return x
