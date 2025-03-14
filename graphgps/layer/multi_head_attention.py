import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_layer

@register_layer('MultiHeadAttention')
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_hidden: int, n_heads: int, dropout: float, **kwargs):
        super().__init__()
        self.model = nn.MultiheadAttention(
            dim_hidden,
            n_heads,
            dropout,
        )

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch, batch, batch, need_weights=False)[0]
        else:
            batch.x = self.model(batch.x, batch.x, batch.x, need_weights=False)[0]
        return batch
