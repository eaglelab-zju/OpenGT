import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_layer

@register_layer('MultiHeadAttention')
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer wrapper for PyTorch Geometric.

    Parameters:
        dim_hidden (int): Number of input features.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    
    Input:
        batch.x (Tensor): Input node features.
    
    Output:
        batch.x (Tensor): Output node features after applying the Multi-Head Attention layer.
    """

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
