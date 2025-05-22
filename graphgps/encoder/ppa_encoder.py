import torch
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)


@register_node_encoder('PPANode')
class PPANodeEncoder(torch.nn.Module):
    """
    Uniform input node embedding for PPA that has no node features.

    Parameters:
        emb_dim (int): The dimension of the output node features.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.encoder = torch.nn.Embedding(1, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch


@register_edge_encoder('PPAEdge')
class PPAEdgeEncoder(torch.nn.Module):
    """
    Edge encoder for PPA that applies a linear transformation to the edge features.
    
    Parameters:
        emb_dim (int): The dimension of the output edge features.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch
