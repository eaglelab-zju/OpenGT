import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder

@register_node_encoder('WLSE')
class WLSENodeEncoder(torch.nn.Module):
    """
    Encodes the Weisfeiler-Lehman subgraph type of each node in the graph.

    Parameters:
        dim_emb (int): The dimension of the embedding.
        expand_x (bool): If True, expands the input node features by the
            embedding dimension. Default: True.
    
    """
    def __init__(self, dim_emb, expand_x = True):
        super().__init__()
        dim_in = cfg.share.dim_in

        pecfg = cfg.posenc_WLSE
        num_types = pecfg.num_types
        dim_pe = pecfg.dim_pe

        if num_types < 1:
            raise ValueError(f"Invalid 'WLSE_num_types': {num_types}")
        
        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"WLSE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=dim_pe)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        
        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Encode just the first dimension if more exist
        batch.x = torch.cat((h, self.encoder(batch.WLTag[:, 0])), dim = 1)

        return batch