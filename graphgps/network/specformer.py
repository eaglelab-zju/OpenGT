import torch
import torch.nn as nn
from graphgps.layer.spec_layer import SpecLayer
from graphgps.layer.multi_head_attention import MultiHeadAttention
from graphgps.encoder.sine_encoder import SineEncoder
from graphgps.layer.layer_norm import LayerNorm
from graphgps.encoder.feature_encoder import FeatureEncoder

from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config , MLP, GCNConv, Linear
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network


class swapex(nn.Module):
    """
    Swaps the x and EigVals attributes of the input batch. 

    Input: 
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.EigVals (torch.Tensor): Eigenvalues of the graph Laplacian.

    Output:
        batch (torch_geometric.data.Batch): Output batch with swapped x and EigVals.
            
    """
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        batch.x, batch.EigVals = batch.EigVals, batch.x
        return batch
    

@register_network("SpecFormer")
class SpecFormer(nn.Module):
    """
    SpecFormer model. Adapted from https://github.com/DSL-Lab/Specformer
    Only supports the case where the input is a batch of graphs with the same number of nodes.
    Needs preprocessing for LapRaw positional encoding.

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.dim_hidden: Hidden dimension for GNN layers and SpecFormer layers.
            - cfg.gt.n_heads: Number of attention heads.
            - cfg.gt.dropout: Dropout rate for the model.
            - cfg.gt.attn_dropout: Dropout rate for the attention mechanism.
            - cfg.gnn.head: Type of head to use for the final output layer.
    
    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.edge_index (torch.Tensor): Edge indices of the graph.
            - batch.EigVals (torch.Tensor): Eigenvalues of the graph Laplacian.
            - batch.EigVecs (torch.Tensor): Eigenvectors of the graph Laplacian.
    
    Output:
        batch (task dependent type, see output head): Output after model processing.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.pre_mp = MLP(new_layer_config(dim_in = dim_in, dim_out = cfg.gt.dim_hidden, num_layers = 2, has_act = True, has_bias = True, cfg = cfg))

        ### dirty hack
        self.swap1 = swapex()
        ### eig v
        self.eig_encoder = SineEncoder(cfg.gt.dim_hidden)

        self.mha_eig=Sequential('x',[
            (LayerNorm(cfg.gt.dim_hidden), 'x -> x1'), 
            (MultiHeadAttention(dim_hidden = cfg.gt.dim_hidden, n_heads = cfg.gt.n_heads, dropout = cfg.gt.attn_dropout), 'x1 -> x1'),
            (lambda x1, x2: self.aggregate_batches_add(x1, x2), 'x, x1 -> x')
        ])

        self.ffn_eig=Sequential('x',[
            (LayerNorm(cfg.gt.dim_hidden), 'x -> x1'), 
            (MLP(new_layer_config(dim_in = cfg.gt.dim_hidden, dim_out = cfg.gt.dim_hidden, num_layers = 2, has_act = True, has_bias = True, cfg = cfg)), 'x1-> x1'),
            (lambda x1, x2: self.aggregate_batches_add(x1, x2), 'x, x1 -> x')
        ])

        self.decoder=Linear(new_layer_config(dim_in = cfg.gt.dim_hidden, dim_out = cfg.gt.n_heads, num_layers = 0, has_act = True, has_bias = True, cfg = cfg))
        ### eig ^
        ### dirty hack
        self.swap2 = swapex()

        if cfg.gt.layer_norm:
            norm = 'layer'
        elif cfg.gt.batch_norm:
            norm = 'batch'
        else:
            norm = 'none'
        
        self.spec_layers = SpecLayer(dim_out = cfg.gt.dim_hidden, n_heads = cfg.gt.n_heads, dropout = cfg.gt.dropout, norm = norm)
        
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def aggregate_batches_add(self, x1, x2):
        new_batch = x1.clone()
        new_batch.x = x1.x + x2.x
        return new_batch

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
