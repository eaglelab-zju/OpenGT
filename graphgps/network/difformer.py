import torch
import torch.nn as nn
from graphgps.layer.difformer_layer import DIFFormerConv


from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, GeneralLayer, GCNConv, Linear
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.feature_encoder import FeatureEncoder

@register_network("DIFFormer")
class DIFFormer(nn.Module):
    '''
    DIFFormer model. Adapted from https://github.com/qitianwu/DIFFormer
    
    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.layers: Number of DIFFormer layers.
            - cfg.gt.dim_hidden: Hidden dimension for GNN layers and DIFFormer layers.
            - cfg.gnn.head: Type of head to use for the final output layer.
    
    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.edge_index (torch.Tensor): Edge indices of the graph.
    
    Output:
        batch (task dependent type, see output head): Output after model processing.
    '''
    def __init__(self, dim_in, dim_out):
        super(DIFFormer, self).__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.pre_mp = GeneralLayer('linear', new_layer_config(dim_in = dim_in, dim_out = cfg.gt.dim_hidden, num_layers = 1, has_act = True, has_bias = True, cfg = cfg))

        convlist = [(lambda x: x, 'x -> x_0')]

        for i in range(cfg.gt.layers):
            convlist.append((DIFFormerConv(cfg.gt.dim_hidden, cfg.gt.dim_hidden, config=cfg.gt), 'x, x_0 -> x'))
        
        self.convs = Sequential('x', convlist)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)
    
    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
