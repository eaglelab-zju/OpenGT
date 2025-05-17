import torch
import torch.nn as nn
from graphgps.layer.difformer_layer import DIFFormerConv


from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, GeneralLayer, GCNConv, Linear
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.feature_encoder import FeatureEncoder

@register_network("Graphtransformer")
class Graphtransformer(nn.Module):
    '''
    Graphtransformer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, dim_in, dim_out):
        super(Graphtransformer, self).__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        convlist = []

        for i in range(cfg.gt.layers):
            convlist.append((GeneralLayer(cfg.gt.layer_type.split('+')[0].lower()+'conv', new_layer_config(dim_in = cfg.gt.dim_hidden, dim_out = cfg.gt.dim_hidden, has_bias = True, has_act = False, num_layers = cfg.gnn.layers, cfg = cfg)), 'x -> x'))
        
        self.convs = Sequential('x', convlist)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)
    
    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
