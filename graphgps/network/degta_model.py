
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.utils import to_dense_adj

from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from graphgps.layer.degta_layer import DeGTAConv
from graphgps.encoder.feature_encoder import FeatureEncoder

@register_network('DeGTA')
class DeGTA(torch.nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int
                 ):

        super().__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        convs = []
        for _ in range(cfg.gt.layers):
            convs.append((DeGTAConv(dim_in), 'x -> x'))
        
        self.convs = Sequential('x', convs)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
