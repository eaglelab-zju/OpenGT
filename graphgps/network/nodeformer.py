import torch
import torch.nn as nn
from graphgps.layer.nodeformer_layer import NodeFormerConv

from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, GeneralLayer, GCNConv, Linear
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.feature_encoder import FeatureEncoder

@register_network("NodeFormer")
class NodeFormer(nn.Module):
    '''
    NodeFormer model implementation
    return: predicted node labels, a list of edge losses at every layer
    '''
    def __init__(self, dim_in, dim_out): # does not support use_edge_loss currently
                 # cfg.gt.dim_hidden, num_layers=2, num_heads=4, dropout=0.0,
                 # kernel_transformation=softmax_kernel_transformation, nb_random_features=30, use_bn=True, use_gumbel=True,
                 # use_residual=True, use_act=False, use_jk=False, nb_gumbel_sample=10, rb_order=2, rb_trans='sigmoid', use_edge_loss=True):
        super(NodeFormer, self).__init__()

        
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gt.use_edge_loss:
            raise NotImplementedError("Edge loss is not supported")

        self.pre_mp = GeneralLayer('linear', new_layer_config(dim_in = dim_in, dim_out = cfg.gt.dim_hidden, num_layers = 1, has_act = True, has_bias = True, cfg = cfg))

        if cfg.gt.use_jk:
            convlist = [(lambda x: [x.x], 'x -> ls')]
            for i in range(cfg.gt.layers):
                convlist.append((NodeFormerConv(cfg.gt.dim_hidden, cfg.gt.dim_hidden, config=cfg.gt), 'x -> x'))
                convlist.append((lambda x1, x2: x1 + [x2.x], 'ls, x -> ls'))
            convlist.append((lambda x: torch.cat(x, dim=-1), 'ls -> x'))
        else:
            convlist = []
            for i in range(cfg.gt.layers):
                convlist.append((NodeFormerConv(cfg.gt.dim_hidden, cfg.gt.dim_hidden, config=cfg.gt), 'x -> x'))
        
        self.convs = Sequential('x', convlist)

        GNNHead = register.head_dict[cfg.gnn.head]
        if cfg.gt.use_jk:
            self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden * cfg.gt.layers + cfg.gt.dim_hidden, dim_out=dim_out)
        else:
            self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
