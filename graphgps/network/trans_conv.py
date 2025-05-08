import torch
import torch.nn as nn
from graphgps.layer.trans_conv_layer import TransConvLayer

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_network

@register_network("TransConv")
class TransConv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.mlp = MLP(new_layer_config(dim_in=dim_in, dim_out=dim_out, num_layers=1, has_act=True, has_bias=True, cfg=cfg))
        
        layers = nn.ModuleList()
        for i in range(cfg.gt.layers):
            layers.append(
                TransConvLayer(dim_out, dim_out, config=cfg.gt))
        self.layers=nn.Sequential(*layers)


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
