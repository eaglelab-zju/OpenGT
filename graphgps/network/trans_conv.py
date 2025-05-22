import torch
import torch.nn as nn
from graphgps.layer.trans_conv_layer import TransConvLayer

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_network

@register_network("TransConv")
class TransConv(nn.Module):
    """
    TransConv module for SGFormer model. Adapted from https://github.com/qitianwu/SGFormer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.layers: Number of TransConv layers.
    
    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.edge_index (torch.Tensor): Edge indices of the graph.
    
    Output:
        batch (torch_geometric.data.Batch): Output batch after processing.
    """
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
