import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_network

from opengt.encoder.feature_encoder import FeatureEncoder
from opengt.layer.bga_layer import BGALayer

@register_network('BGA')
class BGA(nn.Module):
    """
    Bilevel Graph Attention model. Used in CoBFormer model.
    Adapted from https://github.com/null-xyj/CoBFormer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        dropout1 (float): Dropout rate for the final layer.
        dropout2 (float): Dropout rate for the BGA layers.
    
    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.patch (torch.Tensor): Patch indices.
    
    Output:
        batch.x (torch.Tensor): Output node features after applying the BGA model.
    """
    def __init__(self, dim_in: int, dim_out: int, dropout1=0.5, dropout2=0.1):
        super(BGA, self).__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        
        self.layers = cfg.gt.layers
        self.n_head = cfg.gt.n_heads
        self.dropout = nn.Dropout(dropout1)
        self.BGALayers = nn.ModuleList()
        for _ in range(0, cfg.gt.layers):
            self.BGALayers.append(
                BGALayer(cfg.gt.n_heads, cfg.gt.dim_hidden, dropout=dropout2))
        self.classifier = nn.Linear(cfg.gt.dim_hidden, dim_out)
        self.attn=[]

    def forward(self, batch):
        batch.x = F.pad(batch.x, [0, 0, 0, 1]) # padding for the last node
        num_nodes = batch.x.shape[0]
        
        patch_mask = (batch.patch != num_nodes - 1).float().unsqueeze(-1)
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int()

        batch = self.encoder(batch)
        for i in range(0, self.layers):
            batch.x = self.BGALayers[i](batch.x, batch.patch, attn_mask)
        batch.x = self.dropout(batch.x)
        batch.x = self.classifier(batch.x)
        batch.x = batch.x[:-1]
        return batch