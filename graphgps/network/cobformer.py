import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, GeneralMultiLayer
from torch_geometric.graphgym.register import register_network

from graphgps.network.bga_model import BGA

@register_network('CoBFormer')
class CoBFormer(torch.nn.Module):
    """
    CoBFormer model. Only supports transductive node level tasks.
    Adapted from https://github.com/null-xyj/CoBFormer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.alpha (float): Balance factor for GNN and BGA loss.
            - cfg.gt.tau (float): Temperature parameter for softmax.
            - cfg.gt.layer_type (str): Type of GNN layer to use. e.g., 'GCN'.
            - cfg.gnn.layers (int): Number of GNN layers.
        
    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.edge_index (torch.Tensor): Edge indices of the graph.
            - batch.patch (torch.Tensor): Patch indices.
            - batch.y (torch.Tensor): Input labels.
            - batch.split (str): Split type (train, val, test).
    
    Output:
        pred (torch.Tensor): Predicted node features after applying the CoBFormer model.
        true (torch.Tensor): True labels.
        extra_loss (torch.Tensor): Extra loss term for GNN and BGA cotraining.
    """
    def __init__(self, dim_in: int, dim_out: int):
        super(CoBFormer, self).__init__()
        self.alpha = cfg.gt.alpha
        self.tau = cfg.gt.tau
        self.gnn = GeneralMultiLayer(cfg.gt.layer_type.split('+')[0].lower()+'conv', new_layer_config(dim_in = dim_in, dim_out = dim_out, has_bias = True, has_act = False, num_layers = cfg.gnn.layers, cfg = cfg))
        self.bga = BGA(dim_in, dim_out)
        self.attn = None
        
    def _apply_index(self, batch):
        x = batch.x
        y = batch.y if 'y' in batch else None

        if 'split' not in batch:
            return x, y

        mask = batch[f'{batch.split}_mask']
        return x[mask], y[mask] if y is not None else None
    
    def forward(self, batch):
        tmpbatch = batch.clone()
        batch1 = self.gnn(tmpbatch)
        batch2 = self.bga(batch)
        
        z1 = batch1.x
        z2 = batch2.x
        extra_loss = (F.cross_entropy(z1*self.tau, F.softmax(z2*self.tau, dim=1)) + F.cross_entropy(z2*self.tau, F.softmax(z1*self.tau, dim=1)))*(1-self.alpha)/self.alpha

        pred, true = self._apply_index(batch2)
        return pred, true, extra_loss
