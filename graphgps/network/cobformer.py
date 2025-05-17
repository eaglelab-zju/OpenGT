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
