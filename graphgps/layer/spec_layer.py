import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_scatter import scatter




@register_layer('SpecLayer')
class SpecLayer(nn.Module):

    def __init__(self, dim_out, n_heads, dropout=0.0, norm='none'): # nbases=n_heads+1, ncombines=dim_out
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(dropout)

        if norm == 'none': 
            self.weight = nn.Parameter(torch.ones((1, n_heads+1, dim_out)))
        else:
            self.weight = nn.Parameter(torch.empty((1, n_heads+1, dim_out)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':    # Arxiv
            self.norm = nn.LayerNorm(dim_out)
        elif norm == 'batch':  # Penn
            self.norm = nn.BatchNorm1d(dim_out)
        else:                  # Others
            self.norm = None 
        
        self.n_heads = n_heads

    def forward(self, batch):
        basic_feats = [batch.x]
        utx = batch.EigVec.permute(1,0) @ batch.x
        for i in range(self.n_heads):
            basic_feats.append(batch.EigVec @ (batch.EigVal[:, i].unsqueeze(1) * utx))  # [N, d]
        basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]
        # h = conv(basic_feats) # [SpecLayer(nheads+1, nclass, prop_dropout, norm=norm) for i in range(nlayer)]
        ######## done: rewrite logic

        print(basic_feats.shape, self.weight.shape)
        
        basic_feats = self.prop_dropout(basic_feats) * self.weight      # [N, m, d] * [1, m, d]
        basic_feats = torch.sum(basic_feats, dim=1)

        if self.norm is not None:
            basic_feats = self.norm(basic_feats)
            basic_feats = F.relu(basic_feats)
        
        batch.x = basic_feats

        return batch