import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
import math

class SineEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoder, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, batch):
        # input:  [N]
        # output: [N, d]

        ee = batch.x * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(batch.x.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((batch.x.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        batch.x = self.eig_w(eeig)
        return batch