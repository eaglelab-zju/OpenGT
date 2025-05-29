import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('mlp_mixer_graph')
class MLPMixerGraphHead(nn.Module):
    """
    Graph MLP Mixer prediction head for graph prediction tasks.

    Note that this head is specially designed for Graph MLP Mixer (without pooling layer). Cannot work on other models.

    Parameters:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    
    Input:
        batch.x (torch.Tensor): Graph embedding.
        batch.y (torch.Tensor): Graph labels.
    
    Output:
        pred (torch.Tensor): Predicted graph labels.
        true (torch.Tensor): True graph labels.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = batch.x
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, true = self._apply_index(batch)
        return pred, true
