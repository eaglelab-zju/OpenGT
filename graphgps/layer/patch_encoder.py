import torch.nn as nn
import functorch.einops.rearrange as Rearrange
from torch_scatter import scatter

from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, MLP, GeneralLayer, GCNConv
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

@register_layer('patch_encoder')
class PatchEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.gnns = nn.ModuleList([GeneralLayer(cfg.gt.layer_type.split('+')[0].lower()+'conv', new_layer_config(dim_in = cfg.gt.dim_hidden, dim_out = cfg.gt.dim_hidden, has_bias = True, has_act = False, num_layers = 1, cfg = cfg)) for _ in range(cfg.gt.layers)])
        self.U = nn.ModuleList([MLP(new_layer_config(dim_in=cfg.gt.dim_hidden, dim_out=cfg.gt.dim_hidden, num_layers=1, has_act=True, has_bias=True, cfg=cfg)) for _ in range(cfg.gt.layers-1)])
        self.pooling = cfg.gt.pooling

    def forward(self, batch):
        x = batch.x[batch.subgraphs_nodes_mapper]
        if batch.edge_attr is not None:
            e = batch.edge_attr[batch.subgraphs_edges_mapper]
        else:
            e = None
        edge_index = batch.combined_subgraphs
        batch_x = batch.subgraphs_batch
        tmp_batch = batch.clone()
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, batch.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[batch.subgraphs_nodes_mapper]
            tmp_batch.x = x
            tmp_batch.edge_index = edge_index
            tmp_batch.edge_attr = e
            tmp_batch = gnn(tmp_batch)
            x = tmp_batch.x
            edge_index = tmp_batch.edge_index
            e = tmp_batch.edge_attr

        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

        ret = batch.clone()
        ret.x = subgraph_x
        return ret