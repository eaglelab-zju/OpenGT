import torch
import torch.nn as nn
from graphgps.network.trans_conv import TransConv


from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, GCNConv, Linear
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.feature_encoder import FeatureEncoder


@register_network("SGFormer")
class SGFormer(nn.Module):
    """
    SGFormer model. Adapted from https://github.com/qitianwu/SGFormer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.layers: Number of SGFormer layers.
            - cfg.gt.dim_hidden: Hidden dimension for GNN layers and SGFormer layers.
            - cfg.gt.aggregate: Type of aggregation to use for the graph. e.g., 'add' or 'cat'.
            - cfg.gt.use_graph: Whether to use graph information.
            - cfg.gt.graph_weight: Weight for the graph information in the aggregation.
            - cfg.gnn.head: Type of head to use for the final output layer.
    
    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.edge_index (torch.Tensor): Edge indices of the graph.

    Output:
        batch (task dependent type, see output head): Output after model processing.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        
        if cfg.gt.use_graph:
            if cfg.gt.aggregate == 'add':
                self.model=Sequential('x',[
                    (TransConv(dim_in, cfg.gt.dim_hidden), 'x -> x1'),
                    (GCNConv(new_layer_config(dim_in=cfg.gt.dim_hidden, dim_out=cfg.gt.dim_hidden, num_layers=1, has_act=True, has_bias=True, cfg=cfg)), 'x -> x2'),
                    (lambda x1, x2: self.aggregate_batches_add(x1, x2), 'x1, x2 -> x'),
                ])
                dim_mid=cfg.gt.dim_hidden
            elif cfg.gt.aggregate == 'cat':
                self.model=Sequential('x',[
                    (TransConv(dim_in, cfg.gt.dim_hidden), 'x -> x1'),
                    (GCNConv(new_layer_config(dim_in=cfg.gt.dim_hidden, dim_out=cfg.gt.dim_hidden, num_layers=1, has_act=True, has_bias=True, cfg=cfg)), 'x -> x2'),
                    (lambda x1, x2: self.aggregate_batches_cat(x1, x2), 'x1, x2 -> x'),
                ])
                dim_mid=2*cfg.gt.dim_hidden
            else:
                raise ValueError(f'Invalid aggregate type:{cfg.gt.aggregate}')
        else:
            self.model=Sequential('x',[
                (TransConv(dim_in, cfg.gt.dim_hidden), 'x -> x'),
            ])
            dim_mid=cfg.gt.dim_hidden
        
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=dim_mid, dim_out=dim_out)

    def aggregate_batches_add(self, x1, x2):
        new_batch = x1.clone()
        new_batch.x = cfg.gt.graph_weight * x2.x + (1 - cfg.gt.graph_weight) * x1.x
        return new_batch
    
    def aggregate_batches_cat(self, x1, x2):
        new_batch = x1.clone()
        new_batch.x = torch.cat((x1.x, x2.x), dim=1)
        return new_batch

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
