import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import TransformerConv
from opengt.encoder.feature_encoder import FeatureEncoder


class GraphTransformerBlock(nn.Module):
    """Graph Transformer block close to the original AAAI'21 design."""

    def __init__(self, dim_h, num_heads, dropout, layer_norm, batch_norm, residual):
        super().__init__()
        if dim_h % num_heads != 0:
            raise ValueError(f"dim_hidden={dim_h} must be divisible by n_heads={num_heads}")

        edge_dim = getattr(cfg.gnn, 'dim_edge', None) if cfg.dataset.edge_encoder else None
        self.attn = TransformerConv(
            in_channels=dim_h,
            out_channels=dim_h // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.proj = nn.Linear(dim_h, dim_h)

        self.ffn1 = nn.Linear(dim_h, dim_h * 2)
        self.ffn2 = nn.Linear(dim_h * 2, dim_h)

        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(dim_h)
            self.layer_norm2 = nn.LayerNorm(dim_h)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(dim_h)
            self.batch_norm2 = nn.BatchNorm1d(dim_h)

    def forward(self, batch):
        h = batch.x
        h_in1 = h

        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        h = self.attn(h, batch.edge_index, edge_attr=edge_attr)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.proj(h)

        if self.residual:
            h = h + h_in1
        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h
        h = self.ffn1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.ffn2(h)

        if self.residual:
            h = h + h_in2
        if self.layer_norm:
            h = self.layer_norm2(h)
        if self.batch_norm:
            h = self.batch_norm2(h)

        ret = batch.clone()
        ret.x = h
        return ret

@register_network("Graphtransformer")
class Graphtransformer(torch.nn.Module):
    '''
    Graphtransformer model. Adapted from https://github.com/graphdeeplearning/graphtransformer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.layers (int): Number of Graphtransformer layers.
            - cfg.gt.dim_hidden (int): Hidden dimension for GNN layers and Graphtransformer layers.
            - cfg.gt.layer_type (str): Type of layer to use for the Graphtransformer layers.
            - cfg.gnn.head (str): Type of head to use for the final output layer.
        
    Input:
        batch (torch_geometric.data.Batch): input batch containing node features and graph structure.
            - batch.x (torch.Tensor): input node features.
            - batch.edge_index (torch.Tensor): edge indices of the graph.
    
    Output:
        batch (task dependent type, see output head): Output after model processing.
    '''
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"dim_hidden={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GraphTransformerBlock(
                dim_h=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=cfg.gt.residual,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
    
    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
