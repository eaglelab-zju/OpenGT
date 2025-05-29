import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from opengt.layer.graphormer_layer import GraphormerLayer
from opengt.encoder.feature_encoder import FeatureEncoder



@register_network('Graphormer')
class GraphormerModel(torch.nn.Module):
    """Graphormer port to GraphGPS.
    https://arxiv.org/abs/2106.05234
    Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., ... & Liu, T. Y.
    Do transformers really perform badly for graph representation? (NeurIPS2021)
    Adapted from https://github.com/rampasek/GraphGPS

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.graphormer.num_layers (int): Number of Graphormer layers.
            - cfg.graphormer.embed_dim (int): Embedding dimension for Graphormer layers. Need to match cfg.gnn.dim_inner.
            - cfg.graphormer.num_heads (int): Number of attention heads.
            - cfg.graphormer.dropout (float): Dropout rate.
            - cfg.graphormer.attention_dropout (float): Attention dropout rate.
            - cfg.graphormer.mlp_dropout (float): MLP dropout rate.
            - cfg.gnn.head (str): Type of head to use for the final output layer.
            - cfg.gnn.layers_pre_mp (int): Number of pre-message-passing layers.
            - cfg.gnn.dim_inner (int): Inner dimension for GNN layers. Need to match cfg.graphormer.embed_dim.
    
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

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.graphormer.embed_dim == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and embed dims must match: "
                f"embed_dim={cfg.graphormer.embed_dim} "
                f"dim_inner={cfg.gnn.dim_inner} dim_in={dim_in}"
            )

        layers = []
        for _ in range(cfg.graphormer.num_layers):
            layers.append(GraphormerLayer(
                embed_dim=cfg.graphormer.embed_dim,
                num_heads=cfg.graphormer.num_heads,
                dropout=cfg.graphormer.dropout,
                attention_dropout=cfg.graphormer.attention_dropout,
                mlp_dropout=cfg.graphormer.mlp_dropout
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
