import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.san_layer import SANLayer
from graphgps.layer.san2_layer import SAN2Layer


@register_network('SANTransformer')
class SANTransformer(torch.nn.Module):
    """Spectral Attention Network (SAN) Graph Transformer.
    https://arxiv.org/abs/2106.03893
    Adapted from https://github.com/rampasek/GraphGPS

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.layers (int): Number of SAN layers.
            - cfg.gt.dim_hidden (int): Hidden dimension for GNN layers and SAN layers. Need to match cfg.gnn.dim_inner.
            - cfg.gt.gamma (float): Gamma parameter for SAN layers.
            - cfg.gt.n_heads (int): Number of attention heads.
            - cfg.gt.full_graph (bool): Whether to use full graph attention.
            - cfg.gt.dropout (float): Dropout rate for the SAN layers.
            - cfg.gt.layer_norm (bool): Whether to use layer normalization.
            - cfg.gt.batch_norm (bool): Whether to use batch normalization.
            - cfg.gt.residual (bool): Whether to use residual connections.
            - cfg.gnn.head (str): Type of head to use for the final output layer.
            - cfg.gnn.layers_pre_mp (int): Number of pre-message-passing layers.
            - cfg.gnn.dim_inner (int): Inner dimension for GNN layers. Need to match cfg.gt.dim_hidden.

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

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        fake_edge_emb = torch.nn.Embedding(1, cfg.gt.dim_hidden)
        # torch.nn.init.xavier_uniform_(fake_edge_emb.weight.data)
        Layer = {
            'SANLayer': SANLayer,
            'SAN2Layer': SAN2Layer,
        }.get(cfg.gt.layer_type)
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(Layer(gamma=cfg.gt.gamma,
                                in_dim=cfg.gt.dim_hidden,
                                out_dim=cfg.gt.dim_hidden,
                                num_heads=cfg.gt.n_heads,
                                full_graph=cfg.gt.full_graph,
                                fake_edge_emb=fake_edge_emb,
                                dropout=cfg.gt.dropout,
                                layer_norm=cfg.gt.layer_norm,
                                batch_norm=cfg.gt.batch_norm,
                                residual=cfg.gt.residual))
        self.trf_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
