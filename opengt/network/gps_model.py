import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from opengt.layer.gps_layer import GPSLayer
from opengt.encoder.feature_encoder import FeatureEncoder


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    Adapted from https://github.com/rampasek/GraphGPS

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.layers (int): Number of GPS layers.
            - cfg.gt.dim_hidden (int): Hidden dimension for GPS layers. Need to match cfg.gnn.dim_inner.
            - cfg.gt.layer_type (str): Type of layer to use, containing '+'-separated local model type and global model type, e.g., 'GINE+Transformer'.
            - cfg.gt.pna_degrees (list): List of PNA degrees for local model.
            - cfg.gt.n_heads (int): Number of attention heads.
            - cfg.gt.dropout (float): Dropout rate.
            - cfg.gt.attn_dropout (float): Attention dropout rate.
            - cfg.gt.layer_norm (bool): Whether to use layer normalization.
            - cfg.gt.batch_norm (bool): Whether to use batch normalization.
            - cfg.gnn.head (str): Type of head to use for the final output layer.
            - cfg.gnn.layers_pre_mp (int): Number of pre-message-passing layers.
            - cfg.gnn.dim_inner (int): Inner dimension for GNN layers. Need to match cfg.gt.dim_hidden.
            - cfg.gnn.act (str): Activation function to use.

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

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
