import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import GeneralLayer, new_layer_config
from torch_geometric.graphgym.register import register_network

from opengt.encoder.feature_encoder import FeatureEncoder


@register_network("PureGNN")
class PureGNN(torch.nn.Module):
    """Pure message-passing GNN model driven by `cfg.gt.layer_type`."""

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

        layer_name = f"{cfg.gt.layer_type.lower()}conv"
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GeneralLayer(
                layer_name,
                new_layer_config(
                    dim_in=cfg.gt.dim_hidden,
                    dim_out=cfg.gt.dim_hidden,
                    has_bias=True,
                    has_act=False,
                    num_layers=1,
                    cfg=cfg,
                ),
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
