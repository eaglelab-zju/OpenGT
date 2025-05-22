import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Parameters:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    
    Input:
        batch.x (torch.Tensor): Node features.
        batch.y (torch.Tensor): Node labels.
    
    Output:
        pred (torch.Tensor): Predicted node labels.
        true (torch.Tensor): True node labels.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, true = self._apply_index(batch)
        return pred, true
