import torch.nn as nn
import functorch.einops.rearrange as Rearrange

from torch_geometric.nn import Sequential
from torch_geometric.graphgym.models.layer import LayerConfig , MLP
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

def gen_layer_config(dim_in_out, dim_inner, num_layers, dropout):
    return LayerConfig(
        has_batchnorm=cfg.gnn.batchnorm,
        bn_eps=cfg.bn.eps,
        bn_mom=cfg.bn.mom,
        mem_inplace=cfg.mem.inplace,
        dim_in=dim_in_out,
        dim_out=dim_in_out,
        edge_dim=cfg.dataset.edge_dim,
        has_l2norm=cfg.gnn.l2norm,
        dropout=dropout,
        has_act=True,
        final_act=False,
        act=cfg.gnn.act,
        has_bias=True,
        keep_edge=True,
        dim_inner=dim_inner,
        num_layers=num_layers,
    )

@register_layer("mixer_block")
class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        cfg.gnn.dropout = dropout
        self.token_mix = Sequential('x',[
            (nn.LayerNorm(dim), 'x -> x'),
            lambda x:self.rearrange(x),
            MLP(gen_layer_config(dim_in_out = num_patch, dim_inner = token_dim, num_layers = 2, dropout = dropout)),
            lambda x:self.rearrange(x)
        ])
        self.channel_mix = Sequential('x',[
            (nn.LayerNorm(dim), 'x -> x'),
            MLP(gen_layer_config(dim_in_out = dim, dim_inner = channel_dim, num_layers = 2, dropout = dropout))
        ])

    def rearrange(self, x):
        return Rearrange(x, 'b p d -> b d p')

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x
