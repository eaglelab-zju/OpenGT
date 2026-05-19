import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch import Tensor
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_batch

from opengt.encoder.feature_encoder import FeatureEncoder


class BareTransformerBlock(nn.Module):
    """Global self-attention over nodes, intentionally ignoring graph edges."""

    def __init__(self, dim_h, num_heads, dropout, attn_dropout,
                 layer_norm, batch_norm, residual):
        super().__init__()
        if dim_h % num_heads != 0:
            raise ValueError(f"dim_hidden={dim_h} must be divisible by n_heads={num_heads}")

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_h,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
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

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        h_in1 = x
        key_padding_mask = ~valid_mask
        h, _ = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h + h_in1
        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h[valid_mask])
            x = x.clone()
            x[valid_mask] = h
            h = x

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
            h_valid = self.batch_norm2(h[valid_mask])
            h = h.clone()
            h[valid_mask] = h_valid
        return h


@register_network('BareTransformer')
class BareTransformer(nn.Module):
    """Vanilla Transformer baseline for PE ablations; no graph structure is used."""

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
                f"BareTransformer expects gt.dim_hidden == gnn.dim_inner == dim_in, "
                f"got dim_hidden={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        self.layers = nn.ModuleList([
            BareTransformerBlock(
                dim_h=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=cfg.gt.residual,
            )
            for _ in range(cfg.gt.layers)
        ])

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def _batch_vector(self, batch):
        b = getattr(batch, 'batch', None)
        if b is None:
            n = batch.num_nodes if getattr(batch, 'num_nodes', None) is not None \
                else batch.x.size(0)
            b = torch.zeros(n, dtype=torch.long, device=batch.x.device)
        return b

    def forward(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        b = self._batch_vector(batch)
        b_sorted, sort_idx = b.sort()
        rev_perm = torch.empty_like(sort_idx)
        rev_perm[sort_idx] = torch.arange(len(sort_idx), device=sort_idx.device)

        x_dense, valid_mask = to_dense_batch(batch.x[sort_idx], b_sorted)
        for layer in self.layers:
            x_dense = layer(x_dense, valid_mask)
        batch.x = x_dense[valid_mask][rev_perm]
        return self.post_mp(batch)
