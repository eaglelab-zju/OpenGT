# HubGT encoder adapted for OpenGT GraphGym node classification.
# Structural blocks vendored from https://github.com/gdmnl/HubGT (MIT).
#
# OpenGT changes vs upstream ``model.GT``:
# - No graph-level ``downstream_out_proj`` / hub readout; node logits from GraphGym ``post_mp``.
# - Full-graph node tasks: attention bias from undirected unweighted shortest-path hop counts
#   (clamped to [0, 254]; unreachable pairs = 255 = INF8 for StructuralEmbedding mask).
# - Single-graph (or one fused batch) only; multi-graph mini-batches are not supported.
# - Layer count / heads / FFN & attention dropout follow ``cfg.gt`` (same keys as other GT stacks).
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from opengt.encoder.feature_encoder import FeatureEncoder
from opengt.layer.hubgt_modules import INF8, EncoderLayer, init_params


def _num_graphs(batch) -> int:
    ptr = getattr(batch, 'ptr', None)
    if ptr is not None:
        return int(ptr.numel() - 1)
    b = getattr(batch, 'batch', None)
    if b is None:
        return 1
    return int(b.max().item()) + 1


def shortest_path_hop_bias(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Integer hop distances [N, N, 1] for StructuralEmbedding (INF8 = no edge path)."""
    ei = edge_index.detach().cpu().numpy()
    row, col = ei[0], ei[1]
    data = np.ones(len(row), dtype=np.float64)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    dist = shortest_path(adj, directed=False, unweighted=True)
    out = np.full((num_nodes, num_nodes), INF8, dtype=np.int64)
    finite = np.isfinite(dist)
    clipped = np.clip(dist[finite].astype(np.int64), 0, INF8 - 1)
    out[finite] = clipped
    t = torch.as_tensor(out, dtype=torch.int32, device=device)
    return t.unsqueeze(-1)


@register_network('HubGT')
class HubGT(nn.Module):
    """FeatureEncoder (+ optional pre-MP) + HubGT encoder stack + GraphGym head."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in_enc = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in_enc, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp
            )
            dim_in_enc = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in_enc:
            raise ValueError(
                'HubGT expects gt.dim_hidden == gnn.dim_inner == dim_in after encoder/pre_mp, '
                f'got dim_hidden={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} dim_in={dim_in_enc}'
            )

        hidden = cfg.gt.dim_hidden
        n_heads = int(cfg.gt.n_heads)
        if hidden % n_heads != 0:
            raise ValueError(
                f'HubGT: dim_hidden={hidden} must be divisible by gt.n_heads={n_heads}'
            )

        ffn_ratio = float(cfg.hubgt.ffn_ratio)
        ffn_size = max(int(round(hidden * ffn_ratio)), hidden)

        self.input_dropout = nn.Dropout(float(cfg.hubgt.dp_input))
        num_global = int(cfg.hubgt.num_global_node)
        n_layers = int(cfg.gt.layers)
        dp_bias = float(cfg.hubgt.dp_bias)
        ff_dropout = float(cfg.gt.dropout)
        attn_dropout = float(cfg.gt.attn_dropout)

        encoders = [
            EncoderLayer(
                hidden,
                ffn_size,
                ff_dropout,
                attn_dropout,
                n_heads,
                1,
                num_global,
                dp_bias,
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden)

        for layer in self.layers:
            layer.apply(lambda m: init_params(m, n_layers=n_layers))

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=hidden, dim_out=dim_out)

        self._cached_ei: Optional[torch.Tensor] = None
        self._cached_bias: Optional[torch.Tensor] = None
        self._cached_n: Optional[int] = None

    def _attn_bias(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device):
        if (
            self._cached_bias is not None
            and self._cached_n == num_nodes
            and self._cached_ei is not None
            and self._cached_ei.shape == edge_index.shape
            and self._cached_ei.device == edge_index.device
            and bool(torch.equal(self._cached_ei, edge_index))
        ):
            return self._cached_bias
        max_n = int(cfg.hubgt.max_nodes)
        if num_nodes > max_n:
            raise ValueError(
                f'HubGT: num_nodes={num_nodes} exceeds hubgt.max_nodes={max_n} '
                '(dense attention + SPD precompute).'
            )
        bias = shortest_path_hop_bias(edge_index, num_nodes, device)
        self._cached_ei = edge_index.detach().clone()
        self._cached_bias = bias
        self._cached_n = num_nodes
        return bias

    def forward(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        if _num_graphs(batch) != 1:
            raise NotImplementedError(
                'HubGT (dense full-graph) currently supports a single graph per batch '
                '(Planetoid / Critical full-batch). Use a different model for batched graphs.'
            )

        n = batch.x.size(0)
        attn_bias = self._attn_bias(batch.edge_index, n, batch.x.device)
        x = batch.x.unsqueeze(0)
        x = self.input_dropout(x)
        attn_bias_b = attn_bias.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, attn_bias_b)
        x = self.final_ln(x)
        batch.x = x.squeeze(0)
        return self.post_mp(batch)
