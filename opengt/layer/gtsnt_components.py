# Vendored from GT-SNT (MIT) — https://github.com/Zhhuizhe/GT-SNT
# OpenGT: imports adjusted; kept as isolated building blocks for ``opengt.network.gtsnt``.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

import torch_sparse
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops as add_self_loops_fn,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.utils import spmm

try:
    from spikingjelly.clock_driven.neuron import (
        MultiStepIFNode,
        MultiStepLIFNode,
        MultiStepParametricLIFNode,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'GTSNT requires ``spikingjelly`` (see GT-SNT README). '
        'Example: ``pip install spikingjelly==0.0.0.0.14``'
    ) from e


def gcn_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    improved=False,
    add_self_loops=True,
    flow='source_to_target',
    dtype=None,
):
    fill_value = 2.0 if improved else 1.0

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.0)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError(
                "Sparse CSC matrices are not yet supported in 'gcn_norm'"
            )

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class GTSNTGCNConv(MessagePassing):
    """GCNConv from GT-SNT (GCN norm + optional output L2 normalize)."""

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(
            in_channels, out_channels, bias=False, weight_initializer='glorot'
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if isinstance(edge_index, Tensor):
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    self.improved,
                    self.add_self_loops,
                    self.flow,
                    x.dtype,
                )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        elif isinstance(edge_index, SparseTensor):
            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    self.improved,
                    self.add_self_loops,
                    self.flow,
                    x.dtype,
                )
                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        x = self.lin(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class SMultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        cb_channels: int,
        num_heads: int,
        dropout: float = 0.5,
        attn_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hid_channels = hid_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.lin_cb = nn.Linear(cb_channels, hid_channels)

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def vq_attn_block(
        self,
        query: Tensor,
        value: Tensor,
        codebook: Tensor,
        coding_indices: Tensor,
        mask: Optional[Tensor] = None,
    ):
        num_nodes, head_dim = query.size(0), self.hid_channels // self.num_heads

        q, v, cb = query, value, self.lin_cb(codebook)
        q = q.view(-1, self.num_heads, head_dim).transpose(0, 1)
        v = v.view(-1, self.num_heads, head_dim).transpose(0, 1)
        cb = cb.view(-1, self.num_heads, head_dim).transpose(0, 1)
        q, v, cb = F.normalize(q, dim=-1), F.normalize(v, dim=-1), F.normalize(cb, dim=-1)

        v = scatter(v, coding_indices, dim=1, reduce='mean')
        if mask is not None:
            cb, v = cb[:, mask], v[:, mask]

        q_scaled = q / (q.size(2) ** 0.5)
        attn_output_weights = torch.bmm(q_scaled, cb.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.attn_dropout, training=self.training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            num_nodes, self.hid_channels
        )
        attn_output = attn_output + value
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

        return attn_output


class SNT(MessagePassing):
    def __init__(
        self,
        num_nodes: int,
        cb_channels: int,
        T: int,
        neuron: str = 'PLIF',
        v_threshold: float = 1.0,
        init_beta: float = 0.2,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.T = T
        self.init_beta = init_beta
        self.beta = nn.Parameter(torch.tensor([self.init_beta], dtype=torch.float))
        self.rand_feat = nn.Parameter(torch.rand((num_nodes, cb_channels), dtype=torch.float))

        if neuron == 'IF':
            self.neuron = MultiStepIFNode(v_threshold=v_threshold)
        elif neuron == 'LIF':
            self.neuron = MultiStepLIFNode(v_threshold=v_threshold)
        elif neuron == 'PLIF':
            self.neuron = MultiStepParametricLIFNode(v_threshold=v_threshold)
        else:
            raise NotImplementedError(f'Unknown neuron type: {neuron}')

        self.reset_parameters()

    def neuron_reset(self):
        self.neuron.reset()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.uniform_(self.rand_feat)
        nn.init.constant_(self.beta, self.init_beta)
        self.neuron_reset()

    def forward(self, edge_index: Adj, edge_weight: Optional[Tensor] = None):
        embds = [self.rand_feat]
        while len(embds) < self.T:
            embd = embds[-1]
            embd = self.propagate(edge_index, x=embd, edge_weight=edge_weight)
            embds.append(embd)

        embds = torch.stack(embds, dim=0)
        spikes_out = self.neuron(embds).sum(0)

        _, codes_indices, counts = torch.unique(
            spikes_out.detach(), return_inverse=True, return_counts=True, dim=0
        )
        codebook = scatter(spikes_out, codes_indices, dim=0, reduce='mean')
        return codebook, codes_indices, counts

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
