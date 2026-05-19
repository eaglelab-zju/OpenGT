# Vendored from https://github.com/null-xyj/M3Dphormer (NeurIPS 2025; no LICENSE file in repo).
# Upstream paper: "Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework".
#
# OpenGT changes:
# - `isinstance(..., torch.Tensor)` instead of `type(em) == torch.Tensor`.
# - `calculate_norm_A` kept here (uses torch_sparse like upstream `utils.py`).
# - `M3DphormerEncoder`: no classification head; returns hidden states for all virtual nodes
#   (original + cluster + global); GraphGym `post_mp` consumes only the first N rows.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import ones_, zeros_
from torch_geometric.nn.inits import glorot as pyg_glorot
from torch_scatter import scatter
from torch_sparse import SparseTensor
from typing import List, Optional


def calculate_norm_A(edge_index: torch.Tensor, val: Optional[torch.Tensor] = None):
    """Symmetric normalized adjacency as COO + broadcastable edge weights (GCN expert)."""
    N = int(edge_index.max().item()) + 1
    if val is None:
        val = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=torch.float32)

    A = SparseTensor.from_edge_index(
        edge_index=edge_index, edge_attr=val, sparse_sizes=(N, N))
    deg = A.sum(1)
    deg_inv_sqrt = deg.pow(-1).view(-1, 1).pow(0.5)
    norm_A = A * deg_inv_sqrt * deg_inv_sqrt.t()

    row, col, v = norm_A.coo()
    return torch.stack([row, col]), v.view(-1, 1, 1)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = bias
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(dim))

    def reset_parameters(self):
        ones_(self.weight)
        if self.bias:
            zeros_(self.offset)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.bias:
            return output * self.weight + self.offset
        return output * self.weight


class BiAttentionMoeLayer(nn.Module):
    def __init__(
        self,
        share_experts: List[nn.Module],
        router_experts: List[nn.Module],
        gate_weights: nn.Parameter,
        alpha_weights: nn.Parameter,
    ):
        super().__init__()
        self.share_experts = nn.ModuleList(share_experts) if share_experts else None
        self.router_experts = nn.ModuleList(router_experts) if router_experts else None
        self.gate = gate_weights
        self.alpha = alpha_weights

    def reset_parameters(self):
        if self.share_experts is not None:
            for expert in self.share_experts:
                expert.reset_parameters()
        if self.router_experts is not None:
            for expert in self.router_experts:
                expert.reset_parameters()
        zeros_(self.gate)
        zeros_(self.alpha)

    def forward(self, inputs, edge_masks, need_weights: bool = False):
        num_ori_nodes = int(edge_masks[0][0].max().item()) + 1
        num_cluster = int(edge_masks[1][0].max().item()) + 1 - num_ori_nodes

        if self.share_experts is not None:
            results, _ = self.share_experts[0](inputs, edge_masks[0], need_weights)
        else:
            results = torch.zeros_like(inputs)

        weights = torch.zeros(
            [inputs.size(0), len(self.router_experts)], device=inputs.device, dtype=inputs.dtype)
        weights[:num_ori_nodes, 0] = torch.sigmoid((inputs[:num_ori_nodes] * self.gate).sum(dim=-1))
        weights[:num_ori_nodes, 1] = 1.0 - weights[:num_ori_nodes, 0]
        weights[num_ori_nodes:num_ori_nodes + num_cluster, 0] = 1.0
        weights[num_ori_nodes + num_cluster:, 1] = 1.0

        alpha = torch.zeros(inputs.size(0), device=inputs.device, dtype=inputs.dtype).unsqueeze(-1)
        alpha[:num_ori_nodes] = torch.sigmoid(
            (inputs[:num_ori_nodes] * self.alpha).sum(dim=-1, keepdim=True))

        results = results * alpha
        for i, expert in enumerate(self.router_experts):
            out, _ = expert(inputs, edge_masks[i + 1], need_weights)
            results = results + (1.0 - alpha) * weights[:, i].unsqueeze(-1) * out
        return results


class DualAttention(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_cache: bool = False,
        agg_type: str = 'Trans',
    ):
        super().__init__()
        self.n_head = 1 if agg_type == 'GCN' else num_heads
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.attn_dropout = nn.Dropout(p=dropout)
        self.agg_type = agg_type
        self.use_cache = use_cache

        if agg_type == 'Trans':
            self.lin_q = nn.Linear(in_dim, h_dim, bias=bias)
            self.lin_k = nn.Linear(in_dim, h_dim, bias=bias)
            self.temperature = (h_dim // num_heads) ** 0.5
        elif agg_type == 'GAT':
            self.lin_a = nn.Linear(in_dim, h_dim, bias=bias)
            self.alpha1 = nn.Parameter(torch.empty(1, num_heads, h_dim // num_heads))
            self.alpha2 = nn.Parameter(torch.empty(1, num_heads, h_dim // num_heads))
            self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        elif agg_type == 'GCN':
            self.gcn_cache = None

        self.lin_v = nn.Linear(in_dim, h_dim, bias=bias)

    def reset_parameters(self):
        self.lin_v.reset_parameters()
        if self.agg_type == 'Trans':
            self.lin_q.reset_parameters()
            self.lin_k.reset_parameters()
        elif self.agg_type == 'GAT':
            self.lin_a.reset_parameters()
            pyg_glorot(self.alpha1)
            pyg_glorot(self.alpha2)

    def dense_forward(self, q, k, v, edge_mask, out, need_weights: bool = False):
        if len(edge_mask) == 3:
            G1, G2, M = edge_mask[0], edge_mask[1], edge_mask[-1]
        else:
            G1 = edge_mask[0]
            G2 = edge_mask[0]
            M = edge_mask[-1]

        if self.agg_type == 'Trans':
            q_sel = q[G1]
            k_sel = k[G2]
            q_sel = q_sel.transpose(0, 1)
            k_sel = k_sel.transpose(0, 1)
            attn = q_sel @ k_sel.transpose(-2, -1)
        elif self.agg_type == 'GAT':
            q_sel = q[G1].transpose(0, 1).unsqueeze(-1)
            k_sel = k[G2].transpose(0, 1).unsqueeze(1)
            attn = self.leakyrelu(q_sel + k_sel)

        if M is not None:
            attn = attn.masked_fill(~M.to(torch.bool), -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        v_sel = v[G2].transpose(0, 1)
        out[G1] = (attn @ v_sel).transpose(0, 1).contiguous().view(G1.size(0), -1)
        return out, None if not need_weights else attn

    def sparse_forward(self, q, k, v, edge_index, out, need_weights: bool = False):
        attn = None
        if self.agg_type == 'Trans':
            score = (q[edge_index[0]] * k[edge_index[1]]).sum(-1)
        elif self.agg_type == 'GAT':
            score = q[edge_index[0]] + k[edge_index[1]]
            score = self.leakyrelu(score)
        elif self.agg_type == 'GCN':
            if self.use_cache:
                if self.gcn_cache is None:
                    self.gcn_cache = calculate_norm_A(edge_index)
                edge_index, attn = self.gcn_cache
            else:
                edge_index, attn = calculate_norm_A(edge_index)
        else:
            raise ValueError(f'DualAttention: unknown agg_type {self.agg_type}')

        if self.agg_type in ('GAT', 'Trans'):
            score_max = scatter(score.detach(), index=edge_index[0], dim=0, reduce='max')
            score = score - score_max[edge_index[0]]
            score = torch.exp(score.unsqueeze(-1))
            score_sum = scatter(score, index=edge_index[0], dim=0, reduce='sum')
            attn = score / score_sum[edge_index[0]]
            attn = self.attn_dropout(attn)

        msg = attn * v[edge_index[1]]
        out = out.view(-1, self.n_head, self.h_dim // self.n_head)
        out = scatter(msg, index=edge_index[0], dim=0, out=out, reduce='sum').view(-1, self.h_dim)
        return out, None if not need_weights else attn

    def forward(self, x, edge_mask, need_weights: bool = False):
        N = x.size(0)
        d = self.h_dim // self.n_head

        if self.agg_type == 'Trans':
            q = self.lin_q(x).view(N, self.n_head, d) / self.temperature
            k = self.lin_k(x).view(N, self.n_head, d)
        elif self.agg_type == 'GAT':
            h = self.lin_a(x).view(N, self.n_head, d)
            q = (h * self.alpha1).sum(-1)
            k = (h * self.alpha2).sum(-1)
        else:
            q = k = None
        v = self.lin_v(x).view(N, self.n_head, d)

        out = torch.zeros(x.size(0), self.h_dim, device=x.device, dtype=x.dtype)
        for em in edge_mask:
            if isinstance(em, torch.Tensor):
                out, _ = self.sparse_forward(q, k, v, em, out, need_weights)
            else:
                out, _ = self.dense_forward(q, k, v, em, out, need_weights)
        return out, None


class M3DLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_head: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        bias: bool = True,
        use_cache: bool = False,
        use_res: bool = False,
        norm_type: str = 'ln',
        norm_pos: str = 'pre',
        local_type: str = 'GAT',
    ):
        super().__init__()
        self.n_head = n_head
        self.h_dim = h_dim
        self.norm_pos = norm_pos
        self.use_res = use_res

        if norm_pos == 'pre':
            self.norm1 = self._make_norm(norm_type, h_dim, in_dim, pre=True)
        elif norm_pos == 'post':
            self.norm1 = self._make_norm(norm_type, h_dim, in_dim, pre=False)
        else:
            self.norm1 = None

        self.attn_moe = BiAttentionMoeLayer(
            share_experts=[
                DualAttention(
                    in_dim=in_dim,
                    h_dim=h_dim,
                    num_heads=n_head,
                    dropout=attn_dropout,
                    bias=bias,
                    use_cache=use_cache,
                    agg_type=local_type,
                )
            ],
            router_experts=[
                DualAttention(
                    in_dim, h_dim, n_head, attn_dropout, bias, use_cache=False, agg_type='Trans'),
                DualAttention(
                    in_dim, h_dim, n_head, attn_dropout, bias, use_cache=False, agg_type='Trans'),
            ],
            gate_weights=nn.Parameter(torch.zeros([1, in_dim])),
            alpha_weights=nn.Parameter(torch.zeros([1, in_dim])),
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.res = nn.Linear(in_dim, h_dim, bias=bias) if use_res else None
        self.act = nn.ReLU()

    @staticmethod
    def _make_norm(norm_type: str, h_dim: int, in_dim: int, pre: bool):
        if norm_type == 'ln':
            return nn.LayerNorm(h_dim, eps=1e-6)
        if norm_type == 'rms':
            return RMSNorm(h_dim, eps=1e-6, bias=True)
        if norm_type == 'bn':
            return nn.BatchNorm1d(in_dim if pre else h_dim)
        return None

    def reset_parameters(self):
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        self.attn_moe.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()

    def forward(self, x, edge_masks, need_weights: bool = False):
        residual = x
        if self.norm_pos == 'pre' and self.norm1 is not None:
            x = self.norm1(x)
        x = self.attn_moe(x, edge_masks=edge_masks, need_weights=need_weights)
        x = self.dropout1(self.act(x))
        if self.res is not None:
            x = x + self.res(residual)
        if self.norm_pos == 'post' and self.norm1 is not None:
            x = self.norm1(x)
        return x


class M3DphormerEncoder(torch.nn.Module):
    """Backbone only: hierarchical-mask stack + deep residual accumulation (no classifier)."""

    def __init__(
        self,
        n_global: int,
        x_dim: int,
        h_dim: int,
        n_head: int,
        layers: int,
        dropout: float,
        attn_dropout: float,
        local_type: str = 'GAT',
        learn_global: bool = False,
        use_cache: bool = False,
        use_res: bool = True,
        norm_type: str = 'ln',
        norm_pos: str = 'pre',
    ):
        super().__init__()
        self.n_global = n_global
        self.learn_global = learn_global
        self.lin = nn.Linear(x_dim, h_dim)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                M3DLayer(
                    h_dim,
                    h_dim,
                    n_head,
                    dropout,
                    attn_dropout,
                    local_type=local_type,
                    use_cache=use_cache,
                    use_res=use_res,
                    norm_type=norm_type,
                    norm_pos=norm_pos,
                ))
        if learn_global:
            self.global_embed = nn.Embedding(n_global, h_dim)
        else:
            self.global_embed = None
        self.in_dropout = nn.Dropout(p=dropout)
        self.in_act = nn.ReLU()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.global_embed is not None:
            self.global_embed.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, edge_masks) -> torch.Tensor:
        x = self.lin(x)
        x = self.in_dropout(x + self.in_act(x))
        if self.learn_global and self.global_embed is not None:
            x = x.clone()
            x[-self.n_global:] = self.global_embed.weight
        out = x.clone()
        for layer in self.layers:
            x = layer(x, edge_masks)
            out = out + x
        return out
