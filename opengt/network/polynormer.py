# Vendored from PyTorch Geometric 2.7.x (Polynormer model), adapted for GraphGym.
# SPDX: PyG uses MIT License — see upstream: https://github.com/pyg-team/pytorch_geometric
# Source reference: torch_geometric/nn/models/polynormer.py
#
# OpenGT changes vs upstream:
# - Removed pred_local / pred_global and final log_softmax; logits come from GraphGym
#   ``post_mp`` like other networks here.
# - Two-stage local -> global schedule is handled inside ``train()`` to keep the
#   GraphGym training loop model-agnostic.
# - Dropped unused ``lin_in`` (present in upstream but never referenced in forward).
# - Global attention class is ``opengt.layer.polynormer_attention`` (register_layer).
from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch import Tensor
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_batch

from opengt.encoder.feature_encoder import FeatureEncoder
from opengt.layer.polynormer_attention import PolynormerAttention


@register_network("Polynormer")
class Polynormer(torch.nn.Module):
    """Polynormer local (+ optional global) stack + GraphGym encoder/head."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in_enc = self.encoder.dim_in

        hidden = cfg.gnn.dim_inner
        if cfg.gt.dim_hidden != hidden:
            raise ValueError(
                "Polynormer expects gt.dim_hidden == gnn.dim_inner, got "
                f"{cfg.gt.dim_hidden} vs {hidden}"
            )

        heads = int(cfg.polynormer.heads)
        self._two_stage = bool(getattr(cfg.polynormer, "two_stage", False))
        self._local_epochs = int(getattr(cfg.polynormer, "local_epochs", 0))
        self._global_epochs = int(getattr(cfg.polynormer, "global_epochs", 0))
        self._default_use_global = bool(getattr(cfg.polynormer, "use_global", True))
        self._stage_epoch = -1
        self._use_global = False if self._two_stage else self._default_use_global
        self.in_drop = cfg.polynormer.in_dropout
        self.dropout = cfg.polynormer.dropout
        self.pre_ln = cfg.polynormer.pre_ln
        self.post_bn = cfg.polynormer.post_bn
        self.beta = cfg.polynormer.beta

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.post_bn:
            self.post_bns = torch.nn.ModuleList()

        inner_channels = heads * hidden
        self.inner_channels = inner_channels
        local_layers = int(cfg.gt.layers)
        local_attn = cfg.polynormer.local_attn

        self.h_lins.append(torch.nn.Linear(dim_in_enc, inner_channels))
        if local_attn:
            self.local_convs.append(
                GATConv(dim_in_enc, hidden, heads=heads, concat=True,
                        add_self_loops=False, bias=False))
        else:
            self.local_convs.append(
                GCNConv(dim_in_enc, inner_channels, cached=False,
                        normalize=True))

        self.lins.append(torch.nn.Linear(dim_in_enc, inner_channels))
        self.lns.append(torch.nn.LayerNorm(inner_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(dim_in_enc))
        if self.post_bn:
            self.post_bns.append(torch.nn.BatchNorm1d(inner_channels))

        for _ in range(local_layers - 1):
            self.h_lins.append(torch.nn.Linear(inner_channels, inner_channels))
            if local_attn:
                self.local_convs.append(
                    GATConv(inner_channels, hidden, heads=heads,
                            concat=True, add_self_loops=False, bias=False))
            else:
                self.local_convs.append(
                    GCNConv(inner_channels, inner_channels, cached=False,
                            normalize=True))

            self.lins.append(torch.nn.Linear(inner_channels, inner_channels))
            self.lns.append(torch.nn.LayerNorm(inner_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads * hidden))
            if self.post_bn:
                self.post_bns.append(torch.nn.BatchNorm1d(inner_channels))

        self.ln = torch.nn.LayerNorm(inner_channels)

        self.global_attn = torch.nn.ModuleList()
        for _ in range(cfg.polynormer.global_layers):
            self.global_attn.append(
                PolynormerAttention(
                    channels=hidden,
                    heads=heads,
                    head_channels=hidden,
                    beta=cfg.polynormer.beta,
                    dropout=cfg.polynormer.global_dropout,
                    qk_shared=cfg.polynormer.qk_shared,
                ))

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=inner_channels, dim_out=dim_out)

        self.reset_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self._two_stage:
                self._stage_epoch += 1
                self._use_global = self._stage_epoch >= self._local_epochs
            else:
                self._use_global = self._default_use_global
        return self

    def reset_parameters(self) -> None:
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for attn in self.global_attn:
            attn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.post_bn:
            for p_bn in self.post_bns:
                p_bn.reset_parameters()
        self.ln.reset_parameters()
        if hasattr(self.post_mp, 'reset_parameters'):
            self.post_mp.reset_parameters()
        self._stage_epoch = -1
        self._use_global = False if self._two_stage else self._default_use_global

    def _polynormer_stack(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor],
    ) -> Tensor:
        x = F.dropout(x, p=self.in_drop, training=self.training)

        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = (1 - self.beta) * self.lns[i](h * x) + self.beta * x
            x_local = x_local + x

        if self._use_global:
            if batch is None:
                raise ValueError(
                    "Polynormer(use_global=True) requires a batch vector."
                )
            b_sorted, sort_idx = batch.sort()
            rev_perm = torch.empty_like(sort_idx)
            rev_perm[sort_idx] = torch.arange(
                len(sort_idx), device=sort_idx.device)
            x_local = self.ln(x_local[sort_idx])
            x_global, mask = to_dense_batch(x_local, b_sorted)
            for attn in self.global_attn:
                x_global = attn(x_global, mask)
            return x_global[mask][rev_perm]

        return x_local

    def forward(self, batch):
        batch = self.encoder(batch)
        batch_idx = getattr(batch, "batch", None)
        if self._use_global and batch_idx is None:
            n = batch.num_nodes if getattr(batch, "num_nodes", None) is not None \
                else batch.x.size(0)
            batch_idx = torch.zeros(
                n, dtype=torch.long, device=batch.x.device,
            )
        batch.x = self._polynormer_stack(batch.x, batch.edge_index, batch_idx)
        return self.post_mp(batch)
