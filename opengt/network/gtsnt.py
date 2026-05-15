# GT-SNT adapted for OpenGT GraphGym (node classification).
# Upstream: https://github.com/Zhhuizhe/GT-SNT (MIT)
#
# OpenGT changes:
# - No final ``lin_out``; logits from GraphGym ``post_mp``.
# - Input is already ``cfg.gnn.dim_inner`` after ``FeatureEncoder``; residual skips use identity.
# - Requires ``cfg.share.num_nodes`` (set in ``main.py`` after ``create_loader()``).

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch import Tensor
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from opengt.encoder.feature_encoder import FeatureEncoder
from opengt.layer.gtsnt_components import GTSNTGCNConv, SMultiHeadAttention, SNT


class _GTSNTBackbone(nn.Module):
    """Vendored stack from GT-SNT ``GTSNT`` without ``lin_in`` / ``lin_out``."""

    def __init__(
        self,
        num_nodes: int,
        hid_channels: int,
        cb_channels: int,
        num_layers: int,
        num_heads: int,
        T: int,
        init_beta: float,
        neuron: str,
        v_threshold: float,
        dropout: float,
        attn_dropout: float,
        maximum_codes_num: Optional[int],
        normalize: bool,
        enable_residual: bool,
        enable_norm: bool,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        self.maximum_codes_num = maximum_codes_num
        self.enable_norm = enable_norm
        self.enable_residual = enable_residual

        channels = [hid_channels] * (num_layers + 1)
        self.trans_convs = nn.ModuleList()
        self.cb_convs = nn.ModuleList()
        self.mpnn_convs = nn.ModuleList()
        self.norm_convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.cb_convs.append(
                SNT(
                    num_nodes,
                    cb_channels,
                    T,
                    neuron,
                    v_threshold=v_threshold,
                    init_beta=init_beta,
                )
            )
            self.trans_convs.append(
                SMultiHeadAttention(
                    channels[i],
                    channels[i + 1],
                    cb_channels,
                    num_heads,
                    dropout,
                    attn_dropout,
                )
            )
            self.mpnn_convs.append(
                GTSNTGCNConv(
                    channels[i],
                    channels[i + 1],
                    normalize=normalize,
                )
            )
            self.norm_convs.append(nn.BatchNorm1d(channels[i + 1]))

        self._cached_cb: List = []
        self._cached_code_indices: List = []

    def reset_membrane_potential(self):
        for conv in self.cb_convs:
            conv.neuron_reset()

    def reset_parameters(self):
        for i in range(len(self.trans_convs)):
            self.trans_convs[i].reset_parameters()
            self.mpnn_convs[i].reset_parameters()
            self.cb_convs[i].reset_parameters()
            self.norm_convs[i].reset_parameters()
        self._cached_cb, self._cached_code_indices = [], []
        self.reset_membrane_potential()

    def forward(
        self,
        embd: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        self._cached_cb, self._cached_code_indices = [], []
        for i in range(self.num_layers):
            embd_res = embd

            embd_mpnn = self.mpnn_convs[i](embd, edge_index, edge_weight).relu()
            embd_mpnn = F.dropout(embd_mpnn, self.dropout, training=self.training)

            codebook, code_indices, counts = self.cb_convs[i](edge_index, edge_weight)

            if (
                self.maximum_codes_num is not None
                and codebook.size(0) > self.maximum_codes_num
            ):
                sorted_counts_indices = torch.argsort(counts, descending=True)
                mask = sorted_counts_indices[: self.maximum_codes_num]
            else:
                mask = None
            embd_trans = self.trans_convs[i].vq_attn_block(
                embd_mpnn, embd_mpnn, codebook, code_indices, mask
            )
            embd = embd_trans + embd_mpnn

            if self.enable_residual:
                embd = embd + embd_res

            if self.enable_norm:
                embd = self.norm_convs[i](embd)

            self._cached_cb.append(codebook)
            self._cached_code_indices.append((code_indices, counts))

        self.reset_membrane_potential()
        return embd


@register_network('GTSNT')
class GTSNT(nn.Module):
    """GT-SNT: MPNN + spiking codebook + VQ attention; classification via ``post_mp``."""

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
                'GTSNT expects gt.dim_hidden == gnn.dim_inner == dim_in after encoder/pre_mp, '
                f'got dim_hidden={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} dim_in={dim_in_enc}'
            )

        hidden = cfg.gt.dim_hidden
        n_heads = int(cfg.gt.n_heads)
        if hidden % n_heads != 0:
            raise ValueError(
                f'GTSNT: dim_hidden={hidden} must be divisible by gt.n_heads={n_heads}'
            )

        num_nodes = int(getattr(cfg.share, 'num_nodes', 0))
        if num_nodes <= 0:
            raise ValueError(
                'GTSNT requires cfg.share.num_nodes > 0. '
                'OpenGT main.py sets this after create_loader(); if you run the model '
                'outside main, set cfg.share.num_nodes to the graph node count.'
            )

        mcn = getattr(cfg.gtsnt, 'maximum_codes_num', None)
        if mcn is not None and mcn <= 0:
            mcn = None

        self.backbone = _GTSNTBackbone(
            num_nodes=num_nodes,
            hid_channels=hidden,
            cb_channels=int(cfg.gtsnt.cb_channels),
            num_layers=int(cfg.gt.layers),
            num_heads=n_heads,
            T=int(cfg.gtsnt.T),
            init_beta=float(cfg.gtsnt.init_beta),
            neuron=str(cfg.gtsnt.neuron),
            v_threshold=float(cfg.gtsnt.v_threshold),
            dropout=float(cfg.gt.dropout),
            attn_dropout=float(cfg.gt.attn_dropout),
            maximum_codes_num=mcn,
            normalize=bool(cfg.gtsnt.normalize),
            enable_residual=bool(cfg.gtsnt.enable_residual),
            enable_norm=bool(cfg.gtsnt.enable_norm),
        )

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=hidden, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        edge_weight = getattr(batch, 'edge_weight', None)
        batch.x = self.backbone(batch.x, batch.edge_index, edge_weight)
        return self.post_mp(batch)
