# GraphGym wrapper for M3Dphormer (hierarchical-mask graph transformer).
# Core layers vendored in ``opengt/layer/m3dphormer_modules.py`` from
# https://github.com/null-xyj/M3Dphormer (NeurIPS 2025).
#
# OpenGT integration:
# - METIS clustering via ``pymetis`` (repo standard) instead of upstream ``metis`` package.
# - No built-in classifier / no duplicate log-softmax; logits only from ``post_mp``.
# - Single full graph per batch (Planetoid / Critical), same scope as HubGT.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
import pymetis
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import add_self_loops, to_undirected

from opengt.encoder.feature_encoder import FeatureEncoder
from opengt.layer.m3dphormer_modules import M3DphormerEncoder


def _num_graphs(batch) -> int:
    ptr = getattr(batch, 'ptr', None)
    if ptr is not None:
        return int(ptr.numel() - 1)
    b = getattr(batch, 'batch', None)
    if b is None:
        return 1
    return int(b.max().item()) + 1


def _metis_membership(num_nodes: int, edge_index: torch.Tensor, num_clusters: int) -> torch.Tensor:
    device = edge_index.device
    if num_nodes < num_clusters:
        memb = torch.randperm(num_clusters, device=device)[:num_nodes]
    else:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.t().detach().cpu().numpy().tolist())
        _, memb_np = pymetis.part_graph(num_clusters, adjacency=G)
        memb = torch.as_tensor(np.asarray(memb_np[:num_nodes], dtype=np.int64), device=device)
    max_patch_id = int(memb.max().item()) + 1
    memb = memb + (num_clusters - max_patch_id)
    return memb.long()


def _cluster_topology(
    num_nodes: int,
    edge_index: torch.Tensor,
    num_clusters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed METIS membership and undirected cluster–node edges (no node features)."""
    membership = _metis_membership(num_nodes, edge_index, num_clusters)
    device = edge_index.device
    dtype = edge_index.dtype
    cluster_ids = num_nodes + membership
    node_ids = torch.arange(num_nodes, device=device, dtype=dtype)
    node2cluster = torch.stack([cluster_ids, node_ids])
    tot = num_nodes + num_clusters
    node2cluster = to_undirected(node2cluster.long(), num_nodes=tot)
    node2cluster = add_self_loops(node2cluster, num_nodes=tot)[0]
    return membership, node2cluster


def _cluster_features(node_x: torch.Tensor, membership: torch.Tensor, num_clusters: int) -> torch.Tensor:
    dim = node_x.size(-1)
    device = node_x.device
    dtype = node_x.dtype
    rows = []
    for k in range(num_clusters):
        sel = (membership == k).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            rows.append(torch.zeros(dim, device=device, dtype=dtype))
        else:
            rows.append(node_x[sel].mean(0))
    return torch.stack(rows, dim=0)


def _global_topology_and_labels(
    num_nodes: int,
    num_clusters: int,
    num_classes: int,
    global_nodes_per_class: int,
    train_mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, List, List[torch.Tensor]]:
    """Bipartite global edges + per-class train index lists (features recomputed each forward)."""
    y_flat = y.view(-1).long()
    train_idx = train_mask.view(-1).nonzero(as_tuple=True)[0].view(-1)
    label_idx = [train_idx[y_flat[train_idx] == c] for c in range(num_classes)]
    g_per = int(global_nodes_per_class)
    global_nodes = int(num_classes * g_per)

    n0: List[int] = []
    n1: List[int] = []
    for i, li in enumerate(label_idx):
        li_list = li.tolist()
        for j in range(g_per):
            gid = num_nodes + num_clusters + i * g_per + j
            n0.extend([gid] * len(li_list))
            n1.extend(li_list)
    node2global = torch.tensor([n0, n1], device=device, dtype=torch.long)

    g2n_0 = torch.arange(num_nodes, device=device, dtype=torch.long)
    g2n_1 = torch.arange(
        num_nodes + num_clusters,
        num_nodes + num_clusters + global_nodes,
        device=device,
        dtype=torch.long,
    )
    global2node: List[Optional[torch.Tensor]] = [g2n_0, g2n_1, None]
    return node2global, global2node, label_idx


def _global_features_from_labels(
    node_x: torch.Tensor,
    label_idx: List[torch.Tensor],
    num_classes: int,
    global_nodes_per_class: int,
) -> torch.Tensor:
    g_per = int(global_nodes_per_class)
    feat_rows: List[torch.Tensor] = []
    for c in range(num_classes):
        li = label_idx[c]
        if li.numel() == 0:
            proto = node_x.mean(0)
        else:
            proto = node_x[li].mean(0)
        for _ in range(g_per):
            feat_rows.append(proto)
    return torch.stack(feat_rows, dim=0)


@dataclass
class _AugmentCache:
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    membership: torch.Tensor
    node2cluster: torch.Tensor
    node2global: torch.Tensor
    global2node: List
    num_clusters: int
    num_classes: int
    global_nodes_per_class: int
    label_idx: List[torch.Tensor]


@register_network('M3Dphormer')
class M3Dphormer(nn.Module):
    """FeatureEncoder + M3Dphormer hierarchical stack; node logits from ``post_mp`` only."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in_enc = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in_enc, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in_enc = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in_enc:
            raise ValueError(
                'M3Dphormer expects gt.dim_hidden == gnn.dim_inner == dim after encoder/pre_mp, '
                f'got dim_hidden={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} dim_in={dim_in_enc}'
            )

        hidden = int(cfg.gt.dim_hidden)
        n_heads = int(cfg.gt.n_heads)
        if hidden % n_heads != 0:
            raise ValueError(
                f'M3Dphormer: dim_hidden={hidden} must be divisible by gt.n_heads={n_heads}'
            )

        num_classes = int(dim_out)
        self._num_classes = num_classes
        g_per = int(cfg.m3dphormer.global_nodes_per_class)
        n_global = num_classes * g_per
        n_layers = int(cfg.gt.layers)

        self.backbone = M3DphormerEncoder(
            n_global=n_global,
            x_dim=hidden,
            h_dim=hidden,
            n_head=n_heads,
            layers=n_layers,
            dropout=float(cfg.gt.dropout),
            attn_dropout=float(cfg.gt.attn_dropout),
            local_type=str(cfg.m3dphormer.local_type),
            learn_global=bool(cfg.m3dphormer.learn_global),
            use_cache=bool(cfg.m3dphormer.use_cache),
            use_res=bool(cfg.m3dphormer.use_res),
            norm_type=str(cfg.m3dphormer.norm_type),
            norm_pos=str(cfg.m3dphormer.norm_pos),
        )

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=hidden, dim_out=dim_out)

        self._aug: Optional[_AugmentCache] = None
        self._max_nodes = int(cfg.m3dphormer.max_nodes)

    def _ensure_batch_vector(self, batch):
        if getattr(batch, 'batch', None) is None:
            batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)

    def _build_augment(self, batch) -> _AugmentCache:
        n = batch.num_nodes
        if n > self._max_nodes:
            raise ValueError(
                f'M3Dphormer: num_nodes={n} exceeds m3dphormer.max_nodes={self._max_nodes}.'
            )
        ei = batch.edge_index
        device = ei.device
        num_clusters = int(cfg.m3dphormer.num_clusters)
        membership, node2cluster = _cluster_topology(n, ei, num_clusters)

        num_classes = self._num_classes
        g_per = int(cfg.m3dphormer.global_nodes_per_class)
        node2global, global2node, label_idx = _global_topology_and_labels(
            n,
            num_clusters,
            num_classes,
            g_per,
            batch.train_mask,
            batch.y,
            device,
        )

        return _AugmentCache(
            edge_index=ei.detach().clone(),
            train_mask=batch.train_mask.detach().clone(),
            membership=membership.detach().clone(),
            node2cluster=node2cluster.detach().clone(),
            node2global=node2global.detach().clone(),
            global2node=global2node,
            num_clusters=num_clusters,
            num_classes=num_classes,
            global_nodes_per_class=g_per,
            label_idx=[t.detach().clone() for t in label_idx],
        )

    def _cache_hit(self, batch) -> bool:
        if self._aug is None:
            return False
        if self._aug.edge_index.shape != batch.edge_index.shape:
            return False
        if not torch.equal(self._aug.edge_index, batch.edge_index):
            return False
        if not torch.equal(self._aug.train_mask, batch.train_mask):
            return False
        return True

    def forward(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        if _num_graphs(batch) != 1:
            raise NotImplementedError(
                'M3Dphormer supports a single graph per batch (full-batch transductive).'
            )

        self._ensure_batch_vector(batch)

        if not self._cache_hit(batch):
            self._aug = self._build_augment(batch)

        n = batch.num_nodes
        aug = self._aug
        assert aug is not None
        x = batch.x
        cluster_feat = _cluster_features(x, aug.membership, aug.num_clusters)
        if bool(cfg.m3dphormer.learn_global):
            global_feat = torch.zeros(
                (aug.num_classes * aug.global_nodes_per_class, x.size(-1)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            global_feat = _global_features_from_labels(
                x, aug.label_idx, aug.num_classes, aug.global_nodes_per_class)
        x_cat = torch.cat([x, cluster_feat, global_feat], dim=0)
        edge_masks = [
            [batch.edge_index],
            [aug.node2cluster],
            [aug.node2global, aug.global2node],
        ]
        h = self.backbone(x_cat, edge_masks)
        global_h = h[n + aug.num_clusters:]
        batch.x = h[:n]
        pred, true = self.post_mp(batch)

        if (
            self.training
            and getattr(batch, 'split', None) == 'train'
            and bool(cfg.m3dphormer.use_global_aux_loss)
        ):
            global_logits = self.post_mp.layer_post_mp(global_h)
            global_true = torch.arange(
                aug.num_classes, device=global_logits.device
            ).repeat_interleave(aug.global_nodes_per_class)
            n_train = max(int(batch.train_mask.sum().item()), 1)
            gamma = global_true.numel() / n_train
            aux_loss = F.cross_entropy(global_logits, global_true)
            aux_loss = aux_loss * gamma * float(cfg.m3dphormer.global_aux_loss_weight)
            return pred, true, aux_loss

        return pred, true
