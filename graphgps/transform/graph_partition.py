import torch
from torch_geometric.data import Data
import torch_geometric
from torch_sparse import SparseTensor  # for propagation
import numpy as np
import networkx as nx
import re
import pymetis

def random_walk(A, n_iter):
    # Geometric diffusion features with Random Walk
    Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
    RW = A * Dinv
    M = RW
    M_power = M
    # Iterate
    PE = [torch.diagonal(M)]
    for _ in range(n_iter-1):
        M_power = torch.matmul(M_power, M)
        PE.append(torch.diagonal(M_power))
    PE = torch.stack(PE, dim=-1)
    return PE

def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
    # return k-hop subgraphs for all nodes in the graph
    if is_directed:
        row, col = edge_index
        birow, bicol = torch.cat([row, col]), torch.cat([col, row])
        edge_index = torch.stack([birow, bicol])
    else:
        row, col = edge_index
    sparse_adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    # each one contains <= i hop masks
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
                           device=edge_index.device)]
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    return node_mask


def random_subgraph(g, n_patches, num_hops=1):
    membership = np.arange(g.num_nodes)
    np.random.shuffle(membership)
    membership = torch.tensor(membership % n_patches)
    max_patch_id = torch.max(membership)+1
    membership = membership+(n_patches-max_patch_id)

    node_mask = torch.stack([membership == i for i in range(n_patches)])

    if num_hops > 0:
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops)
        node_mask[subgraphs_batch] += k_hop_node_mask[subgraphs_node_mapper]

    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask


def metis_subgraph(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
    if is_directed:
        if g.num_nodes < n_patches:
            membership = torch.arange(g.num_nodes)
        else:
            G = torch_geometric.utils.to_networkx(g, to_undirected="lower")
            cuts, membership = pymetis.part_graph(n_patches, adjacency = G)
    else:
        if g.num_nodes < n_patches:
            membership = torch.randperm(n_patches)
        else:
            # data augmentation
            adjlist = g.edge_index.t()
            arr = torch.rand(len(adjlist))
            selected = arr > drop_rate
            G = nx.Graph()
            G.add_nodes_from(np.arange(g.num_nodes))
            G.add_edges_from(adjlist[selected].tolist())
            # metis partition
            cuts, membership = pymetis.part_graph(n_patches, adjacency = G)

    assert len(membership) >= g.num_nodes
    membership = torch.tensor(np.array(membership[:g.num_nodes]))
    max_patch_id = torch.max(membership)+1
    membership = membership+(n_patches-max_patch_id)

    node_mask = torch.stack([membership == i for i in range(n_patches)])

    if num_hops > 0:
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops, is_directed)
        node_mask.index_add_(0, subgraphs_batch,
                             k_hop_node_mask[subgraphs_node_mapper])

    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask


class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')]+'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            return 1+getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

def cal_coarsen_adj(subgraphs_nodes_mask):
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t())
    return coarsen_adj


def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]
                      ] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected)*num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs

def metis_partition(data, n_patches):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N < n_patches:
        membership = torch.randperm(n_patches)
    else:
        # data augmentation
        adjlist = data.edge_index.t()
        G = nx.Graph()
        G.add_nodes_from(np.arange(N))
        G.add_edges_from(adjlist.tolist())
        # metis partition
        cuts, membership = pymetis.part_graph(n_patches, G)

    assert len(membership) >= N
    membership = torch.tensor(membership[:N])


    patch = []
    max_patch_size = -1
    for i in range(n_patches):
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [N] * (max_patch_size - l)

    patch = torch.tensor(patch)

    return patch

class GraphPartitionTransform(object):
    def __init__(self, n_patches, metis=True, drop_rate=0.0, num_hops=1, is_directed=False, patch_rw_dim=0, patch_num_diff=0):
        super().__init__()
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis = metis

    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        # Iterate
        for _ in range(self.patch_num_diff-1):
            M_power = torch.matmul(M_power, M)
        return M_power

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if self.metis:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=self.n_patches, drop_rate=self.drop_rate, num_hops=self.num_hops, is_directed=self.is_directed)
            data.patch = metis_partition(data, n_patches=self.n_patches)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=self.n_patches, num_hops=self.num_hops)
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)
        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges, num_selected=self.n_patches, num_nodes=data.num_nodes)

        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_rw_dim > 0:
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            if self.patch_num_diff > -1:
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0]
        mask = torch.zeros(self.n_patches).bool()
        mask[subgraphs_batch] = True
        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.mask = mask.unsqueeze(0)

        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        return data


