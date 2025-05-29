import torch

from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree, remove_self_loops, add_self_loops

# required by nodeformer

def adj_mul(adj_i, adj, N):
	if adj_i.shape[1] <= N * N / 100 or \
	   adj.shape[1] <= N * N / 100: # 1% density, sparse enough
		adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
		adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
		adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
		adj_j = adj_j.coalesce().indices()
		return adj_j
	else:
		adj_i_dense = torch.zeros((N, N), device=adj.device)
		adj_dense = torch.zeros((N, N), device=adj.device)
		adj_i_dense[adj_i[0], adj_i[1]] = 1
		adj_dense[adj[0], adj[1]] = 1
		adj_j_dense = torch.matmul(adj_i_dense, adj_dense)
		adj_j = adj_j_dense.nonzero(as_tuple=False).t()
		return adj_j

def generate_multihop_adj(data, cfg):
	if hasattr(data, 'num_nodes'):
		N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
	else:
		N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
	
	data.adjs = []
	adj, _ = remove_self_loops(data.edge_index)
	adj, _ = add_self_loops(adj, num_nodes=N)
	adj0 = adj
	data.adjs.append(adj)
	for i in range(cfg.prep.rb_order - 1): # edge_index of high order adjacency # args.rb_order == 2 
		adj = adj_mul(adj, adj0, N)
		data.adjs.append(adj)
	return data