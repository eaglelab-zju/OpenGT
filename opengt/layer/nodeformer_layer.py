import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_layer

from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

import math
import numpy as np

BIG_CONSTANT = 1e8

def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
	nb_full_blocks = int(m/d)
	block_list = []
	current_seed = seed
	for _ in range(nb_full_blocks):
		torch.manual_seed(current_seed)
		if struct_mode:
			q = create_products_of_givens_rotations(d, current_seed)
		else:
			unstructured_block = torch.randn((d, d))
			q, _ = torch.qr(unstructured_block)
			q = torch.t(q)
		block_list.append(q)
		current_seed += 1
	remaining_rows = m - nb_full_blocks * d
	if remaining_rows > 0:
		torch.manual_seed(current_seed)
		if struct_mode:
			q = create_products_of_givens_rotations(d, current_seed)
		else:
			unstructured_block = torch.randn((d, d))
			q, _ = torch.linalg.qr(unstructured_block)
			q = torch.t(q)
		block_list.append(q[0:remaining_rows])
	final_matrix = torch.vstack(block_list)

	current_seed += 1
	torch.manual_seed(current_seed)
	if scaling == 0:
		multiplier = torch.norm(torch.randn((m, d)), dim=1)
	elif scaling == 1:
		multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
	else:
		raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

	return torch.matmul(torch.diag(multiplier), final_matrix)

def create_products_of_givens_rotations(dim, seed):
	nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
	q = np.eye(dim, dim)
	np.random.seed(seed)
	for _ in range(nb_givens_rotations):
		random_angle = math.pi * np.random.uniform()
		random_indices = np.random.choice(dim, 2)
		index_i = min(random_indices[0], random_indices[1])
		index_j = max(random_indices[0], random_indices[1])
		slice_i = q[index_i]
		slice_j = q[index_j]
		new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
		new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
		q[index_i] = new_slice_i
		q[index_j] = new_slice_j
	return torch.tensor(q, dtype=torch.float32)

def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):
	del is_query
	if projection_matrix is None:
		return data.relu() + numerical_stabilizer
	else:
		ratio = 1.0 / torch.sqrt(
			torch.tensor(projection_matrix.shape[0], torch.float32)
		)
		data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)
		return data_dash.relu() + numerical_stabilizer

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
	data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
	data = data_normalizer * data
	ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
	data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
	diag_data = torch.square(data)
	diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
	diag_data = diag_data / 2.0
	diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
	last_dims_t = len(data_dash.shape) - 1
	attention_dims_t = len(data_dash.shape) - 3
	if is_query:
		data_dash = ratio * (
			torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
		)
	else:
		data_dash = ratio * (
			torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
					dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
		)
	return data_dash

def numerator(qs, ks, vs):
	kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper
	return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)

def denominator(qs, ks):
	all_ones = torch.ones([ks.shape[0]]).to(qs.device)
	ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper
	return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

def numerator_gumbel(qs, ks, vs):
	kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs) # kvs refers to U_k in the paper
	return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)

def denominator_gumbel(qs, ks):
	all_ones = torch.ones([ks.shape[0]]).to(qs.device)
	ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones) # ks_sum refers to O_k in the paper
	return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)

def kernelized_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None, tau=0.25, return_weight=True):
	'''
	fast computation of all-pair attentive aggregation with linear complexity
	input: query/key/value [B, N, H, D]
	return: updated node emb, attention weight (for computing edge loss)
	B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
	M = random feature dimension, D = hidden size
	'''
	query = query / math.sqrt(tau)
	key = key / math.sqrt(tau)
	query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
	key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
	query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
	key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
	value = value.permute(1, 0, 2, 3) # [N, B, H, D]

	# compute updated node emb, this step requires O(N)
	z_num = numerator(query_prime, key_prime, value)
	z_den = denominator(query_prime, key_prime)

	z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
	z_den = z_den.permute(1, 0, 2)
	z_den = torch.unsqueeze(z_den, len(z_den.shape))
	z_output = z_num / z_den # [B, N, H, D]

	if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
		start, end = edge_index
		query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
		edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
		edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
		attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
		edge_attn_dem = attn_normalizer[end]  # [E, B, H]
		edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
		A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

		return z_output, A_weight

	else:
		return z_output

def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None,
								K=10, tau=0.25, return_weight=True):
	'''
	fast computation of all-pair attentive aggregation with linear complexity
	input: query/key/value [B, N, H, D]
	return: updated node emb, attention weight (for computing edge loss)
	B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
	M = random feature dimension, D = hidden size, K = number of Gumbel sampling
	'''
	query = query / math.sqrt(tau)
	key = key / math.sqrt(tau)
	query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
	key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
	query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
	key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
	value = value.permute(1, 0, 2, 3) # [N, B, H, D]

	# compute updated node emb, this step requires O(N)
	gumbels = (
		-torch.empty(key_prime.shape[:-1]+(K, ), memory_format=torch.legacy_contiguous_format).exponential_().log()
	).to(query.device) / tau # [N, B, H, K]
	key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4) # [N, B, H, K, M]
	z_num = numerator_gumbel(query_prime, key_t_gumbel, value) # [N, B, H, K, D]
	z_den = denominator_gumbel(query_prime, key_t_gumbel) # [N, B, H, K]

	z_num = z_num.permute(1, 0, 2, 3, 4) # [B, N, H, K, D]
	z_den = z_den.permute(1, 0, 2, 3) # [B, N, H, K]
	z_den = torch.unsqueeze(z_den, len(z_den.shape))
	z_output = torch.mean(z_num / z_den, dim=3) # [B, N, H, D]

	if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
		start, end = edge_index
		query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
		edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
		edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
		attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
		edge_attn_dem = attn_normalizer[end]  # [E, B, H]
		edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
		A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

		return z_output, A_weight

	else:
		return z_output

def add_conv_relational_bias(x, edge_index, b, trans='sigmoid'):
	'''
	compute updated result by the relational bias of input adjacency
	the implementation is similar to the Graph Convolution Network with a (shared) scalar weight for each edge
	'''
	row, col = edge_index
	d_in = degree(col, x.shape[1]).float()
	d_norm_in = (1. / d_in[col]).sqrt()
	d_out = degree(row, x.shape[1]).float()
	d_norm_out = (1. / d_out[row]).sqrt()
	conv_output = []
	for i in range(x.shape[2]):
		if trans == 'sigmoid':
			b_i = b[i].sigmoid()
		elif trans == 'identity':
			b_i = b[i]
		else:
			raise NotImplementedError
		value = torch.ones_like(row) * b_i * d_norm_in * d_norm_out
		adj_i = SparseTensor(row=col, col=row, value=value, sparse_sizes=(x.shape[1], x.shape[1]))
		conv_output.append( matmul(adj_i, x[:, :, i]) )  # [B, N, D]
	conv_output = torch.stack(conv_output, dim=2) # [B, N, H, D]
	return conv_output

@register_layer("NodeFormerConv")
class NodeFormerConv(nn.Module):
	'''
	One layer of NodeFormer that attentive aggregates all nodes over a latent graph
	Adapted from https://github.com/qitianwu/NodeFormer

	Parameters:
		dim_in (int): Number of input features.
		dim_out (int): Number of output features.
		config (object): Configuration object containing hyperparameters.
		    - rb_order (int): Order of relational bias
			- rb_trans (str): Transformation for relational bias, either 'sigmoid' or 'identity'
			- kernel_trans (str): Type of kernel transformation, either 'softmax' or 'relu'
			- projection_matrix_type (str): Type of projection matrix, either 'a' or None
			- nb_random_features (int): Number of random features
			- use_gumbel (bool): Whether to use Gumbel sampling
			- nb_gumbel_sample (int): Number of Gumbel samples
			- use_edge_loss (bool): Whether to use edge loss
			- use_bn (bool): Whether to use batch normalization
			- use_residual (bool): Whether to use residual connection
			- use_act (bool): Whether to use activation function
			- dropout (float): Dropout rate
			- tau (float): Temperature parameter for Gumbel softmax
			- n_heads (int): Number of attention heads
	
	Input:
		batch.x (torch.Tensor): Input node features.
		batch.adjs (list): List of adjacency matrices for different orders of relational bias.
		batch.extra_loss (torch.Tensor): Aggregated extra loss for edge regularization in previous layers.
	
	Output:
		ret.x (torch.Tensor): Output node features after applying the NodeFormer layer.
		ret.extra_loss (torch.Tensor): Aggregated extra loss for edge regularization.
	

	return: node embeddings for next layer, edge loss at this layer
	'''
	def __init__(self, dim_in, dim_out, config):
	# config.n_heads, kernel_transformation=softmax_kernel_transformation, projection_matrix_type='a',
	#              nb_random_features=10, use_gumbel=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True):
		super(NodeFormerConv, self).__init__()
		self.Wk = nn.Linear(dim_in, dim_out * config.n_heads)
		self.Wq = nn.Linear(dim_in, dim_out * config.n_heads)
		self.Wv = nn.Linear(dim_in, dim_out * config.n_heads)
		self.Wo = nn.Linear(dim_out * config.n_heads, dim_out)
		if config.rb_order >= 1:
			self.b = torch.nn.Parameter(torch.FloatTensor(config.rb_order, config.n_heads), requires_grad=True)

		if config.rb_order >= 1:
			if config.rb_trans == 'sigmoid':
				torch.nn.init.constant_(self.b, 0.1)
			elif config.rb_trans == 'identity':
				torch.nn.init.constant_(self.b, 1.0)
		
		self.dim_out = dim_out
		self.n_heads = config.n_heads
		if config.kernel_trans == 'softmax':
			self.kernel_transformation = softmax_kernel_transformation
		elif config.kernel_trans == 'relu':
			self.kernel_transformation = relu_kernel_transformation
		else:
			raise ValueError('Unknown kernel transformation')

		if config.batch_norm:
			self.norm = nn.LayerNorm(config.dim_hidden)
		
		self.projection_matrix_type = config.projection_matrix_type
		self.nb_random_features = config.nb_random_features
		self.use_gumbel = config.use_gumbel
		self.nb_gumbel_sample = config.nb_gumbel_sample
		self.rb_order = config.rb_order
		self.rb_trans = config.rb_trans
		self.use_edge_loss = config.use_edge_loss
		self.tau = config.tau
		self.use_residual = config.use_residual
		self.use_bn = config.batch_norm
		self.use_act = config.use_act
		self.dropout = config.dropout

	def forward(self, batch):
		z = batch.x.unsqueeze(0)
		adjs = batch.adjs
		B, N = z.size(0), z.size(1)
		query = self.Wq(z).reshape(-1, N, self.n_heads, self.dim_out)
		key = self.Wk(z).reshape(-1, N, self.n_heads, self.dim_out)
		value = self.Wv(z).reshape(-1, N, self.n_heads, self.dim_out)

		if self.projection_matrix_type is None:
			projection_matrix = None
		else:
			dim = query.shape[-1]
			seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
			projection_matrix = create_projection_matrix(
				self.nb_random_features, dim, seed=seed).to(query.device)

		# compute all-pair message passing update and attn weight on input edges, requires O(N) or O(N + E)
		if self.use_gumbel and self.training:  # only using Gumbel noise for training
			z_next, weight = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,adjs[0],
												  self.nb_gumbel_sample, self.tau, True)
		else:
			z_next, weight = kernelized_softmax(query, key, value, self.kernel_transformation, projection_matrix, adjs[0],
												self.tau, True)

		# compute update by relational bias of input adjacency, requires O(E)
		for i in range(self.rb_order):
			z_next += add_conv_relational_bias(value, adjs[i], self.b[i], self.rb_trans)

		# aggregate results of multiple heads
		z_next = self.Wo(z_next.flatten(-2, -1))

		if self.use_residual:
			z_next += z
		if self.use_bn:
			z_next = self.norm(z_next)
		if self.use_act:
			z_next = F.elu(z_next)
		z_next = F.dropout(z_next, p=self.dropout, training=self.training)

		ret = batch.clone()
		ret.x = z_next.squeeze(0)
		
		if self.use_edge_loss: # compute edge regularization loss on input adjacency
			row, col = adjs[0]
			d_in = degree(col, query.shape[1]).float()
			d_norm = 1. / d_in[col]
			d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
			link_loss = torch.mean(weight.log() * d_norm_)

			ret.extra_loss += link_loss


		return ret