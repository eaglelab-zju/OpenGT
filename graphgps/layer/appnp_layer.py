import torch
import torch_geometric.nn as pygnn

import torch.nn as nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg

@register_layer("appnpconv")
class APPNP(nn.Module):
	"""
	Wrapper layer for the APPNP layer from torch_geometric.nn.

	Parameters:
		dim_in (int): Number of input features. Handled by GraphGym.
		dim_out (int): Number of output features. Handled by GraphGym.
		K (int): Number of propagation steps. Default is 10.
		alpha (float): Teleport probability. Default is 0.1.
	
	Input:
		batch.x (Tensor): Input node features of shape.
		batch.edge_index (Tensor): Edge indices of the graph.

	Output:
		ret.x (Tensor): Output node features after applying the APPNP layer.
	"""
	def __init__(self, layer_config: LayerConfig, **kwargs):
		super(APPNP, self).__init__()
		K = 10
		alpha = 0.1
		self.appnp = pygnn.APPNP(K=K, alpha=alpha, dropout=0., add_self_loops=False) # Dropout is handled in GraphGym Wrapper

	def forward(self, batch):
		x = batch.x
		edge_index = batch.edge_index
		x = self.appnp(x, edge_index)
		ret = batch.clone()
		ret.x = x
		return ret