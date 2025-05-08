import torch
import torch_geometric.nn as pygnn

import torch.nn as nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg

@register_layer("appnpconv")
class APPNP(nn.Module):
	def __init__(self, layer_config: LayerConfig, **kwargs):
		"""
		Wrapper layer for the APPNP layer from torch_geometric.nn.

		Args:
			dim_in (int): Number of input features.
			dim_out (int): Number of output features.
			K (int): Number of propagation steps. Default is 10.
			alpha (float): Teleport probability. Default is 0.1.
		"""
		super(APPNP, self).__init__()
		K = 10
		alpha = 0.1
		self.appnp = pygnn.APPNP(K=K, alpha=alpha, dropout=0., add_self_loops=False) # Dropout is handled in GraphGym Wrapper

	def forward(self, batch):
		"""
		Forward pass for the APPNPWrapper.

		Args:
			batch (torch_geometric.data.Batch): Input batch containing node features and edge indices.

		Returns:
			Tensor: Output node features of shape [num_nodes, out_channels].
		"""
		x = batch.x
		edge_index = batch.edge_index
		x = self.appnp(x, edge_index)
		ret = batch.clone()
		ret.x = x
		return ret