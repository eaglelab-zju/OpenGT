import torch

import torch.nn as nn

class LayerNorm(nn.Module):
	"""
	Layer normalization module for PyTorch Geometric.
	Applies layer normalization to the input node features.

	Parameters:
		normalized_shape (int or tuple): Shape of the input features to be normalized.
		Can be a single integer or a tuple of integers.
	
	Input:
		batch.x (Tensor): Input node features.
	
	Output:
		batch.x (Tensor): Output node features after applying layer normalization.
	"""
	def __init__(self, normalized_shape):
		super(LayerNorm, self).__init__()
		self.layer_norm = nn.LayerNorm(normalized_shape)

	def forward(self, batch):
		batch.x = self.layer_norm(batch.x)
		return batch
