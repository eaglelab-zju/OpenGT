import torch

import torch.nn as nn

class LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(LayerNorm, self).__init__()
		self.layer_norm = nn.LayerNorm(normalized_shape)

	def forward(self, batch):
		batch.x = self.layer_norm(batch.x)
		return batch
