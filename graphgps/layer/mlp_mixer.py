
import torch.nn as nn
from graphgps.layer.mixer_block import MixerBlock
from torch_geometric.graphgym.register import register_layer

@register_layer("mlp_mixer")
class MLPMixer(nn.Module):
    def __init__(self,
                 layers,
                 dim_hidden,
                 patches,
                 with_final_norm=True,
                 dropout=0):
        super().__init__()
        self.patches = patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(dim_hidden, self.patches, dim_hidden*4, dim_hidden//2, dropout=dropout) for _ in range(layers)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(dim_hidden)

    def forward(self, batch):
        for mixer_block in self.mixer_blocks:
            batch.x = mixer_block(batch.x)
        if self.with_final_norm:
            batch.x = self.layer_norm(batch.x)
        return batch