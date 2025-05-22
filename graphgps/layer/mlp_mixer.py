
import torch.nn as nn
from graphgps.layer.mixer_block import MixerBlock
from torch_geometric.graphgym.register import register_layer

@register_layer("mlp_mixer")
class MLPMixer(nn.Module):
    """
    GraphMLPMixer layer.
    Adapted from https://github.com/XiaoxinHe/Graph-ViT-MLPMixer

    Parameters:
        layers (int): Number of Mixer blocks.
        dim_hidden (int): Number of input features.
        patches (int): Number of patches.
        with_final_norm (bool): Whether to apply final normalization. Default: True.
        dropout (float): Dropout rate. Default: 0.0.
    
    Input:
        batch.x (torch.Tensor): Input node features.
    
    Output:
        batch.x (torch.Tensor): Output node features after applying the Mixer blocks.
    """
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
        ret = batch.clone()
        for mixer_block in self.mixer_blocks:
            ret.x = mixer_block(ret.x)
        if self.with_final_norm:
            ret.x = self.layer_norm(ret.x)
        return ret