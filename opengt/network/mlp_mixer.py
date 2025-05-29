import torch.nn as nn
from opengt.layer.mlp_mixer import MLPMixer
from opengt.layer.patch_encoder import PatchEncoder
import functorch.einops.rearrange as Rearrange
from torch_scatter import scatter

from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, MLP, GeneralLayer, GCNConv
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from opengt.encoder.feature_encoder import FeatureEncoder

@register_network('GraphMLPMixer')
class GraphMLPMixer(nn.Module):
    """
    GraphMLPMixer model. Only supports graph-level tasks.
    Adapted from https://github.com/XiaoxinHe/Graph-ViT-MLPMixer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.dim_hidden (int): Hidden dimension for GNN layers and GraphMLPMixer layers.
            - cfg.gt.mlpmixer_layers (int): Number of Mixer blocks.
            - cfg.gt.mlpmixer_dropout (float): Dropout rate for the Mixer blocks.
            - cfg.metis.patches (int): Number of patches.
            - cfg.metis.patch_rw_dim (int): Dimension of the patch random walk encoder.

    Input:
        batch (torch_geometric.data.Batch): Input batch containing node features and graph structure.
            - batch.x (torch.Tensor): Input node features.
            - batch.edge_index (torch.Tensor): Edge indices of the graph.
            - batch.subgraphs_nodes_mapper (torch.Tensor): Mapping of nodes to subgraphs.
            - batch.subgraphs_edges_mapper (torch.Tensor): Mapping of edges to subgraphs.
            - batch.combined_subgraphs (torch.Tensor): Combined subgraphs.
            - batch.subgraphs_batch (torch.Tensor): Batch indices for subgraphs.
            - batch.mask (torch.Tensor): Mask for global pooling.
    
    Output:
        pred (torch.Tensor): Graph logits after model processing.
        true (torch.Tensor): Labels for the input batch.
    """

    def __init__(self, dim_in, dim_out):

        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.dropout = cfg.gt.dropout

        self.pooling = cfg.gt.pooling

        self.pre_mp = GeneralLayer('linear', new_layer_config(dim_in = dim_in, dim_out = cfg.gt.dim_hidden, num_layers = 1, has_act = True, has_bias = True, cfg = cfg))

        '''
        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        '''

        self.patch_encoder = PatchEncoder(dim_in=cfg.gt.dim_hidden, dim_out=cfg.gt.dim_hidden)
        # needs scatter & patch rw encoder residual

        # patch PE
        if cfg.metis.patch_rw_dim > 0:
            self.patch_rw_encoder = Sequential('x', [(lambda x: self.copy_patch_pe(x), 'x -> x1'),
                                                     (MLP(new_layer_config(dim_in=cfg.metis.patch_rw_dim, dim_out=cfg.gt.dim_hidden, num_layers=2, has_act=True, has_bias=True, cfg=cfg)), 'x1 -> x1'),
                                                     (lambda x, x1: self.aggregate_batches_add(x, x1), 'x, x1 -> x')])

        self.reshape_layer = Sequential('x', [(lambda x: self.reshape(x), 'x -> x')])

        self.transformer_encoder = MLPMixer(dim_hidden=cfg.gt.dim_hidden, dropout=cfg.gt.mlpmixer_dropout, layers=cfg.gt.mlpmixer_layers, patches=cfg.metis.patches)

        # global pooling
        self.pooling_layer = Sequential('x',[(lambda x: self.global_pooling(x), 'x -> x')])

        GNNHead = register.head_dict['mlp_mixer_graph']
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def copy_patch_pe(self, batch):
        # Copy the patch PE to batch.x
        new_batch = batch.clone()
        new_batch.x = batch.patch_pe
        return new_batch

    def aggregate_batches_add(self, x, x1):
        new_batch = x.clone()
        new_batch.x = x.x + x1.x
        return new_batch

    def reshape(self, batch):
        # Reshape the input tensor to (B, P, D) format
        ret = batch.clone()
        ret.x = Rearrange(batch.x,'(B P) D -> B P D', P=cfg.metis.patches)
        return ret

    def global_pooling(self, batch):
        ret = batch.clone()
        ret.x = (batch.x * batch.mask.unsqueeze(-1)).sum(1) / batch.mask.sum(1, keepdim=True)
        return ret

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
