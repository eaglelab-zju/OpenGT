import torch.nn as nn
from graphgps.layer.mlp_mixer import MLPMixer
import functorch.einops.rearrange as Rearrange
from torch_scatter import scatter

from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, MLP, GeneralLayer, GCNConv
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.feature_encoder import FeatureEncoder

@register_network('GraphMLPMixer')
class GraphMLPMixer(nn.Module):

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

        convs=[]
        for i in range(cfg.gt.layers):
            if(i > 0):
                convs.append((lambda x: self.pre_layer_scatter(x), 'x -> x1'))
                convs.append((MLP(new_layer_config(dim_in=cfg.gt.dim_hidden, dim_out=cfg.gt.dim_hidden, num_layers=1, has_act=True, has_bias=True, cfg=cfg)), 'x1 -> x1'))
                convs.append((lambda x, x1: self.aggregate_batches_add(x, x1), 'x, x1 -> x'))
                convs.append((lambda x: self.post_layer_scatter(x), 'x -> x'))
            convs.append((GCNConv(new_layer_config(dim_in=cfg.gt.dim_hidden, dim_out=cfg.gt.dim_hidden, num_layers=1, has_act=True, has_bias=True, cfg=cfg)), 'x -> x'))
            

        convs.append((lambda x: self.final_scatter(x), 'x -> x'))

        self.convs = Sequential('x', convs)
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

    def pre_layer_scatter(self, batch):
        x = batch.x[batch.subgraphs_nodes_mapper]
        batch_x = batch.subgraphs_batch
        batch.x = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
        return batch

    def post_layer_scatter(self, batch):
        x = batch.x[batch.subgraphs_nodes_mapper]
        batch.x = scatter(x, batch.subgraphs_nodes_mapper, dim=0, reduce='mean')[batch.subgraphs_nodes_mapper]
        return batch
    
    def final_scatter(self, batch):
        x = batch.x[batch.subgraphs_nodes_mapper]
        batch_x = batch.subgraphs_batch
        batch.x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        return batch

    def reshape(self, batch):
        # Reshape the input tensor to (B, P, D) format
        batch.x = Rearrange(batch.x,'(B P) D -> B P D', P=cfg.metis.patches)
        return batch

    def global_pooling(self, batch):
        batch.x = (batch.x * batch.mask.unsqueeze(-1)).sum(1) / batch.mask.sum(1, keepdim=True)
        return batch

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
