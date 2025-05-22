import torch
import torch.nn as nn
from graphgps.layer.difformer_layer import DIFFormerConv


from torch_geometric.nn import Sequential
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config, GeneralLayer, GCNConv, Linear
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.feature_encoder import FeatureEncoder

@register_network("Graphtransformer")
class Graphtransformer(nn.Module):
    '''
    Graphtransformer model. Adapted from https://github.com/graphdeeplearning/graphtransformer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        cfg (dict): Configuration dictionary containing model parameters from GraphGym.
            - cfg.gt.layers (int): Number of Graphtransformer layers.
            - cfg.gt.dim_hidden (int): Hidden dimension for GNN layers and Graphtransformer layers.
            - cfg.gt.layer_type (str): Type of layer to use for the Graphtransformer layers.
            - cfg.gnn.head (str): Type of head to use for the final output layer.
        
    Input:
        batch (torch_geometric.data.Batch): input batch containing node features and graph structure.
            - batch.x (torch.Tensor): input node features.
            - batch.edge_index (torch.Tensor): edge indices of the graph.
    
    Output:
        batch (task dependent type, see output head): Output after model processing.
    '''
    def __init__(self, dim_in, dim_out):
        super(Graphtransformer, self).__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        convlist = []

        for i in range(cfg.gt.layers):
            convlist.append((GeneralLayer(cfg.gt.layer_type.split('+')[0].lower()+'conv', new_layer_config(dim_in = cfg.gt.dim_hidden, dim_out = cfg.gt.dim_hidden, has_bias = True, has_act = False, num_layers = cfg.gnn.layers, cfg = cfg)), 'x -> x'))
        
        self.convs = Sequential('x', convlist)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)
    
    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
