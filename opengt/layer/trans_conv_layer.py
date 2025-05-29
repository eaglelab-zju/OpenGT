import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_layer

def full_attention_conv(qs, ks, vs, kernel='simple', output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    return attn_output


@register_layer('TransConvLayer')
class TransConvLayer(nn.Module):
    '''
    Transformer with fast attention. Used in SGFormer.
    Adapted from https://github.com/qitianwu/SGFormer

    Parameters:
        dim_in (int): Number of input features.
        dim_out (int): Number of output features.
        config (object): Configuration object containing hyperparameters.
            - n_heads (int): Number of attention heads.
            - use_weight (bool): Whether to use weight for value.
            - use_residual (bool): Whether to use residual connection.
            - use_act (bool): Whether to use activation function.
            - layer_norm (bool): Whether to use layer normalization.
            - batch_norm (bool): Whether to use batch normalization.
            - dropout (float): Dropout rate.
    
    Input:
        batch.x (torch.Tensor): Input node features.
        batch.edge_index (torch.Tensor): Edge indices of the graph.
    
    Output:
        ret.x (torch.Tensor): Output node features after applying the TransConv layer.
    '''

    def __init__(self, dim_in,
                 dim_out,
                 config):
        super().__init__()
        self.Wk = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wq = nn.Linear(dim_in, dim_out * config.n_heads)
        if config.use_weight:
            self.Wv = nn.Linear(dim_in, dim_out * config.n_heads)

        self.dim_out = dim_out
        self.num_heads = config.n_heads
        self.use_weight = config.use_weight
        self.residual = config.use_residual
        self.use_act = config.use_act

        if config.layer_norm:
            self.norm = nn.LayerNorm(dim_out)
        elif config.batch_norm:
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            self.norm = None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, batch):

        input=batch.x

        # feature transformation
        query = self.Wq(input).reshape(-1,
                                             self.num_heads, self.dim_out)
        key = self.Wk(input).reshape(-1,
                                            self.num_heads, self.dim_out)
        if self.use_weight:
            value = self.Wv(input).reshape(-1,
                                                  self.num_heads, self.dim_out)
        else:
            value = input.reshape(-1, 1, self.dim_out)

        # compute full attentive aggregation
        
        attention_output = full_attention_conv(
            query, key, value)  # [N, H, D]

        x = attention_output
        x = x.mean(dim=1)
        
        if self.residual:
            x = x + input
        if self.norm is not None:
            x = self.norm(x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x)

        ret = batch.clone()
        ret.x = x

        return ret
