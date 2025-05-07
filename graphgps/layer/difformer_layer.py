import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_layer
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

def full_attention_conv(qs, ks, vs, kernel, output_attn=False):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    '''
    if kernel == 'simple':
        # normalize input
        qs = qs / torch.norm(qs, p=2) # [N, H, M]
        ks = ks / torch.norm(ks, p=2) # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer # [N, L, H]

    elif kernel == 'sigmoid':
        # numerator
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))  # [N, L, H]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[0], 1)  # [N, L, H]

        # compute attention and attentive aggregated results
        attention = attention_num / attention_normalizer
        attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  # [N, H, D]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

def gcn_conv(x, edge_index, edge_weight = None):
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    gcn_conv_output = []
    # if edge_weight is None:
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    # else:
    # value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for i in range(x.shape[1]):
        gcn_conv_output.append( matmul(adj, x[:, i]) )  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

@register_layer('DIFFormerConv')
class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, dim_in, dim_out, config):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wq = nn.Linear(dim_in, dim_out * config.n_heads)
        if config.use_weight:
            self.Wv = nn.Linear(dim_in, dim_out * config.n_heads)

        self.norm = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(config.dropout)
        
        self.dim_out = dim_out
        self.n_heads = config.n_heads
        self.kernel = config.kernel
        self.use_graph = config.use_graph
        self.use_weight = config.use_weight
        self.graph_weight = config.graph_weight
        self.residual = config.use_residual
        self.use_source = config.use_source
        self.alpha = config.alpha
        self.use_bn = config.batch_norm
        

    def forward(self, batch, batch_orig): # batch_orig conects to the original node features

        x = batch.x
        edge_index = batch.edge_index
        # feature transformation
        query = self.Wq(x).reshape(-1, self.n_heads, self.dim_out)
        key = self.Wk(x).reshape(-1, self.n_heads, self.dim_out)
        if self.use_weight:
            value = self.Wv(x).reshape(-1, self.n_heads, self.dim_out)
        else:
            value = x.reshape(-1, 1, self.dim_out)

        # compute full attentive aggregation
        attention_output = full_attention_conv(query, key, value, self.kernel) # [N, H, D]

        # use input graph for gcn conv
        if self.use_graph:
            if self.graph_weight > 0:
                final_output = (1 - self.graph_weight) * attention_output \
                               + self.graph_weight * gcn_conv(value, edge_index)
            else:
                final_output = attention_output + gcn_conv(value, edge_index)
        else:
            final_output = attention_output
        final_output = final_output.mean(dim=1)

        if self.use_source:
            final_output += batch_orig.x
        if self.residual:
            final_output = final_output * self.alpha + batch.x * (1 - self.alpha)
        if self.use_bn:
            final_output = self.norm(final_output)
        final_output = self.dropout(final_output)

        ret = batch.clone()
        ret.x = final_output

        return ret

