import inspect
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GATConv, GCNConv
from torch_geometric.utils import to_dense_adj

from torch.nn import Sequential
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

@register_layer('DeGTAConv')
class DeGTAConv(torch.nn.Module):
    def __init__(
            self, dim_in
    ):
        super().__init__()

        enc_name = cfg.dataset.node_encoder_name.split('+')
        if len(enc_name) != 3:
            raise ValueError(f"Expected 3 encoders, got {len(enc_name)}: {enc_name}")
        
        self.pe_channels = getattr(cfg, f"posenc_{enc_name[1]}").dim_pe
        self.se_channels = getattr(cfg, f"posenc_{enc_name[2]}").dim_pe
        self.ae_channels = dim_in - self.pe_channels - self.se_channels
        self.heads = cfg.gt.n_heads
        self.dropout = cfg.gt.dropout

        # multi-view encoder
        self.ae_encoder = Sequential(
            torch.nn.Linear(self.ae_channels, self.ae_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.ae_channels * 2, self.ae_channels),
            torch.nn.Dropout(self.dropout),
        )

        self.pe_encoder = Sequential(
            torch.nn.Linear(self.pe_channels, self.pe_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.pe_channels * 2, self.pe_channels),
        )

        self.se_encoder = Sequential(
            torch.nn.Linear(self.se_channels, self.se_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.se_channels * 2, self.se_channels),
        )

        # local_channel
        self.ae_attn_l = GATConv(self.ae_channels, self.ae_channels, add_self_loops = False)
        self.pe_attn_l = GATConv(self.pe_channels, self.pe_channels, add_self_loops = False)
        self.se_attn_l = GATConv(self.se_channels, self.se_channels, add_self_loops = False)
        self.weightedconv = GCNConv(self.ae_channels, self.ae_channels)

        # global_channel
        self.ae_attn_g = torch.nn.MultiheadAttention(self.ae_channels, self.heads)
        self.pe_attn_g = torch.nn.MultiheadAttention(self.pe_channels, self.heads)
        self.se_attn_g = torch.nn.MultiheadAttention(self.se_channels, self.heads)

        # multi-view adaptive integration
        self.a_l = torch.nn.Parameter(torch.tensor(0.33))
        self.b_l = torch.nn.Parameter(torch.tensor(0.33))
        self.c_l = torch.nn.Parameter(torch.tensor(0.33))
        self.a_g = torch.nn.Parameter(torch.tensor(0.33))
        self.b_g = torch.nn.Parameter(torch.tensor(0.33))
        self.c_g = torch.nn.Parameter(torch.tensor(0.33))

        # local-global adaptive integration
        self.localchannel = torch.nn.Linear(self.ae_channels, self.ae_channels, bias=False)
        self.globalchannel = torch.nn.Linear(self.ae_channels, self.ae_channels, bias=False)

        self.norm1 = torch.nn.LayerNorm(self.ae_channels)
        self.norm2 = torch.nn.LayerNorm(self.ae_channels)
        self.norm3 = torch.nn.LayerNorm(self.ae_channels)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def forward(
            self,
            batch
    ) -> torch.Tensor:

        # multi_view encoder
        ae, pe, se = torch.split(batch.x, [self.ae_channels, self.pe_channels, self.se_channels], dim=1)
        pe_0 = pe.clone()
        se_0 = se.clone()

        ae = self.ae_encoder(ae)
        pe = self.pe_encoder(pe)
        se = self.se_encoder(se)
        edge_index = batch.edge_index
        K = cfg.gt.K
        # print("after encoder", ae, pe, se)

        # local_channel
        _, peattn = self.pe_attn_l(pe, edge_index, return_attention_weights=True)
        _, seattn = self.se_attn_l(se, edge_index, return_attention_weights=True)
        _, aeattn = self.ae_attn_l(ae, edge_index, return_attention_weights=True)
        local_attn = self.a_l * peattn[1] + self.b_l * seattn[1] + self.c_l * aeattn[1]
        out_local = self.weightedconv(ae, edge_index, local_attn)
        out_local = F.dropout(out_local, p=self.dropout, training=self.training)

        # global_channel
        _, peattn = self.pe_attn_g(pe, pe, pe, need_weights=True)
        _, seattn = self.se_attn_g(se, se, se, need_weights=True)
        _, aeattn = self.ae_attn_g(ae, ae, ae, need_weights=True)

        sample_attn = self.a_g * peattn + self.b_g * seattn
        adj = to_dense_adj(edge_index, max_num_nodes=ae.size(0))
        zero_vec = torch.zeros_like(adj)
        sample_attn = torch.where(adj > 0, zero_vec, sample_attn)

        # hard sampling
        values, indices = sample_attn.topk(K, dim=1, largest=True, sorted=True)
        mask = torch.zeros_like(sample_attn).scatter_(1, indices, torch.ones_like(values))
        sample_attn_masked = sample_attn * mask
        aeattn = aeattn * mask

        global_attn = (0.5 * self.a_g + 0.5 * self.b_g) * sample_attn_masked + self.c_g * aeattn
        column_means = torch.sum(global_attn, dim=1)
        global_attn = global_attn / column_means
        #         y_soft = sample_attn
        #         index = y_soft.max(dim=-1, keepdim=True)[1]
        #         y_hard = torch.zeros_like(sample_attn, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        #         ret = y_hard - y_soft.detach() + y_soft
        #         active_edge = ret[:,0]
        out_global = torch.matmul(global_attn, ae)
        out_global = F.dropout(out_global, p=self.dropout, training=self.training)

        # local_global integration
        out = self.localchannel(out_local) + self.globalchannel(out_global)

        if self.norm3 is not None:
            out = self.norm3(out)
        ret = batch.clone()
        ret.x = torch.cat([out.squeeze(0), pe_0, se_0], dim=1)
        return ret