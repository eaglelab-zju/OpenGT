# Vendored from HubGT (MIT) — upstream: https://github.com/gdmnl/HubGT/blob/main/model.py
# OpenGT: only backbone blocks used by ``opengt.network.hubgt``; no downstream classifier
# (logits come from GraphGym ``post_mp``). Removed unused StructuralLinear / KernelEncoderLayer / GT.

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

INF8 = 255


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class StructuralEmbedding(nn.Module):
    def __init__(self, num_heads, num_global_node):
        super().__init__()
        self.num_global_node = num_global_node
        self.linear_bias = nn.Embedding(INF8 + 1, num_heads, padding_idx=INF8)
        if self.num_global_node > 0:
            self.virtual_bias = nn.Embedding(self.num_global_node, num_heads)

    def forward(self, attn_bias):
        attn_bias = attn_bias.squeeze(3)
        n_graph, n_node = attn_bias.size()[:2]

        mask_off = (attn_bias == INF8)
        attn_bias = self.linear_bias(attn_bias.int())
        attn_bias[mask_off] = -torch.inf

        if self.num_global_node > 0:
            vnode_attn_bias = self.virtual_bias.weight.unsqueeze(0)
            attn_bias = torch.cat(
                [
                    attn_bias,
                    vnode_attn_bias.unsqueeze(2).repeat(n_graph, 1, n_node, 1),
                ],
                dim=1,
            )
            attn_bias = torch.cat(
                [
                    attn_bias,
                    vnode_attn_bias.unsqueeze(0).repeat(
                        n_graph, n_node + self.num_global_node, 1, 1
                    ),
                ],
                dim=2,
            )

        return attn_bias.permute(0, 3, 1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        attention_dropout_rate,
        num_heads,
        attn_bias_dim,
        num_global_node,
        dp_bias,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dp_bias = dp_bias
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.bias_enc = StructuralEmbedding(num_heads, num_global_node)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, get_score=False):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            attn_bias = self.bias_enc(attn_bias)
            if self.dp_bias > 0:
                mask = torch.rand_like(attn_bias) < self.dp_bias
                mask &= ~torch.eye(
                    attn_bias.size(2), dtype=torch.bool, device=mask.device
                ).unsqueeze(0).unsqueeze(0)
                mask = torch.where(mask, -torch.inf, 0.0)
                attn_bias = attn_bias + mask
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        if get_score:
            score = x[:, :, 0, :] * torch.norm(v, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        if get_score:
            return x, score.mean(dim=1)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_size,
        dropout_rate,
        attention_dropout_rate,
        num_heads,
        attn_bias_dim,
        num_global_node,
        dp_bias,
    ):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size,
            attention_dropout_rate,
            num_heads,
            attn_bias_dim,
            num_global_node,
            dp_bias,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, get_score=False):
        y = self.self_attention_norm(x)
        if get_score:
            _, score = self.self_attention(y, y, y, attn_bias, get_score=True)
            return score
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
