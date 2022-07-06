import math

import torch
from torch import nn

from .parameters import *


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_seq_len, trg_seq_len, device):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, dim_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, dim_model)
        self.src_positional_encoder = PositionalEncoder(src_seq_len, device)
        self.trg_positional_encoder = PositionalEncoder(trg_seq_len, device)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(dim_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input) # (B, L) => (B, L, dim_model)
        trg_input = self.trg_embedding(trg_input) # (B, L) => (B, L, dim_model)
        src_input = self.src_positional_encoder(src_input) # (B, L, dim_model) => (B, L, dim_model)
        trg_input = self.trg_positional_encoder(trg_input) # (B, L, dim_model) => (B, L, dim_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, dim_model)
        d_output, attn_weight = self.decoder(trg_input, e_output, e_mask, d_mask) # (B, L, dim_model)

        output = self.softmax(self.output_linear(d_output)) # (B, L, dim_model) => # (B, L, trg_vocab_size)

        return output, attn_weight


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x, attn_weight = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x), attn_weight


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(dropout_rate)

        self.layer_norm_2 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_2 = nn.Dropout(dropout_rate)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x) # (B, L, dim_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)[0]
        ) # (B, L, dim_model)
        x_2 = self.layer_norm_2(x) # (B, L, dim_model)
        x = x + self.drop_out_2(self.feed_forward(x_2)) # (B, L, dim_model)

        return x # (B, L, dim_model)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.masked_multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(dropout_rate)

        self.layer_norm_2 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_2 = nn.Dropout(dropout_rate)

        self.layer_norm_3 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_3 = nn.Dropout(dropout_rate)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.layer_norm_1(x) # (B, L, dim_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)[0]
        ) # (B, L, dim_model)
        x_2 = self.layer_norm_2(x) # (B, L, dim_model)
        attn_output, attn_weight = self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        x = x + self.drop_out_2(
            attn_output #self.multihead_attention(x_2, e_output, e_output, mask=e_mask)[0]
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x, attn_weight # (B, L, d_model)



class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(dim_model, dim_model)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(dim_model, dim_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, num_heads, dim_k) # (B, L, num_heads, dim_k)
        k = self.w_k(k).view(input_shape[0], -1, num_heads, dim_k) # (B, L, num_heads, dim_k)
        v = self.w_v(v).view(input_shape[0], -1, num_heads, dim_k) # (B, L, num_heads, dim_k)

        # For convenience, convert all tensors in size (B, num_heads, L, dim_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values, attn_weights = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, dim_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, dim_model) # (B, L, dim_model)

        return self.w_0(concat_output), attn_weights

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(dim_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)
        attn_weights = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values, attn_weights


class FeedFowardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(dim_ff, dim_model, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) # (B, L, dim_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, dim_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([dim_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, device):
        super().__init__()
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, dim_model) # (L, dim_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(dim_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / dim_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / dim_model)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, dim_model)
        #self.positional_encoding = pe_matrix.requires_grad_(False)
        self.positional_encoding = pe_matrix.to(device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(dim_model) # (B, L, dim_model)
        x = x + self.positional_encoding # (B, L, dim_model)

        return x

