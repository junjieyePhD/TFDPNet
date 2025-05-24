import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_size, heads, dropout, d_ff)
                for _ in range(num_layers)
            ]
        )
        # self.dropout = nn.Dropout(dropout)
        # self.layer_id = num_layers

    def forward(self,  query, value, mask):
        y_freq_key = 0
        # value = query

        for layer in self.layers:
            query = layer(query, value, value,y_freq_key, mask)      # 传入 V K Q


        return query


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, d_ff):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = GlobalAttentionLayer(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        # self.dropout = nn.Dropout(dropout)


    def forward(self, query, value, key, y_freq_key, mask):

        attention = self.attention(query, value, key, y_freq_key, mask)

        # Add skip connection and run through normalization
        x = self.norm1(attention + query)

        return x

class GlobalAttentionLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(GlobalAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.keys = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.queries = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size, bias=True)

        self.dropout = nn.Dropout(dropout)

        self.softplus = nn.Softplus(-1)

        self.norm1 = nn.LayerNorm(21)


    def forward(self, query, values, keys, y_freq_key, mask=None):

        ######
        # y_freq_key = self.softplus(y_freq_key)

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.dropout(self.values(values))
        keys = self.dropout(self.keys(keys))
        query = self.dropout(self.queries(query))

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # energy = torch.mul(queries, keys)
        if mask is not None:
            mask = mask.unsqueeze(1)#.repeat(1, self.heads, 1, 1)
            energy = energy.masked_fill(mask == 1, float("-1e-8"))

        attention = F.relu(energy / (self.embed_size ** (1 / 2))) # + y_freq_key.unsqueeze(1)
        # attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # attention = self.norm1(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
             N, query_len, self.heads * self.head_dim
        )
        # out = torch.mul(attention, values).reshape(N, query_len, self.heads * self.head_dim)
        out = self.dropout(self.fc_out(out))

        return out
