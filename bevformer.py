import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TemporalSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        # Split the embedding into self.heads different pieces
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how we get the QK^T operation to get our attention matrix
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Optional mask for the attention. It blocks the model from paying attention to future tokens
        # in the sequence which is important for tasks like causal language modeling.
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class SpatialCrossAttention(nn.Module):
    # This would be similar to the TemporalSelfAttention class but
    # would handle the cross attention mechanism between different feature spaces
    # like history BEV and current BEV Queries Q
    pass

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = TemporalSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size),
        )

    def forward(self, value, key, query, mask, curr_seq=0, prev_seq=0): # TODO: should be taking current and previous sequence
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

# Assuming the embed_size and heads are defined
embed_size = 256  # example value
heads = 8  # example value

# Example of creating one block of the model
transformer_block = TransformerBlock(embed_size, heads)

# You would use this block as a part of your model, and the output of one block can be passed as the input to the next.
