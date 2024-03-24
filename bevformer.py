import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TemporalSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, prev_seq, query):
        N = query.shape[0]
        value_len, key_len, query_len = prev_seq.shape[1], prev_seq.shape[1], query.shape[1]
        # Process inputs
        print("N size: ",N)
        print("Value len: ",value_len)

        """
        TODO: Check this implementation of attention layers figure out einsum and everything else
        """
        
        values = self.values(prev_seq).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(prev_seq).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        # Compute the attention scores
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Softmax to get the attention weights
        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=-1)

        # Apply attention to values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return out


class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SpatialCrossAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.scale = torch.sqrt(torch.FloatTensor([embed_size // heads]))

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = self.values(value).view(N, value_len, self.heads, self.embed_size // self.heads)
        keys = self.keys(key).view(N, key_len, self.heads, self.embed_size // self.heads)
        queries = self.queries(query).view(N, query_len, self.heads, self.embed_size // self.heads)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out


class BevFormer(nn.Module):
    def __init__(self, embed_size, heads, seqL):
        super(BevFormer, self).__init__()
        self.self_temp_attention = TemporalSelfAttention(embed_size, heads)
        self.spatial_cross_attention = SpatialCrossAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size),
        )

    """
    Forwarding the query through neural network architecture
    params: prev_seq (previous sequence(t-1)), query (current (t), to be trained), 
            img_ft (corresponding image features of the query)
    return: out (trained query)
    """
    def forward(self, prev_seq, query, img_ft):
        print("prev_seq shape: ",prev_seq.shape)
        print("query shape: ",query.shape)
        print("img_ft shape: ",img_ft.shape)
        # Self-attention on the query
        attention = self.self_temp_attention(prev_seq, query)
        print("past self_attention")
        # Apply normalization right after self-attention
        x = self.norm1(attention + query)
        
        # Spatial cross-attention using the normalized output of self-attention as query
        cross_attention = self.spatial_cross_attention(img_ft, img_ft, x)
        # Normalize after cross-attention
        x = self.norm2(cross_attention + x)

        # Feed forward network
        forward = self.feed_forward(x)
        # Normalize after feed forward
        out = self.norm3(forward + x)

        return out


class CustomBevformerPooling(nn.Module):
    def __init__(self, embed_size, heads, seqL, outDims):
        super(CustomBevformerPooling, self).__init__()
        self.bevFormer = BevFormer(embed_size=embed_size, heads=heads, seqL=seqL)
        self.flatten = Flatten()  # Assuming Flatten() is a defined module
        self.l2norm = L2Norm()  # Assuming L2Norm() is a defined module

    def forward(self, prev_seq, query, img_ft):
        x = self.bevFormer(prev_seq, query, img_ft)
        x = self.flatten(x)
        x = self.l2norm(x)
        return x


# # Example initialization
# embed_size = 4096  # Example embedding size
# heads = 8  # Example number of heads
# transformer_block = BevFormer(embed_size, heads,seqL=10)
# B=24
# L=10
# # Example tensors for the forward pass
# # Assume batch size of B, sequence length of L, and feature dimension matching embed_size
# prev_seq = torch.rand(B, L, embed_size)  # Previous sequence embeddings
# query = torch.rand(B, L, embed_size)  # Current sequence embeddings (query)
# img_ft = torch.rand(B, L, embed_size)  # Image features corresponding to the current sequence 

# # Forward pass through the transformer block
# output = transformer_block(prev_seq, query, img_ft)
# print(output.shape)
