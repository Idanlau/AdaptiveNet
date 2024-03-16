import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        print("after conv shape: ",seqFt.shape)
        seqFt = torch.mean(seqFt,-1)
        return seqFt
        

class cosine_seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(cosine_seqNet, self).__init__()
        self.inDims = inDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):

        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # print("shape for cosine: ", x.shape) #[24, 5, 4096]
        for idx in range(len(x)):
            cosine_sim_x = x[idx]
            #print("cosine_sim_x shape after slice: ",cosine_sim_x.shape)
            cosine_sim_x = cosine_sim(cosine_sim_x)
            x[idx] = cosine_sim_x
    
        x = x.permute(0,2,1) 
        x = self.conv(x)
        
        x = torch.mean(x,-1)
        return x



# Attention Layer Seqnet
class al_seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(al_seqNet, self).__init__()
        self.inDims = inDims
        self.self_attention_layer = SelfAttentionLayer(4096)
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):

        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # print("x shape for AL: ",x.shape)

        # Create a new tensor for storing output to avoid inplace modification.
        output = torch.zeros_like(x)
        for idx in range(len(x)):
            al_x = x[idx]
            al_x = self.self_attention_layer(al_x)
            output[idx] = al_x

        x = output

        x = x.permute(0,2,1) # shape: [24,4096,10]
        mean_pool = nn.AdaptiveAvgPool1d(1)
        x = mean_pool(x)
        x = x.squeeze(-1)
        return x


# Attention Layer Seqnet
class seqNet_al(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(seqNet_al, self).__init__()
        self.inDims = inDims
        self.self_attention_layer = SelfAttentionLayer(4096)
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):

        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]
        
        # print(x.shape)
        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        x = self.conv(x)

        # print("after conv: ",x.shape)

        x = x.permute(0,2,1)
        # print("permute again: ",x.shape)

        output = torch.zeros_like(x)
        for idx in range(len(x)):
            al_x = x[idx]
            al_x = self.self_attention_layer(al_x)
            output[idx] = al_x

        x = output
        x = x.permute(0,2,1) # shape: [24,4096,x]
        mean_pool = nn.AdaptiveAvgPool1d(1)
        x = mean_pool(x)
        x = x.squeeze(-1)
        return x



class cosine_al_seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(cosine_al_seqNet, self).__init__()
        self.inDims = inDims
        self.self_attention_layer = SelfAttentionLayer(4096)
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # print("shape for cosine: ", x.shape) #[24, 10, 4096]
        for idx in range(len(x)):
            cosine_sim_x = x[idx]
            cosine_sim_x = cosine_sim(cosine_sim_x)
            x[idx] = cosine_sim_x

        # Create a new tensor for storing output to avoid inplace modification.
        output = torch.zeros_like(x)
        for idx in range(len(x)):
            al_x = x[idx]
            al_x = self.self_attention_layer(al_x)
            output[idx] = al_x

        x = output
        
        x = x.permute(0,2,1) #[24, 4096, 10]
        mean_pool = nn.AdaptiveAvgPool1d(1)
        x = mean_pool(x)
        x = x.squeeze(-1)

        return x


class multi_al_seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5, num_heads=8):
        super(multi_al_seqNet, self).__init__()
        self.inDims = inDims
        self.first_self_attention_layer = SelfAttentionLayer(4096)
        self.second_self_attention_layer = SelfAttentionLayer(4096)
        self.w = w
        self.first_conv = nn.Conv1d(inDims, outDims, kernel_size=3)
        self.second_conv = nn.Conv1d(outDims, outDims, kernel_size=3)  # Note: inDims changed to outDims
        self.first_bn = nn.BatchNorm1d(outDims)  # BatchNorm after first conv
        self.second_bn = nn.BatchNorm1d(outDims)  # BatchNorm after second conv
        
    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)  # convert [B,C] to [B,1,C]
        
        # print("before 1st conv: ",x.shape)
        # First Convolution -> BatchNorm -> Activation -> Self-Attention
        x = x.permute(0, 2, 1)  # from [B, T, C] to [B, C, T]
        x = self.first_conv(x)  # Apply first conv layer
        x = self.first_bn(x)    # Apply BatchNorm
        x = F.relu(x)           # Apply ReLU activation

        # print("after 1st conv: ",x.shape)

        x = x.permute(0, 2, 1)  # Permute back to [B, T, C]

        # print("before 1st attention: ",x.shape)

        output = torch.zeros_like(x)
        for idx in range(len(x)):
            al_x = x[idx]
            al_x = self.first_self_attention_layer(al_x)
            output[idx] = al_x
        x = output # Apply first self-attention layer

        # print("after 1st attention: ",x.shape)
        
        # Second Convolution -> BatchNorm -> Activation -> Self-Attention
        x = x.permute(0, 2, 1)  # from [B, T, C] to [B, C, T]
        x = self.second_conv(x)  # Apply second conv layer
        x = self.second_bn(x)    # Apply BatchNorm
        x = F.relu(x)            # Apply ReLU activation

        # print("after 2nd conv: ",x.shape)

        x = x.permute(0, 2, 1)   # Permute back to [B, T, C]

        # print("before 2nd attention: ",x.shape)
        
        output = torch.zeros_like(x)
        for idx in range(len(x)):
            al_x = x[idx]
            al_x = self.second_self_attention_layer(al_x)
            output[idx] = al_x
        x = output

        x = x.permute(0,2,1) # shape: [24,4096,x]

        # print("after 2nd attention and permute: ",x.shape)

        mean_pool = nn.AdaptiveAvgPool1d(1)
        x = mean_pool(x)
        x = x.squeeze(-1)
        return x



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

        self.final_linear = nn.Linear(input_dim, input_dim)

    def forward(self, sequence):
        batch_size = sequence.shape[0]

        # Transform to (batch_size, num_heads, seq_len, dim_per_head)
        query = self.query_linear(sequence).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = self.key_linear(sequence).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.value_linear(sequence).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_sequence = torch.matmul(attention_weights, value)

        # Concatenate heads and put through final linear layer
        attended_sequence = attended_sequence.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)
        return self.final_linear(attended_sequence)


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        
    def forward(self, sequence):
        # print("sequence: ",sequence.shape)
        query = self.query_linear(sequence)
        key = self.key_linear(sequence)
        value = self.value_linear(sequence)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1))

        attention_weights = torch.softmax(attention_scores, dim=-1) #potentially not use softmax maybe we use something like first order norm instead
        
        attended_sequence = torch.matmul(attention_weights, value)

        return attended_sequence


def cosine_sim(sequence):
    # print("sequence type: ", type(sequence))
    num_frames = len(sequence)
    weights = []
    normalized_weights = np.zeros(num_frames)
    
    weighted_seq = []
    central_frame = sequence[(num_frames // 2) + 1].unsqueeze(0) #get central frame

    for j in range(num_frames):
        # print("cosine sim score: ", F.cosine_similarity(sequence[j].unsqueeze(0), central_frame))
        weight = F.cosine_similarity(sequence[j].unsqueeze(0), central_frame)
        weights.append(weight.item())
        
    total_sum = sum(weights)

    for z in range(num_frames):
        normalized_weights[z] = weights[z]/total_sum

    
    for i in range(num_frames):
        weighted_seq.append(sequence[i] * weights[i])

    return torch.stack(weighted_seq)

    
class Delta(nn.Module):
    def __init__(self, inDims, seqL):

        super(Delta, self).__init__()
        self.inDims = inDims
        self.weight = (np.ones(seqL,np.float32))/(seqL/2.0)
        self.weight[:seqL//2] *= -1
        self.weight = nn.Parameter(torch.from_numpy(self.weight),requires_grad=False)

    def forward(self, x):
        # make desc dim as C
        x = x.permute(0,2,1) # makes [B,T,C] as [B,C,T]
        delta = torch.matmul(x,self.weight)

        return delta

