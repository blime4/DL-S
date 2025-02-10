import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):    #  Scaled Dot-Product Attention
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,  # (batch_size, seq_len, feature_dim)
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        # compute raw attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))

        # apply softmax get attn_weights
        attn_weights = F.softmax(scores / math.sqrt(query.size(-1)), dim=-1)

        # compute the final output
        output = torch.matmul(attn_weights, value)

        return output

batch_size=1
seq_len=4
feature_dim=3

query = torch.randn(batch_size, seq_len, feature_dim)
key = torch.randn(batch_size, seq_len, feature_dim)
value = torch.randn(batch_size, seq_len, feature_dim)

attention_layer = Attention()
output = attention_layer(query, key, value)
print(output.shape)
