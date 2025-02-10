import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # Add mask parameter
    ) -> torch.Tensor:
        # Compute raw attention scores by performing matrix multiplication
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # Apply mask (for autoregressive decoding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Compute the final output as a weighted sum of values
        output = torch.matmul(attn_weights, value)
        return output

batch_size=1
seq_len=4
feature_dim=3

query = torch.randn(batch_size, seq_len, feature_dim)
key = torch.randn(batch_size, seq_len, feature_dim)
value = torch.randn(batch_size, seq_len, feature_dim)

mask = torch.tensor([[[1, 1, 1, 0]]])  # (batch_size, 1, seq_len)


attention_layer = Attention()
output = attention_layer(query, key, value, mask)
print(output.shape)


################################################################
# Q: 解释 masked_fill
# A:
# mask = torch.tensor([[[1, 1, 1, 0]]])
# mask ==0 --> tensor([[[False,  True, False,  True]]])
# 将 scores 中对应位置的值替换为 float('-inf')
################################################################
