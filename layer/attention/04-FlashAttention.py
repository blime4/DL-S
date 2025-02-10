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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use optimized FlashAttention for SDPA
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return output

batch_size=1
seq_len=4
feature_dim=3

query = torch.randn(batch_size, seq_len, feature_dim)
key = torch.randn(batch_size, seq_len, feature_dim)
value = torch.randn(batch_size, seq_len, feature_dim)

mask = torch.tensor([[[1, 1, 1, 0]]], dtype=torch.bool) # (batch_size, 1, seq_len)


attention_layer = Attention()
output = attention_layer(query, key, value, mask)
print(output.shape)

################################################################
# torch.nn.functional.scaled_dot_product_attention 的 C++实现是 FlashAttention
# 使用 FlashAttention
# 目的：减少显存占用，提高计算效率。
# 方法：用 torch.nn.functional.scaled_dot_product_attention 取代手写的 softmax 和 matmul。
################################################################
