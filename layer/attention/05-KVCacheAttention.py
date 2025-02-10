import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kv_cache = None  # 存储 (key, value) 的元组作为KV缓存

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,  # 启用KV缓存的标志
    ) -> torch.Tensor:
        if use_cache:
            if self.kv_cache is not None:
                # 拼接缓存的key和value与当前输入的key和value
                cached_key, cached_value = self.kv_cache
                key = torch.cat([cached_key, key], dim=-2)                 # (batch_size, seq_len, feature_dim)
                value = torch.cat([cached_value, value], dim=-2)           # dim=-2 意味着在 seq_len 上面进行 torch.cat
            # 更新缓存为当前的key和value（拼接后的结果）
            self.kv_cache = (key.detach(), value.detach())
        else:
            # 不使用缓存时重置缓存
            self.kv_cache = None

        # 使用优化的FlashAttention计算注意力
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )
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
# 引入 KV Cache
# 目的：缓存 key 和 value，减少重复计算，提高长序列推理速度。
# 方法：维护 self.kv_cache，对增量 tokens 直接读取缓存。
################################################################