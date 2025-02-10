import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):    #  Dot-Product Attention
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
        attn_weights = F.softmax(scores, dim=-1)

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

###

# Q: 为什么要用torch.matmul？

# A: torch.matmul 是PyTorch中的一个函数，用于执行两个张量（tensor）之间的矩阵乘法。
#    在这个场景下，我们使用它来计算query和key之间的相似度得分。
#    具体来说，通过计算query矩阵和key矩阵转置后的乘积，我们可以得到一个得分矩阵（scores），
#    这个矩阵的每个元素代表了一个query向量和一个key向量之间的匹配程度。

# Q: key.transpose(-2, -1) 是什么？

# A: key.transpose(-2, -1) 操作是对key矩阵进行转置操作。
#    这里的-2和-1指的是维度的索引，从后往前数。
#    对于一个二维矩阵而言，-2是行，-1是列。
#    因此，key.transpose(-2, -1)就是将key矩阵的行和列交换，使得原来的列变成了行，行变成了列。
#    这样做的目的是为了让query和key的维度对齐，以便于后续的矩阵乘法运算能够正确执行。

# Q: F.softmax(scores, dim=-1) 是什么？
# A:dim=-1: 这个参数指定了沿着哪一个维度应用Softmax函数。
#   在PyTorch中，维度是从0开始计数的，-1表示沿着最后一个维度操作。
# 例如，如果你有一个形状为 (batch_size, seq_len, d_k) 的张量，那么dim=-1意味着Softmax将在d_k这个维度上对值进行归一化。

###