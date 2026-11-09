import torch
import torch.nn as nn
import math


class MAB(nn.Module):
    """Multihead-Attention Block (无需 PyG)"""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        Q_proj = self.fc_q(Q)
        K_proj, V_proj = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_proj.split(dim_split, 2), dim=0)
        K_ = torch.cat(K_proj.split(dim_split, 2), dim=0)
        V_ = torch.cat(V_proj.split(dim_split, 2), dim=0)

        attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)

        if mask is not None:
            # mask 的形状是 [B, N_k]，扩展为 [B*num_heads, N_q, N_k]
            mask_expanded = mask.unsqueeze(1).repeat(self.num_heads, Q_.size(1), 1)
            # 将填充位置的注意力分数设为极小值
            attention_score = attention_score.masked_fill(~mask_expanded, -1e9)

        A = torch.softmax(attention_score, dim=-1)
        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        out = out + torch.relu(self.fc_o(out))
        return out


class SAB(nn.Module):
    """Self-Attention Block"""

    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        self.mab = MAB(in_channels, in_channels, out_channels, num_heads)

    def forward(self, x, mask=None):
        return self.mab(x, x, mask)


class PMA(nn.Module):
    """Pooling by Multihead Attention (用于生成超级节点)"""

    def __init__(self, channels, num_heads, num_seeds):
        super().__init__()
        # S 就是论文中的超级节点查询向量 (Learnable super-node Query vector)
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, channels))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(channels, channels, channels, num_heads)

    def forward(self, x, mask=None):
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, mask)


class HGTPooling(nn.Module):
    """融合论文中 GraphMultisetTransformer 思想的池化层"""

    def __init__(self, channels, num_heads=4, num_supernodes=10):
        super().__init__()
        # 1. 聚合到超级节点
        self.pool1 = PMA(channels, num_heads, num_seeds=num_supernodes)
        # 2. 超级节点间的自注意力交互
        self.self_att = SAB(channels, channels, num_heads)

    def forward(self, x, mask):
        # x 形状: [batch_size, max_num_nodes, channels]
        x = self.pool1(x, mask)  # 输出形状: [batch_size, num_supernodes, channels]
        x = self.self_att(x)  # 输出形状: [batch_size, num_supernodes, channels]
        return x