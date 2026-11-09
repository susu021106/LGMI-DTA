import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINEConv


class GINE(nn.Module):
    def __init__(self, device, n_layers, node_dim, edge_dim, hidden_dim, out_dim, n_heads=None, in_feat_dropout=0.0,
                 dropout=0.2, pos_enc_dim=8):
        super(GINE, self).__init__()
        self.device = device

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.pos_proj = nn.Linear(pos_enc_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim if i < n_layers - 1 else out_dim)
            )
            self.layers.append(GINEConv(apply_func=mlp, learn_eps=True))

    def forward(self, g):
        # 以当前层参数所在设备为准，最稳
        device = self.node_proj.weight.device

        # 把图搬到和模型一致的设备
        g = g.to(device)

        # 提取原始特征，并显式搬到同一设备
        h = g.ndata['feats'].float().to(device)
        h_pos = g.ndata['lap_pos_enc'].float().to(device)
        e = g.edata['feats'].float().to(device)

        # 训练时对位置编码做随机符号翻转
        if self.training:
            sign_flip = torch.rand(h_pos.size(1), device=device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            h_pos = h_pos * sign_flip.unsqueeze(0)

        # 初始特征投影
        h = self.node_proj(h) + self.pos_proj(h_pos)
        e = self.edge_proj(e)
        h = self.dropout(h)

        # 多层 GINE + 残差
        for conv in self.layers:
            h_res = h
            h = conv(g, h, e)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + h_res

        return h