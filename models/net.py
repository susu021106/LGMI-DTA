import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from . import gt_net_compound, gt_net_protein
from .hgt_pool import HGTPooling

# if torch.cuda.is_available():
#     device = torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# one layer of FFN
# ==========================================================
class SelfAttention(nn.Module):
    """
    标准多头自注意力模块。
    当前 MDGTDTInet 实际没有使用这个类，而是使用 nn.MultiheadAttention。
    """
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0, "hid_dim 必须能整除 n_heads"

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)

        head_dim = hid_dim // n_heads
        self.register_buffer("scale", torch.tensor(head_dim ** 0.5, dtype=torch.float32))

    def forward(self, query, key, value, mask=None):
        """
        query/key/value: [B, L, D]
        mask: [B, 1, Lq, Lk] 或可广播到对应形状
        """
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x

# ==========================================================
# 轻量 BAN 辅助分支
# 支持 drug_mask / protein_mask
# 可同时用于：
#   1) comp_supernodes × prot_seq
#   2) comp_seq × prot_supernodes
# ==========================================================
class LightBAN(nn.Module):
    """
    轻量版 Interpretable Bilinear Attention

    输入：
        drug_tokens    : [B, Nd, D]
        protein_tokens : [B, Np, D]
        drug_mask      : [B, Nd]，True 表示有效 drug token
        protein_mask   : [B, Np]，True 表示有效 protein token

    输出：
        ban_feat : [B, heads * k]
        att_maps : [B, heads, Nd, Np]
    """
    def __init__(self, dim, k=64, heads=2, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.k = k
        self.heads = heads

        # 低秩投影
        self.U = nn.Linear(dim, k, bias=False)
        self.V = nn.Linear(dim, k, bias=False)

        # 每个 head 一个 q
        self.q = nn.Parameter(torch.empty(heads, k))
        nn.init.xavier_uniform_(self.q)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(heads * k)

    def forward(self, drug_tokens, protein_tokens, drug_mask=None, protein_mask=None):
        """
        drug_tokens    : [B, Nd, D]
        protein_tokens : [B, Np, D]
        drug_mask      : [B, Nd]
        protein_mask   : [B, Np]
        """
        drug_proj = self.act(self.U(drug_tokens))       # [B, Nd, K]
        prot_proj = self.act(self.V(protein_tokens))    # [B, Np, K]

        # 避免 drug padding token 产生无意义交互
        if drug_mask is not None:
            drug_proj = drug_proj * drug_mask[:, :, None].float()

        # att[b,h,i,j] = sum_k q[h,k] * drug_proj[b,i,k] * prot_proj[b,j,k]
        att_maps = torch.einsum(
            'bik,hk,bjk->bhij',
            drug_proj,
            self.q,
            prot_proj
        )  # [B, H, Nd, Np]

        # mask protein 维度
        if protein_mask is not None:
            att_maps = att_maps.masked_fill(
                ~protein_mask[:, None, None, :],
                -1e9
            )

        # 对 protein 维归一化
        att_maps = torch.softmax(att_maps, dim=-1)
        att_maps = self.dropout(att_maps)

        # 无效 drug token 的 attention map 置零，方便可视化
        if drug_mask is not None:
            att_maps = att_maps * drug_mask[:, None, :, None].float()

        # 利用注意力把 protein 聚合到每个 drug token
        prot_context = torch.einsum(
            'bhij,bjk->bhik',
            att_maps,
            prot_proj
        )  # [B, H, Nd, K]

        # 与 drug 投影做逐元素交互
        interact = prot_context * drug_proj[:, None, :, :]  # [B, H, Nd, K]

        # 对 drug token 聚合
        if drug_mask is not None:
            mask = drug_mask[:, None, :, None].float()       # [B, 1, Nd, 1]
            denom = mask.sum(dim=2).clamp_min(1.0)           # [B, 1, 1]
            ban_feat = (interact * mask).sum(dim=2) / denom  # [B, H, K]
        else:
            ban_feat = interact.mean(dim=2)                  # [B, H, K]

        ban_feat = ban_feat.reshape(ban_feat.size(0), -1)    # [B, H*K]
        ban_feat = self.out_norm(ban_feat)

        return ban_feat, att_maps


# ==========================================================
# 主模型：双分支门控 LightBAN
# Branch A: comp_supernodes × prot_seq
# Branch B: comp_seq × prot_supernodes
# ==========================================================
class MDGTDTInet(nn.Module):
    """
    主模型：
    1. 图模态：使用 GINE 提取局部结构特征
    2. 图池化：使用 HGT/GraphMultiset 风格的池化生成超级节点
    3. 序列模态：将预训练 embedding 投影到统一维度
    4. 模态融合：拼接 [CLS] + 图超级节点 + 序列 token
    5. 主干交互：双向 cross-attention
    6. 辅助交互：双分支门控 LightBAN
       Branch A: compound supernodes × protein sequence tokens
       Branch B: compound sequence tokens × protein supernodes
    7. 读出：BAN 特征以残差式弱注入进入 orig_feat 后做回归预测
    """
    def __init__(
        self,
        device,
        compound_dim=256,
        protein_dim=256,
        gt_layers=2,
        gt_heads=8,
        out_dim=1
    ):
        super(MDGTDTInet, self).__init__()

        assert compound_dim == protein_dim, \
            "当前双分支 LightBAN 要求 compound_dim == protein_dim"

        self.device = device
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim

        # --------------------------------------------------
        # 超级节点数量
        # --------------------------------------------------
        self.comp_num_supernodes = 32
        self.prot_num_supernodes = 128

        # --------------------------------------------------
        # 1. 图编码器：局部结构特征提取
        # --------------------------------------------------
        self.compound_gt = gt_net_compound.GINE(
            device=device,
            n_layers=gt_layers,
            node_dim=44,
            edge_dim=10,
            hidden_dim=compound_dim,
            out_dim=compound_dim,
            n_heads=gt_heads,
            in_feat_dropout=0.0,
            dropout=0.3,
            pos_enc_dim=8
        )

        self.protein_gt = gt_net_protein.GINE(
            device=device,
            n_layers=gt_layers,
            node_dim=41,
            edge_dim=5,
            hidden_dim=protein_dim,
            out_dim=protein_dim,
            n_heads=gt_heads,
            in_feat_dropout=0.0,
            dropout=0.3,
            pos_enc_dim=8
        )

        # --------------------------------------------------
        # 2. 图池化：把节点特征池化成固定数目的超级节点
        # --------------------------------------------------
        self.compound_hgt = HGTPooling(
            channels=compound_dim,
            num_heads=4,
            num_supernodes=self.comp_num_supernodes
        )

        self.protein_hgt = HGTPooling(
            channels=protein_dim,
            num_heads=4,
            num_supernodes=self.prot_num_supernodes
        )

        # --------------------------------------------------
        # 3. 预训练 embedding 降维/对齐到图特征空间
        # --------------------------------------------------
        self.protein_embedding_fc = nn.Sequential(
            nn.Linear(320, protein_dim),
            nn.LayerNorm(protein_dim),
            nn.ReLU()
        )

        self.compound_embedding_fc = nn.Sequential(
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(384, compound_dim),
            nn.LayerNorm(compound_dim),
            nn.GELU()
        )

        # --------------------------------------------------
        # 4. [CLS] Token
        # --------------------------------------------------
        self.comp_cls = nn.Parameter(torch.Tensor(1, 1, compound_dim))
        nn.init.xavier_uniform_(self.comp_cls)

        self.prot_cls = nn.Parameter(torch.Tensor(1, 1, protein_dim))
        nn.init.xavier_uniform_(self.prot_cls)

        # Pre-Norm
        self.comp_norm = nn.LayerNorm(compound_dim)
        self.prot_norm = nn.LayerNorm(protein_dim)

        # --------------------------------------------------
        # 5. 主干：双向交叉注意力
        # compound <- protein
        # protein  <- compound
        # --------------------------------------------------
        self.comp_to_prot_attn = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        self.prot_to_comp_attn = nn.MultiheadAttention(
            embed_dim=compound_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        # 序列 token dropout
        self.seq_dropout = nn.Dropout(0.05)

        # 主干 FFN
        self.comp_ffn = nn.Sequential(
            nn.LayerNorm(self.compound_dim),
            nn.Linear(self.compound_dim, self.compound_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.compound_dim * 2, self.compound_dim)
        )

        self.prot_ffn = nn.Sequential(
            nn.LayerNorm(self.protein_dim),
            nn.Linear(self.protein_dim, self.protein_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.protein_dim * 2, self.protein_dim)
        )

        # --------------------------------------------------
        # 6. 双分支 LightBAN
        #
        # 为了与原单分支 k=128, heads=2 的 256维输出公平比较：
        #   Branch A: k=64, heads=2 -> 128维
        #   Branch B: k=64, heads=2 -> 128维
        #   concat 后仍然是 256维
        # --------------------------------------------------
        self.ban_graphdrug_seqprot = LightBAN(
            dim=compound_dim,
            k=64,
            heads=2,
            dropout=0.1
        )

        self.ban_seqdrug_graphprot = LightBAN(
            dim=compound_dim,
            k=64,
            heads=2,
            dropout=0.1
        )

        self.single_ban_feat_dim = 2 * 64       # 128
        self.dual_ban_feat_dim = 2 * self.single_ban_feat_dim  # 256

        # --------------------------------------------------
        # 7. 双分支门控融合
        # 输出两个权重：
        #   gate[:, 0] 控制 comp_supernodes × prot_seq
        #   gate[:, 1] 控制 comp_seq × prot_supernodes
        # --------------------------------------------------
        self.ban_gate = nn.Sequential(
            nn.Linear(self.dual_ban_feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

        # --------------------------------------------------
        # 8. BAN -> orig_feat 的残差式弱注入
        # orig_feat 维度 = 4 * compound_dim
        # --------------------------------------------------
        self.orig_feat_dim = compound_dim * 4

        self.ban_proj = nn.Sequential(
            nn.Linear(self.dual_ban_feat_dim, self.orig_feat_dim),
            nn.LayerNorm(self.orig_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 弱注入系数
        # 如果训练不稳，可以手动改成 0.05
        self.ban_scale = nn.Parameter(
            torch.tensor(0.08, dtype=torch.float32)
        )

        # --------------------------------------------------
        # 9. 最终回归头
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.orig_feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, out_dim)
        )

    def dgl_split_with_mask(self, bg, feats):
        """
        将 batched DGLGraph 的节点特征拆成定长张量，并返回有效 mask。

        输入：
            bg    : batched DGLGraph
            feats : [N_total, C]

        输出：
            out  : [B, max_num_nodes, C]
            mask : [B, max_num_nodes]，True 表示真实节点，False 表示 padding
        """
        batch_num_nodes = bg.batch_num_nodes()
        max_num_nodes = int(batch_num_nodes.max().item())
        B = bg.batch_size
        C = feats.size(-1)

        batch = torch.cat([
            torch.full(
                (int(num.item()),),
                i,
                dtype=torch.long,
                device=bg.device
            )
            for i, num in enumerate(batch_num_nodes)
        ], dim=0)

        cum_nodes = torch.cat([
            torch.zeros(1, dtype=torch.long, device=bg.device),
            batch_num_nodes.cumsum(dim=0)
        ], dim=0)

        idx = torch.arange(
            bg.num_nodes(),
            dtype=torch.long,
            device=bg.device
        )

        idx = (idx - cum_nodes[batch]) + batch * max_num_nodes

        out = feats.new_zeros((B, max_num_nodes, C))
        out.view(-1, C)[idx] = feats

        mask = torch.zeros(
            B,
            max_num_nodes,
            dtype=torch.bool,
            device=bg.device
        )

        for i, num in enumerate(batch_num_nodes):
            mask[i, :int(num.item())] = True

        return out, mask

    def _forward_main_interaction(
        self,
        comp_fused,
        prot_fused,
        comp_fused_mask,
        prot_fused_mask
    ):
        """
        主干交互：
        1. Pre-Norm
        2. 双向 Cross-Attention
        3. FFN + 残差
        4. 取 [CLS]
        5. 构造主干交互特征
        """
        comp_fused_norm = self.comp_norm(comp_fused)
        prot_fused_norm = self.prot_norm(prot_fused)

        comp_guided, _ = self.comp_to_prot_attn(
            query=comp_fused_norm,
            key=prot_fused_norm,
            value=prot_fused_norm,
            key_padding_mask=~prot_fused_mask
        )

        prot_guided, _ = self.prot_to_comp_attn(
            query=prot_fused_norm,
            key=comp_fused_norm,
            value=comp_fused_norm,
            key_padding_mask=~comp_fused_mask
        )

        # 残差
        comp_fused_res = comp_fused + comp_guided
        prot_fused_res = prot_fused + prot_guided

        comp_fused_res = comp_fused_res + self.comp_ffn(comp_fused_res)
        prot_fused_res = prot_fused_res + self.prot_ffn(prot_fused_res)

        # 取 [CLS]
        f_comp = comp_fused_res[:, 0, :]
        f_prot = prot_fused_res[:, 0, :]

        # 原主干四路交互特征
        orig_feat = torch.cat(
            [
                f_comp,
                f_prot,
                f_comp * f_prot,
                torch.abs(f_comp - f_prot)
            ],
            dim=-1
        )

        return orig_feat, comp_fused_res, prot_fused_res

    def forward(
        self,
        compound_graph,
        protein_graph,
        protein_embedding,
        compound_embedding,
        label=None,
        use_opm=False,
        return_attention=False,
        return_feat=False
    ):
        """
        参数：
            compound_graph     : DGL batched graph
            protein_graph      : DGL batched graph
            protein_embedding  : [B, Lp, 320]
            compound_embedding : [B, Lc, 384]
            label              : 当前版本未使用
            use_opm            : 当前版本未使用
            return_attention   : 是否返回两个 BAN 分支的注意力图和 gate

        返回：
            默认：
                output: [B, 1]

            若 return_attention=True：
                output, att_info
        """
        # --------------------------------------------------
        # 1. 图模态：GINE 提取节点级特征
        # --------------------------------------------------
        compound_feat = self.compound_gt(compound_graph)   # [N_comp_total, D]
        protein_feat = self.protein_gt(protein_graph)      # [N_prot_total, D]

        compound_feat_x, comp_mask = self.dgl_split_with_mask(
            compound_graph,
            compound_feat
        )

        protein_feat_x, prot_mask = self.dgl_split_with_mask(
            protein_graph,
            protein_feat
        )

        # --------------------------------------------------
        # 2. 图池化：生成固定数目的超级节点
        # --------------------------------------------------
        comp_supernodes = self.compound_hgt(
            compound_feat_x,
            comp_mask
        )  # [B, 32, D]

        prot_supernodes = self.protein_hgt(
            protein_feat_x,
            prot_mask
        )  # [B, 128, D]

        # --------------------------------------------------
        # 3. 序列模态：embedding 对齐到统一空间
        # --------------------------------------------------
        comp_seq = self.compound_embedding_fc(
            compound_embedding
        )  # [B, Lc, D]

        prot_seq = self.protein_embedding_fc(
            protein_embedding
        )  # [B, Lp, D]

        comp_seq = self.seq_dropout(comp_seq)
        prot_seq = self.seq_dropout(prot_seq)

        # --------------------------------------------------
        # 4. 根据原始 embedding 判断有效 token
        # --------------------------------------------------
        comp_valid = (
            compound_embedding.abs().sum(dim=-1) > 1e-6
        )  # [B, Lc]

        prot_valid = (
            protein_embedding.abs().sum(dim=-1) > 1e-6
        )  # [B, Lp]

        B = comp_supernodes.size(0)

        # [CLS] token 扩展到 batch 维度
        comp_cls_token = self.comp_cls.repeat(B, 1, 1)  # [B, 1, D]
        prot_cls_token = self.prot_cls.repeat(B, 1, 1)  # [B, 1, D]

        # --------------------------------------------------
        # 5. 主干拼接融合序列
        # [CLS], [图超级节点], [序列 token]
        # --------------------------------------------------
        comp_fused = torch.cat(
            [
                comp_cls_token,
                comp_supernodes,
                comp_seq
            ],
            dim=1
        )

        prot_fused = torch.cat(
            [
                prot_cls_token,
                prot_supernodes,
                prot_seq
            ],
            dim=1
        )

        comp_cls_mask = torch.ones(
            B,
            1,
            dtype=torch.bool,
            device=comp_fused.device
        )

        prot_cls_mask = torch.ones(
            B,
            1,
            dtype=torch.bool,
            device=prot_fused.device
        )

        comp_super_mask = torch.ones(
            B,
            self.comp_num_supernodes,
            dtype=torch.bool,
            device=comp_fused.device
        )

        prot_super_mask = torch.ones(
            B,
            self.prot_num_supernodes,
            dtype=torch.bool,
            device=prot_fused.device
        )

        comp_fused_mask = torch.cat(
            [
                comp_cls_mask,
                comp_super_mask,
                comp_valid
            ],
            dim=1
        )

        prot_fused_mask = torch.cat(
            [
                prot_cls_mask,
                prot_super_mask,
                prot_valid
            ],
            dim=1
        )

        # --------------------------------------------------
        # 6. 主干：双向 cross-attention
        # --------------------------------------------------
        orig_feat, comp_fused_res, prot_fused_res = self._forward_main_interaction(
            comp_fused,
            prot_fused,
            comp_fused_mask,
            prot_fused_mask
        )

        # --------------------------------------------------
        # 7. 双分支 LightBAN
        #
        # Branch A:
        #     comp_supernodes × prot_seq
        #     药物结构超节点 × 蛋白序列 token
        #
        # Branch B:
        #     comp_seq × prot_supernodes
        #     药物 ChemBERTa token × 蛋白结构超节点
        # --------------------------------------------------

        # Branch A: GraphDrug-SeqProt
        ban_feat_graphdrug_seqprot, att_graphdrug_seqprot = self.ban_graphdrug_seqprot(
            drug_tokens=comp_supernodes,      # [B, 32, D]
            protein_tokens=prot_seq,          # [B, Lp, D]
            drug_mask=None,                   # comp_supernodes 固定 32 个，全部有效
            protein_mask=prot_valid           # [B, Lp]
        )

        # Branch B: SeqDrug-GraphProt
        ban_feat_seqdrug_graphprot, att_seqdrug_graphprot = self.ban_seqdrug_graphprot(
            drug_tokens=comp_seq,             # [B, Lc, D]
            protein_tokens=prot_supernodes,   # [B, 128, D]
            drug_mask=comp_valid,             # [B, Lc]
            protein_mask=None                 # prot_supernodes 固定 128 个，全部有效
        )

        # --------------------------------------------------
        # 8. 门控融合两个 BAN 分支
        # --------------------------------------------------
        ban_cat = torch.cat(
            [
                ban_feat_graphdrug_seqprot,
                ban_feat_seqdrug_graphprot
            ],
            dim=-1
        )  # [B, 256]

        gate_logits = self.ban_gate(ban_cat)          # [B, 2]
        ban_gate = torch.softmax(gate_logits, dim=-1) # [B, 2]

        ban_feat_graphdrug_seqprot = (
            ban_feat_graphdrug_seqprot * ban_gate[:, 0:1]
        )

        ban_feat_seqdrug_graphprot = (
            ban_feat_seqdrug_graphprot * ban_gate[:, 1:2]
        )

        dual_ban_feat = torch.cat(
            [
                ban_feat_graphdrug_seqprot,
                ban_feat_seqdrug_graphprot
            ],
            dim=-1
        )  # [B, 256]

        # --------------------------------------------------
        # 9. BAN 特征残差式弱注入到 orig_feat
        # final_feat = orig_feat + alpha * ban_delta
        # --------------------------------------------------
        ban_delta = self.ban_proj(dual_ban_feat)  # [B, 4D]

        # final_feat = orig_feat + self.ban_scale * ban_delta
        #
        # final_feat = F.dropout(
        #     final_feat,
        #     p=0.1,
        #     training=self.training
        # )
        #
        # output = self.classifier(final_feat)
        #
        # if return_attention:
        #     att_info = {
        #         "att_graphdrug_seqprot": att_graphdrug_seqprot,
        #         "att_seqdrug_graphprot": att_seqdrug_graphprot,
        #         "ban_gate": ban_gate
        #     }
        #     return output, att_info
        #
        # return output

        final_feat = orig_feat + self.ban_scale * ban_delta

        # 用于 t-SNE / UMAP 的特征，取 dropout 前
        feat_for_vis = final_feat

        final_feat = F.dropout(
            final_feat,
            p=0.1,
            training=self.training
        )

        output = self.classifier(final_feat)

        if return_attention:
            att_info = {
                "att_graphdrug_seqprot": att_graphdrug_seqprot,
                "att_seqdrug_graphprot": att_seqdrug_graphprot,
                "ban_gate": ban_gate
            }

            if return_feat:
                return output, feat_for_vis, att_info

            return output, att_info

        if return_feat:
            return output, feat_for_vis

        return output

'''
# comp_supernodes × prot_seq
# ==========================================================
# 轻量 BAN 辅助分支
# 这里不替代主干 cross-attention，只补充一个局部交互特征
# ==========================================================
class LightBAN(nn.Module):
    """
    轻量版 Interpretable Bilinear Attention

    输入：
        drug_tokens    : [B, Nd, D]
        protein_tokens : [B, Np, D]
        protein_mask   : [B, Np]，True 表示有效 token

    输出：
        ban_feat : [B, heads * k]
        att_maps : [B, heads, Nd, Np]
    """
    def __init__(self, dim, k=64, heads=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.k = k
        self.heads = heads

        # 共享低秩投影
        self.U = nn.Linear(dim, k, bias=False)
        self.V = nn.Linear(dim, k, bias=False)

        # 每个 head 一个 q
        self.q = nn.Parameter(torch.empty(heads, k))
        nn.init.xavier_uniform_(self.q)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(heads * k)

    def forward(self, drug_tokens, protein_tokens, protein_mask=None):
        """
        drug_tokens    : [B, Nd, D]
        protein_tokens : [B, Np, D]
        protein_mask   : [B, Np]
        """
        drug_proj = self.act(self.U(drug_tokens))      # [B, Nd, K]
        prot_proj = self.act(self.V(protein_tokens))   # [B, Np, K]

        # att[b,h,i,j] = sum_k q[h,k] * drug_proj[b,i,k] * prot_proj[b,j,k]
        att_maps = torch.einsum('bik,hk,bjk->bhij', drug_proj, self.q, prot_proj)  # [B,H,Nd,Np]

        if protein_mask is not None:
            att_maps = att_maps.masked_fill(~protein_mask[:, None, None, :], -1e9)

        # 对 protein 维归一化
        att_maps = torch.softmax(att_maps, dim=-1)
        att_maps = self.dropout(att_maps)

        # 利用注意力把 protein 聚合到每个 drug token
        prot_context = torch.einsum('bhij,bjk->bhik', att_maps, prot_proj)  # [B,H,Nd,K]

        # 与 drug 投影做逐元素交互，再对 drug token 聚合
        ban_feat = (prot_context * drug_proj[:, None, :, :]).mean(dim=2)   # [B,H,K]

        ban_feat = ban_feat.reshape(ban_feat.size(0), -1)  # [B, H*K]
        ban_feat = self.out_norm(ban_feat)

        return ban_feat, att_maps


# ==========================================================
# 主模型：原版主干 + comp_supernodes × prot_seq 的轻量 BAN 辅助分支
# BAN 特征通过“残差式弱注入”进入 orig_feat
# ==========================================================
class MDGTDTInet(nn.Module):
    """
    主模型：
    1. 图模态：使用 GINE 提取局部结构特征
    2. 图池化：使用 HGT/GraphMultiset 风格的池化生成超级节点
    3. 序列模态：将预训练 embedding 投影到统一维度
    4. 模态融合：拼接 [CLS] + 图超级节点 + 序列 token
    5. 主干交互：双向 cross-attention
    6. 辅助交互：轻量 BAN 分支（comp_supernodes × prot_seq）
    7. 读出：取 [CLS] 主干特征，并把 BAN 特征以残差式弱注入到主干特征后做最终回归预测
    """
    def __init__(self, device, compound_dim=256, protein_dim=256, gt_layers=2, gt_heads=8, out_dim=1):
        super(MDGTDTInet, self).__init__()

        self.device = device
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim

        # --------------------------------------------------
        # 超级节点数量
        # --------------------------------------------------
        self.comp_num_supernodes = 32
        self.prot_num_supernodes = 128

        # --------------------------------------------------
        # 1. 图编码器：局部结构特征提取
        # --------------------------------------------------
        self.compound_gt = gt_net_compound.GINE(
            device=device,
            n_layers=gt_layers,
            node_dim=44,
            edge_dim=10,
            hidden_dim=compound_dim,
            out_dim=compound_dim,
            n_heads=gt_heads,
            in_feat_dropout=0.0,
            dropout=0.3,
            pos_enc_dim=8
        )

        self.protein_gt = gt_net_protein.GINE(
            device=device,
            n_layers=gt_layers,
            node_dim=41,
            edge_dim=5,
            hidden_dim=protein_dim,
            out_dim=protein_dim,
            n_heads=gt_heads,
            in_feat_dropout=0.0,
            dropout=0.3,
            pos_enc_dim=8
        )

        # --------------------------------------------------
        # 2. 图池化：把节点特征池化成固定数目的超级节点
        # --------------------------------------------------
        self.compound_hgt = HGTPooling(
            channels=compound_dim,
            num_heads=4,
            num_supernodes=self.comp_num_supernodes
        )
        self.protein_hgt = HGTPooling(
            channels=protein_dim,
            num_heads=4,
            num_supernodes=self.prot_num_supernodes
        )

        # --------------------------------------------------
        # 3. 预训练 embedding 降维/对齐到图特征空间
        # --------------------------------------------------
        self.protein_embedding_fc = nn.Sequential(
            nn.Linear(320, protein_dim),
            nn.LayerNorm(protein_dim),
            nn.ReLU()
        )

        self.compound_embedding_fc = nn.Sequential(
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(384, compound_dim),
            nn.LayerNorm(compound_dim),
            nn.GELU()
        )

        # --------------------------------------------------
        # 4. [CLS] Token：用于主干全局聚合融合后的信息
        # --------------------------------------------------
        self.comp_cls = nn.Parameter(torch.Tensor(1, 1, compound_dim))
        nn.init.xavier_uniform_(self.comp_cls)

        self.prot_cls = nn.Parameter(torch.Tensor(1, 1, protein_dim))
        nn.init.xavier_uniform_(self.prot_cls)

        # Pre-Norm
        self.comp_norm = nn.LayerNorm(compound_dim)
        self.prot_norm = nn.LayerNorm(protein_dim)

        # --------------------------------------------------
        # 5. 主干：双向交叉注意力
        # compound <- protein
        # protein  <- compound
        # --------------------------------------------------
        self.comp_to_prot_attn = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        self.prot_to_comp_attn = nn.MultiheadAttention(
            embed_dim=compound_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        # 序列 token dropout
        self.seq_dropout = nn.Dropout(0.05)

        # 主干 FFN
        self.comp_ffn = nn.Sequential(
            nn.LayerNorm(self.compound_dim),
            nn.Linear(self.compound_dim, self.compound_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.compound_dim * 2, self.compound_dim)
        )

        self.prot_ffn = nn.Sequential(
            nn.LayerNorm(self.protein_dim),
            nn.Linear(self.protein_dim, self.protein_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.protein_dim * 2, self.protein_dim)
        )

        # --------------------------------------------------
        # 6. 轻量 BAN 辅助分支
        # 只做 comp_supernodes × prot_seq
        # --------------------------------------------------
        self.ban = LightBAN(
            dim=compound_dim,   # 默认 compound_dim == protein_dim == 256
            k=128,
            heads=2,
            dropout=0.1
        )
        self.ban_feat_dim = 2 * 128  # heads * k = 256

        # --------------------------------------------------
        # 7. BAN -> orig_feat 的残差式弱注入
        # orig_feat 维度 = 4 * compound_dim
        # --------------------------------------------------
        self.orig_feat_dim = compound_dim * 4

        self.ban_proj = nn.Sequential(
            nn.Linear(self.ban_feat_dim, self.orig_feat_dim),
            nn.LayerNorm(self.orig_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 初始化为较小值，让 BAN 先作为“弱辅助”
        self.ban_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))

        # --------------------------------------------------
        # 8. 最终回归头
        # 现在 classifier 只吃增强后的 orig_feat，不再额外拼接 ban_feat
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.orig_feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, out_dim)
        )

    def dgl_split_with_mask(self, bg, feats):
        """
        将 batched DGLGraph 的节点特征拆成定长张量，并返回有效 mask。

        输入：
            bg    : batched DGLGraph
            feats : [N_total, C]

        输出：
            out  : [B, max_num_nodes, C]
            mask : [B, max_num_nodes]，True 表示真实节点，False 表示 padding
        """
        batch_num_nodes = bg.batch_num_nodes()
        max_num_nodes = int(batch_num_nodes.max().item())
        B = bg.batch_size
        C = feats.size(-1)

        batch = torch.cat([
            torch.full((int(num.item()),), i, dtype=torch.long, device=bg.device)
            for i, num in enumerate(batch_num_nodes)
        ], dim=0)

        cum_nodes = torch.cat([
            torch.zeros(1, dtype=torch.long, device=bg.device),
            batch_num_nodes.cumsum(dim=0)
        ], dim=0)

        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + batch * max_num_nodes

        out = feats.new_zeros((B, max_num_nodes, C))
        out.view(-1, C)[idx] = feats

        mask = torch.zeros(B, max_num_nodes, dtype=torch.bool, device=bg.device)
        for i, num in enumerate(batch_num_nodes):
            mask[i, :int(num.item())] = True

        return out, mask

    def _forward_main_interaction(self, comp_fused, prot_fused, comp_fused_mask, prot_fused_mask):
        """
        主干交互：
        1. Pre-Norm
        2. 双向 Cross-Attention
        3. FFN + 残差
        4. 取 [CLS]
        5. 构造主干交互特征
        """
        comp_fused_norm = self.comp_norm(comp_fused)
        prot_fused_norm = self.prot_norm(prot_fused)

        comp_guided, _ = self.comp_to_prot_attn(
            query=comp_fused_norm,
            key=prot_fused_norm,
            value=prot_fused_norm,
            key_padding_mask=~prot_fused_mask
        )

        prot_guided, _ = self.prot_to_comp_attn(
            query=prot_fused_norm,
            key=comp_fused_norm,
            value=comp_fused_norm,
            key_padding_mask=~comp_fused_mask
        )

        # 残差
        comp_fused_res = comp_fused + comp_guided
        prot_fused_res = prot_fused + prot_guided

        comp_fused_res = comp_fused_res + self.comp_ffn(comp_fused_res)
        prot_fused_res = prot_fused_res + self.prot_ffn(prot_fused_res)

        # 取 [CLS]
        f_comp = comp_fused_res[:, 0, :]
        f_prot = prot_fused_res[:, 0, :]

        # 原主干四路交互特征
        orig_feat = torch.cat(
            [f_comp, f_prot, f_comp * f_prot, torch.abs(f_comp - f_prot)],
            dim=-1
        )

        return orig_feat, comp_fused_res, prot_fused_res

    def forward(self, compound_graph, protein_graph, protein_embedding, compound_embedding,
                label=None, use_opm=False, return_attention=False):
        """
        参数：
            compound_graph     : DGL batched graph
            protein_graph      : DGL batched graph
            protein_embedding  : [B, Lp, 320]
            compound_embedding : [B, Lc, 384]
            label              : [B]，当前辅助 BAN 版本未使用
            use_opm            : 当前辅助 BAN 版本未使用
            return_attention   : 是否返回辅助 BAN 的注意力图

        返回：
            默认：
                output: [B, 1]
            若 return_attention=True：
                output, ban_att_maps
        """
        # --------------------------------------------------
        # 1. 图模态：GINE 提取节点级特征
        # --------------------------------------------------
        compound_feat = self.compound_gt(compound_graph)   # [N_comp_total, D]
        protein_feat = self.protein_gt(protein_graph)      # [N_prot_total, D]

        compound_feat_x, comp_mask = self.dgl_split_with_mask(compound_graph, compound_feat)
        protein_feat_x, prot_mask = self.dgl_split_with_mask(protein_graph, protein_feat)

        # --------------------------------------------------
        # 2. 图池化：生成固定数目的超级节点
        # --------------------------------------------------
        comp_supernodes = self.compound_hgt(compound_feat_x, comp_mask)   # [B, 32, D]
        prot_supernodes = self.protein_hgt(protein_feat_x, prot_mask)     # [B, 128, D]

        # --------------------------------------------------
        # 3. 序列模态：embedding 对齐到统一空间
        # --------------------------------------------------
        comp_seq = self.compound_embedding_fc(compound_embedding)   # [B, Lc, D]
        prot_seq = self.protein_embedding_fc(protein_embedding)     # [B, Lp, D]

        comp_seq = self.seq_dropout(comp_seq)
        prot_seq = self.seq_dropout(prot_seq)

        # --------------------------------------------------
        # 4. 根据原始 embedding 判断有效 token
        # --------------------------------------------------
        comp_valid = (compound_embedding.abs().sum(dim=-1) > 1e-6)   # [B, Lc]
        prot_valid = (protein_embedding.abs().sum(dim=-1) > 1e-6)    # [B, Lp]

        B = comp_supernodes.size(0)

        # [CLS] token 扩展到 batch 维度
        comp_cls_token = self.comp_cls.repeat(B, 1, 1)   # [B, 1, D]
        prot_cls_token = self.prot_cls.repeat(B, 1, 1)   # [B, 1, D]

        # --------------------------------------------------
        # 5. 主干拼接融合序列
        # [CLS], [图超级节点], [序列 token]
        # --------------------------------------------------
        comp_fused = torch.cat([comp_cls_token, comp_supernodes, comp_seq], dim=1)
        prot_fused = torch.cat([prot_cls_token, prot_supernodes, prot_seq], dim=1)

        comp_cls_mask = torch.ones(B, 1, dtype=torch.bool, device=comp_fused.device)
        prot_cls_mask = torch.ones(B, 1, dtype=torch.bool, device=prot_fused.device)

        comp_super_mask = torch.ones(
            B, self.comp_num_supernodes, dtype=torch.bool, device=comp_fused.device
        )
        prot_super_mask = torch.ones(
            B, self.prot_num_supernodes, dtype=torch.bool, device=prot_fused.device
        )

        comp_fused_mask = torch.cat([comp_cls_mask, comp_super_mask, comp_valid], dim=1)
        prot_fused_mask = torch.cat([prot_cls_mask, prot_super_mask, prot_valid], dim=1)

        # --------------------------------------------------
        # 6. 主干：双向 cross-attention
        # --------------------------------------------------
        orig_feat, comp_fused_res, prot_fused_res = self._forward_main_interaction(
            comp_fused, prot_fused, comp_fused_mask, prot_fused_mask
        )

        # --------------------------------------------------
        # 7. 轻量 BAN 辅助分支
        # 用更纯净的局部交互：
        # drug side    = comp_supernodes
        # protein side = prot_seq
        # --------------------------------------------------
        ban_feat, ban_att_maps = self.ban(
            drug_tokens=comp_supernodes,   # [B, 32, D]
            protein_tokens=prot_seq,       # [B, Lp, D]
            protein_mask=prot_valid
        )

        # --------------------------------------------------
        # 8. BAN 特征残差式弱注入到 orig_feat
        # final_feat = orig_feat + alpha * ban_delta
        # --------------------------------------------------
        ban_delta = self.ban_proj(ban_feat)                     # [B, 4D]
        final_feat = orig_feat + self.ban_scale * ban_delta
        final_feat = F.dropout(final_feat, p=0.1, training=self.training)

        output = self.classifier(final_feat)

        if return_attention:
            return output, ban_att_maps
        return output
'''

'''
# ==========================================================
# 轻量 BAN 辅助分支
# 支持 drug_mask / protein_mask
# 当前实验版本用于：comp_seq × prot_supernodes
# ==========================================================
class LightBAN(nn.Module):
    """
    轻量版 Interpretable Bilinear Attention

    输入：
        drug_tokens    : [B, Nd, D]
        protein_tokens : [B, Np, D]
        drug_mask      : [B, Nd]，True 表示有效 drug token
        protein_mask   : [B, Np]，True 表示有效 protein token

    输出：
        ban_feat : [B, heads * k]
        att_maps : [B, heads, Nd, Np]
    """
    def __init__(self, dim, k=64, heads=2, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.k = k
        self.heads = heads

        # 共享低秩投影
        self.U = nn.Linear(dim, k, bias=False)
        self.V = nn.Linear(dim, k, bias=False)

        # 每个 head 一个 q
        self.q = nn.Parameter(torch.empty(heads, k))
        nn.init.xavier_uniform_(self.q)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(heads * k)

    def forward(self, drug_tokens, protein_tokens, drug_mask=None, protein_mask=None):
        """
        drug_tokens    : [B, Nd, D]
        protein_tokens : [B, Np, D]
        drug_mask      : [B, Nd]
        protein_mask   : [B, Np]
        """
        drug_proj = self.act(self.U(drug_tokens))       # [B, Nd, K]
        prot_proj = self.act(self.V(protein_tokens))    # [B, Np, K]

        # 避免 drug padding token 产生无意义交互
        if drug_mask is not None:
            drug_proj = drug_proj * drug_mask[:, :, None].float()

        # att[b,h,i,j] = sum_k q[h,k] * drug_proj[b,i,k] * prot_proj[b,j,k]
        att_maps = torch.einsum(
            'bik,hk,bjk->bhij',
            drug_proj,
            self.q,
            prot_proj
        )  # [B, H, Nd, Np]

        # mask protein 维度
        if protein_mask is not None:
            att_maps = att_maps.masked_fill(
                ~protein_mask[:, None, None, :],
                -1e9
            )

        # 对 protein 维归一化
        att_maps = torch.softmax(att_maps, dim=-1)
        att_maps = self.dropout(att_maps)

        # 让无效 drug token 的 attention map 也置零，方便后续可视化
        if drug_mask is not None:
            att_maps = att_maps * drug_mask[:, None, :, None].float()

        # 利用注意力把 protein 聚合到每个 drug token
        prot_context = torch.einsum(
            'bhij,bjk->bhik',
            att_maps,
            prot_proj
        )  # [B, H, Nd, K]

        # 与 drug 投影做逐元素交互
        interact = prot_context * drug_proj[:, None, :, :]  # [B, H, Nd, K]

        # 对 drug token 聚合
        if drug_mask is not None:
            mask = drug_mask[:, None, :, None].float()       # [B, 1, Nd, 1]
            denom = mask.sum(dim=2).clamp_min(1.0)           # [B, 1, 1]
            ban_feat = (interact * mask).sum(dim=2) / denom  # [B, H, K]
        else:
            ban_feat = interact.mean(dim=2)                  # [B, H, K]

        ban_feat = ban_feat.reshape(ban_feat.size(0), -1)    # [B, H*K]
        ban_feat = self.out_norm(ban_feat)

        return ban_feat, att_maps


# ==========================================================
# 主模型：原版主干 + comp_seq × prot_supernodes 的 LightBAN 消融版本
# BAN 特征通过“残差式弱注入”进入 orig_feat
# ==========================================================
class MDGTDTInet(nn.Module):
    """
    主模型：
    1. 图模态：使用 GINE 提取局部结构特征
    2. 图池化：使用 HGT/GraphMultiset 风格的池化生成超级节点
    3. 序列模态：将预训练 embedding 投影到统一维度
    4. 模态融合：拼接 [CLS] + 图超级节点 + 序列 token
    5. 主干交互：双向 cross-attention
    6. 辅助交互：轻量 BAN 分支，本版本使用 comp_seq × prot_supernodes
    7. 读出：取 [CLS] 主干特征，并把 BAN 特征以残差式弱注入到主干特征后做最终回归预测
    """
    def __init__(self, device, compound_dim=256, protein_dim=256, gt_layers=2, gt_heads=8, out_dim=1):
        super(MDGTDTInet, self).__init__()

        assert compound_dim == protein_dim, "当前 LightBAN 消融版本要求 compound_dim == protein_dim"

        self.device = device
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim

        # --------------------------------------------------
        # 超级节点数量
        # --------------------------------------------------
        self.comp_num_supernodes = 32
        self.prot_num_supernodes = 128

        # --------------------------------------------------
        # 1. 图编码器：局部结构特征提取
        # --------------------------------------------------
        self.compound_gt = gt_net_compound.GINE(
            device=device,
            n_layers=gt_layers,
            node_dim=44,
            edge_dim=10,
            hidden_dim=compound_dim,
            out_dim=compound_dim,
            n_heads=gt_heads,
            in_feat_dropout=0.0,
            dropout=0.3,
            pos_enc_dim=8
        )

        self.protein_gt = gt_net_protein.GINE(
            device=device,
            n_layers=gt_layers,
            node_dim=41,
            edge_dim=5,
            hidden_dim=protein_dim,
            out_dim=protein_dim,
            n_heads=gt_heads,
            in_feat_dropout=0.0,
            dropout=0.3,
            pos_enc_dim=8
        )

        # --------------------------------------------------
        # 2. 图池化：把节点特征池化成固定数目的超级节点
        # --------------------------------------------------
        self.compound_hgt = HGTPooling(
            channels=compound_dim,
            num_heads=4,
            num_supernodes=self.comp_num_supernodes
        )

        self.protein_hgt = HGTPooling(
            channels=protein_dim,
            num_heads=4,
            num_supernodes=self.prot_num_supernodes
        )

        # --------------------------------------------------
        # 3. 预训练 embedding 降维/对齐到图特征空间
        # --------------------------------------------------
        self.protein_embedding_fc = nn.Sequential(
            nn.Linear(320, protein_dim),
            nn.LayerNorm(protein_dim),
            nn.ReLU()
        )

        self.compound_embedding_fc = nn.Sequential(
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(384, compound_dim),
            nn.LayerNorm(compound_dim),
            nn.GELU()
        )

        # --------------------------------------------------
        # 4. [CLS] Token：用于主干全局聚合融合后的信息
        # --------------------------------------------------
        self.comp_cls = nn.Parameter(torch.Tensor(1, 1, compound_dim))
        nn.init.xavier_uniform_(self.comp_cls)

        self.prot_cls = nn.Parameter(torch.Tensor(1, 1, protein_dim))
        nn.init.xavier_uniform_(self.prot_cls)

        # Pre-Norm
        self.comp_norm = nn.LayerNorm(compound_dim)
        self.prot_norm = nn.LayerNorm(protein_dim)

        # --------------------------------------------------
        # 5. 主干：双向交叉注意力
        # compound <- protein
        # protein  <- compound
        # --------------------------------------------------
        self.comp_to_prot_attn = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        self.prot_to_comp_attn = nn.MultiheadAttention(
            embed_dim=compound_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        # 序列 token dropout
        self.seq_dropout = nn.Dropout(0.05)

        # 主干 FFN
        self.comp_ffn = nn.Sequential(
            nn.LayerNorm(self.compound_dim),
            nn.Linear(self.compound_dim, self.compound_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.compound_dim * 2, self.compound_dim)
        )

        self.prot_ffn = nn.Sequential(
            nn.LayerNorm(self.protein_dim),
            nn.Linear(self.protein_dim, self.protein_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.protein_dim * 2, self.protein_dim)
        )

        # --------------------------------------------------
        # 6. 轻量 BAN 辅助分支
        # 本消融版本：comp_seq × prot_supernodes
        # drug side    = ChemBERTa token sequence
        # protein side = protein graph supernodes
        # --------------------------------------------------
        self.ban = LightBAN(
            dim=compound_dim,
            k=128,
            heads=2,
            dropout=0.1
        )

        self.ban_feat_dim = 2 * 128  # heads * k = 256

        # --------------------------------------------------
        # 7. BAN -> orig_feat 的残差式弱注入
        # orig_feat 维度 = 4 * compound_dim
        # --------------------------------------------------
        self.orig_feat_dim = compound_dim * 4

        self.ban_proj = nn.Sequential(
            nn.Linear(self.ban_feat_dim, self.orig_feat_dim),
            nn.LayerNorm(self.orig_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 初始化为较小值，让 BAN 先作为“弱辅助”
        self.ban_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))

        # --------------------------------------------------
        # 8. 最终回归头
        # classifier 只吃增强后的 orig_feat
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.orig_feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, out_dim)
        )

    def dgl_split_with_mask(self, bg, feats):
        """
        将 batched DGLGraph 的节点特征拆成定长张量，并返回有效 mask。

        输入：
            bg    : batched DGLGraph
            feats : [N_total, C]

        输出：
            out  : [B, max_num_nodes, C]
            mask : [B, max_num_nodes]，True 表示真实节点，False 表示 padding
        """
        batch_num_nodes = bg.batch_num_nodes()
        max_num_nodes = int(batch_num_nodes.max().item())
        B = bg.batch_size
        C = feats.size(-1)

        batch = torch.cat([
            torch.full((int(num.item()),), i, dtype=torch.long, device=bg.device)
            for i, num in enumerate(batch_num_nodes)
        ], dim=0)

        cum_nodes = torch.cat([
            torch.zeros(1, dtype=torch.long, device=bg.device),
            batch_num_nodes.cumsum(dim=0)
        ], dim=0)

        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + batch * max_num_nodes

        out = feats.new_zeros((B, max_num_nodes, C))
        out.view(-1, C)[idx] = feats

        mask = torch.zeros(B, max_num_nodes, dtype=torch.bool, device=bg.device)
        for i, num in enumerate(batch_num_nodes):
            mask[i, :int(num.item())] = True

        return out, mask

    def _forward_main_interaction(self, comp_fused, prot_fused, comp_fused_mask, prot_fused_mask):
        """
        主干交互：
        1. Pre-Norm
        2. 双向 Cross-Attention
        3. FFN + 残差
        4. 取 [CLS]
        5. 构造主干交互特征
        """
        comp_fused_norm = self.comp_norm(comp_fused)
        prot_fused_norm = self.prot_norm(prot_fused)

        comp_guided, _ = self.comp_to_prot_attn(
            query=comp_fused_norm,
            key=prot_fused_norm,
            value=prot_fused_norm,
            key_padding_mask=~prot_fused_mask
        )

        prot_guided, _ = self.prot_to_comp_attn(
            query=prot_fused_norm,
            key=comp_fused_norm,
            value=comp_fused_norm,
            key_padding_mask=~comp_fused_mask
        )

        # 残差
        comp_fused_res = comp_fused + comp_guided
        prot_fused_res = prot_fused + prot_guided

        comp_fused_res = comp_fused_res + self.comp_ffn(comp_fused_res)
        prot_fused_res = prot_fused_res + self.prot_ffn(prot_fused_res)

        # 取 [CLS]
        f_comp = comp_fused_res[:, 0, :]
        f_prot = prot_fused_res[:, 0, :]

        # 原主干四路交互特征
        orig_feat = torch.cat(
            [
                f_comp,
                f_prot,
                f_comp * f_prot,
                torch.abs(f_comp - f_prot)
            ],
            dim=-1
        )

        return orig_feat, comp_fused_res, prot_fused_res

    def forward(
        self,
        compound_graph,
        protein_graph,
        protein_embedding,
        compound_embedding,
        label=None,
        use_opm=False,
        return_attention=False
    ):
        """
        参数：
            compound_graph     : DGL batched graph
            protein_graph      : DGL batched graph
            protein_embedding  : [B, Lp, 320]
            compound_embedding : [B, Lc, 384]
            label              : 当前版本未使用
            use_opm            : 当前版本未使用
            return_attention   : 是否返回辅助 BAN 的注意力图

        返回：
            默认：
                output: [B, 1]
            若 return_attention=True：
                output, ban_att_maps
        """
        # --------------------------------------------------
        # 1. 图模态：GINE 提取节点级特征
        # --------------------------------------------------
        compound_feat = self.compound_gt(compound_graph)   # [N_comp_total, D]
        protein_feat = self.protein_gt(protein_graph)      # [N_prot_total, D]

        compound_feat_x, comp_mask = self.dgl_split_with_mask(
            compound_graph,
            compound_feat
        )

        protein_feat_x, prot_mask = self.dgl_split_with_mask(
            protein_graph,
            protein_feat
        )

        # --------------------------------------------------
        # 2. 图池化：生成固定数目的超级节点
        # --------------------------------------------------
        comp_supernodes = self.compound_hgt(
            compound_feat_x,
            comp_mask
        )  # [B, 32, D]

        prot_supernodes = self.protein_hgt(
            protein_feat_x,
            prot_mask
        )  # [B, 128, D]

        # --------------------------------------------------
        # 3. 序列模态：embedding 对齐到统一空间
        # --------------------------------------------------
        comp_seq = self.compound_embedding_fc(compound_embedding)  # [B, Lc, D]
        prot_seq = self.protein_embedding_fc(protein_embedding)    # [B, Lp, D]

        comp_seq = self.seq_dropout(comp_seq)
        prot_seq = self.seq_dropout(prot_seq)

        # --------------------------------------------------
        # 4. 根据原始 embedding 判断有效 token
        # --------------------------------------------------
        comp_valid = (compound_embedding.abs().sum(dim=-1) > 1e-6)  # [B, Lc]
        prot_valid = (protein_embedding.abs().sum(dim=-1) > 1e-6)   # [B, Lp]

        B = comp_supernodes.size(0)

        # [CLS] token 扩展到 batch 维度
        comp_cls_token = self.comp_cls.repeat(B, 1, 1)  # [B, 1, D]
        prot_cls_token = self.prot_cls.repeat(B, 1, 1)  # [B, 1, D]

        # --------------------------------------------------
        # 5. 主干拼接融合序列
        # [CLS], [图超级节点], [序列 token]
        # --------------------------------------------------
        comp_fused = torch.cat(
            [
                comp_cls_token,
                comp_supernodes,
                comp_seq
            ],
            dim=1
        )

        prot_fused = torch.cat(
            [
                prot_cls_token,
                prot_supernodes,
                prot_seq
            ],
            dim=1
        )

        comp_cls_mask = torch.ones(
            B, 1,
            dtype=torch.bool,
            device=comp_fused.device
        )

        prot_cls_mask = torch.ones(
            B, 1,
            dtype=torch.bool,
            device=prot_fused.device
        )

        comp_super_mask = torch.ones(
            B,
            self.comp_num_supernodes,
            dtype=torch.bool,
            device=comp_fused.device
        )

        prot_super_mask = torch.ones(
            B,
            self.prot_num_supernodes,
            dtype=torch.bool,
            device=prot_fused.device
        )

        comp_fused_mask = torch.cat(
            [
                comp_cls_mask,
                comp_super_mask,
                comp_valid
            ],
            dim=1
        )

        prot_fused_mask = torch.cat(
            [
                prot_cls_mask,
                prot_super_mask,
                prot_valid
            ],
            dim=1
        )

        # --------------------------------------------------
        # 6. 主干：双向 cross-attention
        # --------------------------------------------------
        orig_feat, comp_fused_res, prot_fused_res = self._forward_main_interaction(
            comp_fused,
            prot_fused,
            comp_fused_mask,
            prot_fused_mask
        )

        # --------------------------------------------------
        # 7. LightBAN 消融版本
        # 原版本：
        #     comp_supernodes × prot_seq
        #
        # 当前版本：
        #     comp_seq × prot_supernodes
        #
        # 注意：
        #     comp_seq 是 ChemBERTa token 序列，需要 comp_valid mask
        #     prot_supernodes 是固定 128 个超级节点，默认全部有效
        # --------------------------------------------------
        ban_feat, ban_att_maps = self.ban(
            drug_tokens=comp_seq,              # [B, Lc, D]
            protein_tokens=prot_supernodes,    # [B, 128, D]
            drug_mask=comp_valid,              # [B, Lc]
            protein_mask=None                  # prot_supernodes 全部有效
        )

        # --------------------------------------------------
        # 8. BAN 特征残差式弱注入到 orig_feat
        # final_feat = orig_feat + alpha * ban_delta
        # --------------------------------------------------
        ban_delta = self.ban_proj(ban_feat)  # [B, 4D]

        final_feat = orig_feat + self.ban_scale * ban_delta
        final_feat = F.dropout(
            final_feat,
            p=0.1,
            training=self.training
        )

        output = self.classifier(final_feat)

        if return_attention:
            return output, ban_att_maps

        return output
'''