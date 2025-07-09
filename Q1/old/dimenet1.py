import torch
import torch.nn as nn
import numpy as np
from math import pi as PI


# --- Helper Functions and Classes from the Paper ---

class Swish(nn.Module):
    """
    Swish激活函数: f(x) = x * sigmoid(x)
    [cite_start]论文中提到使用它来替代ReLU，以保证模型的二次连续可微性 [cite: 279]
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class Envelope(nn.Module):
    """
    平滑截断函数 (Envelope function)
    [cite_start]用于确保径向基函数在截断距离c处平滑地降为0，保证能量和力的连续性 [cite: 188, 244]
    """

    def __init__(self, p):
        super(Envelope, self).__init__()
        self.p = p + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class BesselBasis(nn.Module):
    """
    径向基函数 (Radial Basis Functions - RBF)
    [cite_start]使用球贝塞尔函数构建，相比高斯基函数更具物理意义且参数效率更高 [cite: 157, 173]
    """

    def __init__(self, num_radial, cutoff, envelope_exponent):
        super(BesselBasis, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.arange(1, num_radial + 1) * PI)

    def forward(self, d):
        d_scaled = d / self.cutoff
        # d_scaled.unsqueeze(-1) adds a dimension for broadcasting
        # This allows computing sin for all frequencies at once
        bessel_term = torch.sin(self.frequencies * d_scaled.unsqueeze(-1)) / d.unsqueeze(-1)

        # Apply the envelope function for smooth cutoff
        envelope_val = self.envelope(d_scaled)
        # We need to reshape envelope_val to match the bessel_term for broadcasting
        return envelope_val.unsqueeze(-1) * bessel_term


class SphericalBasis(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(SphericalBasis, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        bessel_forms = torch.arange(num_spherical) * (num_spherical - 1) * PI
        bessel_zeros = torch.arange(1, num_radial + 1) * PI
        self.bessel_weights = nn.Parameter(bessel_zeros.unsqueeze(0) + bessel_forms.unsqueeze(1), requires_grad=False)
        self.spherical_harmonics_weights = nn.Parameter(torch.arange(num_spherical), requires_grad=False)

    def forward(self, d, angle, idx_kj):
        d_scaled = d / self.cutoff

        # Bessel (distance) part
        bessel_term = torch.sin(self.bessel_weights * d_scaled[idx_kj].unsqueeze(-1).unsqueeze(-1))

        # --- vvvvvvvvvvvvvvvvvv  THIS LINE IS CORRECTED  vvvvvvvvvvvvvvvvvv ---
        # Spherical harmonics (angle) part
        # We add .unsqueeze(-1) at the end to make its shape compatible for multiplication
        spherical_harmonics_term = torch.cos(self.spherical_harmonics_weights * angle.unsqueeze(-1)).unsqueeze(-1)
        # --- ^^^^^^^^^^^^^^^^^^  END OF CORRECTION  ^^^^^^^^^^^^^^^^^^ ---

        # Combine them
        spherical_bessel_term = bessel_term * spherical_harmonics_term

        # Apply envelope
        envelope_val = self.envelope(d_scaled[idx_kj])
        return envelope_val.unsqueeze(-1).unsqueeze(-1) * spherical_bessel_term

# --- Main Architectural Blocks ---

class ResidualBlock(nn.Module):
    """
    [cite_start]残差模块，用于DimeNet的交互模块中，加深网络并稳定训练 [cite: 273]
    """

    def __init__(self, num_feat):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(num_feat, num_feat)
        self.lin2 = nn.Linear(num_feat, num_feat)
        self.act = Swish()

    def forward(self, x):
        return x + self.lin2(self.act(self.lin1(x)))


class EmbeddingBlock(nn.Module):
    """
    [cite_start]嵌入模块：将原子序数（这里是固定的金原子）和初始距离信息转换为消息嵌入 m_ji [cite: 260]
    """

    def __init__(self, num_feat, num_radial):
        super(EmbeddingBlock, self).__init__()
        self.atom_embedding = nn.Embedding(1, num_feat)  # 1 for Gold (Au)
        self.rbf_proj = nn.Linear(num_radial, num_feat)
        self.act = Swish()

    def forward(self, z, rbf, i, j):
        # Since all atoms are Gold, z will be all zeros.
        x_i = self.atom_embedding(z[i])
        x_j = self.atom_embedding(z[j])
        rbf_feat = self.act(self.rbf_proj(rbf))

        # Combine features to create initial messages
        return x_i * x_j * rbf_feat


class InteractionBlock(nn.Module):
    """
    [cite_start]交互模块: DimeNet的核心，实现定向消息传递 [cite: 269]
    """

    def __init__(self, num_feat, num_radial, num_spherical, envelope_exponent, cutoff):
        super(InteractionBlock, self).__init__()
        self.act = Swish()

        # Projections for basis functions
        self.rbf_proj = nn.Linear(num_radial, num_feat, bias=False)

        self.sbf_proj = nn.Linear(num_spherical * num_radial, 8)  # 8 is N_bilinear in paper [cite: 480]

        # Projections for messages
        self.m_proj = nn.Linear(num_feat, num_feat)
        self.bilinear = nn.Bilinear(8, num_feat, num_feat)

        # Down-projection after aggregation
        self.down_proj = nn.Linear(num_feat, num_feat)

        # Residual connection blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_feat) for _ in range(2)])

    def forward(self, m, rbf, sbf, idx_kj, idx_ji):
        rbf_feat = self.act(self.rbf_proj(rbf))
        sbf_feat = self.act(self.sbf_proj(sbf.view(sbf.shape[0], -1)))

        # Project messages from previous layer
        m_kj = self.act(self.m_proj(m))

        # Aggregate information from neighbors (k) to update message (m_ji)
        # This is the core directional message passing step
        # [cite_start]Equation 4 in the paper describes this aggregation [cite: 138, 139]
        bilinear_out = self.bilinear(sbf_feat, m_kj[idx_kj])

        # The aggregation sum: sum over k in N_j \ {i}
        # torch.index_add_ does this efficiently
        aggregated_m = torch.zeros_like(m)
        aggregated_m.index_add_(0, idx_ji, bilinear_out)

        # Update messages with a residual connection
        m_updated = self.act(self.down_proj(aggregated_m))
        m = m + m_updated

        for res_block in self.res_blocks:
            m = res_block(m)
        return m


class OutputBlock(nn.Module):
    """
    [cite_start]输出模块：将每个交互层之后的消息聚合为原子级别的能量贡献，最后加总得到总能量 [cite: 274, 277]
    """

    def __init__(self, num_feat, num_radial):
        super(OutputBlock, self).__init__()
        self.rbf_proj = nn.Linear(num_radial, num_feat, bias=False)
        self.act = Swish()

        # Layers to transform aggregated messages to energy scalar
        self.lin1 = nn.Linear(num_feat, num_feat // 2)
        self.lin2 = nn.Linear(num_feat // 2, 1)

    def forward(self, m, rbf, i, num_atoms):
        # Project RBF features
        rbf_feat = self.rbf_proj(rbf)

        # Get atom-wise features by summing messages
        # h_i = sum_j m_ji
        atom_wise_feat = torch.zeros(num_atoms, m.shape[1], device=m.device)
        atom_wise_feat.index_add_(0, i, m * rbf_feat)

        # Predict atomic energy contribution
        out = self.lin2(self.act(self.lin1(self.act(atom_wise_feat))))
        return out


# --- The Complete DimeNet Model ---

class DimeNet(nn.Module):
    """
    优化后的DimeNet模型。
    它不再自己计算三元组，而是直接接收预处理好的三元组索引作为输入。
    """

    def __init__(self, num_feat=128, num_blocks=6, num_radial=6, num_spherical=7,
                 cutoff=5.0, envelope_exponent=6):
        super(DimeNet, self).__init__()
        self.num_blocks = num_blocks
        self.rbf = BesselBasis(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasis(num_spherical, num_radial, cutoff, envelope_exponent)
        self.embedding_block = EmbeddingBlock(num_feat, num_radial)
        self.output_blocks = nn.ModuleList([OutputBlock(num_feat, num_radial) for _ in range(num_blocks + 1)])
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(num_feat, num_radial, num_spherical, envelope_exponent, cutoff) for _ in range(num_blocks)
        ])

    def _get_angles(self, pos, i, j, idx_kj, idx_ji):
        pos_i = pos[i[idx_ji]]
        pos_j = pos[j[idx_ji]]
        pos_k = pos[j[idx_kj]]
        vec_ji = pos_i - pos_j
        vec_jk = pos_k - pos_j
        dot_product = (vec_ji * vec_jk).sum(dim=1)
        norm_product = torch.norm(vec_ji, dim=1) * torch.norm(vec_jk, dim=1)
        # 增加一个小的epsilon防止除以零
        angle = torch.acos(torch.clamp(dot_product / (norm_product + 1e-7), -1.0, 1.0))
        return angle

    def forward(self, z, pos, i, j, idx_kj, idx_ji):  # <-- 直接接收三元组索引
        num_atoms = z.shape[0]
        dist = (pos[i] - pos[j]).pow(2).sum(dim=1).sqrt()

        # 不再需要在这里计算三元组和角度
        angles = self._get_angles(pos, i, j, idx_kj, idx_ji)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angles, idx_kj)
        m = self.embedding_block(z, rbf, i, j)
        out = self.output_blocks[0](m, rbf, i, num_atoms)
        for idx in range(self.num_blocks):
            m = self.interaction_blocks[idx](m, rbf, sbf, idx_kj, idx_ji)
            out = out + self.output_blocks[idx + 1](m, rbf, i, num_atoms)
        return out.sum()