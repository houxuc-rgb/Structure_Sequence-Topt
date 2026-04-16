import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, context, mask=None):
        batch_size = x.shape[0]

        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.fc_out(out)


class IntraModalFusion(nn.Module):
    """Bidirectional cross-attention fusion between two branches within one modality."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn_a2b = CrossAttention(d_model, num_heads)
        self.cross_attn_b2a = CrossAttention(d_model, num_heads)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, feat_a, feat_b):
        a2b = self.cross_attn_a2b(x=feat_a, context=feat_b)
        b2a = self.cross_attn_b2a(x=feat_b, context=feat_a)

        fused = self.layer_norm1(feat_a + a2b) + self.layer_norm2(feat_b + b2a)

        ffn_out = self.ffn(fused)
        fused = self.layer_norm3(fused + ffn_out)

        return fused
