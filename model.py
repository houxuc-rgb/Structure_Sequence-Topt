import torch
import torch.nn as nn
import torch.nn.functional as F

#Cross Attention Block
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)  # Queries (Sequence)
        self.w_k = nn.Linear(d_model, d_model)  # Keys (Structure)
        self.w_v = nn.Linear(d_model, d_model)  # Values (Structure)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, context, mask=None):
        batch_size = x.shape[0]

        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.fc_out(out)

# 2. The Main Predictor Network
class SeqStructToptPredictor(nn.Module):
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()

        # Cross Attention: fusing Sequence and Structure
        self.cross_attn = CrossAttention(d_model, num_heads)

        # Standard Transformer Add & Norm layer
        self.layer_norm1 = nn.LayerNorm(d_model)

        # Feed Forward Network to process the fused features
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

        # --- Regression Head for Topt ---
        # We pool the sequence and map it to a single float value
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: 1 neuron for the Topt scalar
        )

    def forward(self, seq_emb, struct_emb, mask=None):
        """
        seq_emb: Tensor of shape (batch_size, seq_len, d_model)
        struct_emb: Tensor of shape (batch_size, struct_len, d_model)
        """

        # 1. Fuse Modalities
        # Sequence acts as Query (x), Structure acts as Context (context)
        attn_out = self.cross_attn(x=seq_emb, context=struct_emb, mask=mask)

        # 2. Residual Connection & Normalization
        x = self.layer_norm1(seq_emb + attn_out)

        # 3. Feed Forward & Residual
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        # x shape is still (batch_size, seq_len, d_model)

        # 4. Global Average Pooling
        # Averages across the sequence length to get a single vector per protein
        # (batch_size, seq_len, d_model) -> (batch_size, d_model)
        pooled_features = torch.mean(x, dim=1)

        # 5. Predict Topt
        # (batch_size, d_model) -> (batch_size, 1)
        t_opt_pred = self.regressor(pooled_features)

        return t_opt_pred.squeeze(-1)  # Output shape: (batch_size,)