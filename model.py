import torch
import torch.nn as nn

from encoders import (
    CrossAttention,
    IntraModalFusion,
    CNNBranch,
    GCNBranch,
    TransformerBranch,
)


class SeqStructToptPredictor(nn.Module):
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()

        # --- Sequence Stream ---
        self.seq_cnn = CNNBranch(d_model, num_layers=3, kernel_size=5)
        self.seq_transformer = TransformerBranch(d_model, num_heads, num_layers=2)
        self.seq_fusion = IntraModalFusion(d_model, num_heads)

        # --- Structure Stream ---
        self.struct_gcn = GCNBranch(d_model, num_layers=3)
        self.struct_transformer = TransformerBranch(d_model, num_heads, num_layers=2)
        self.struct_fusion = IntraModalFusion(d_model, num_heads)

        # --- Final Cross-Modal Fusion (Sequence ↔ Structure) ---
        self.cross_modal_attn = CrossAttention(d_model, num_heads)
        self.cross_modal_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # --- Regression Head for Topt ---
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq_emb, struct_emb, adj, seq_mask=None, struct_mask=None):
        """
        Args:
            seq_emb:     (batch_size, seq_len, d_model)    - sequence embeddings
            struct_emb:  (batch_size, num_nodes, d_model)  - structure embeddings
            adj:         (batch_size, num_nodes, num_nodes) - adjacency matrix for GCN
            seq_mask:    optional padding mask for sequence
            struct_mask: optional padding mask for structure
        Returns:
            t_opt_pred:  (batch_size,) - predicted Topt values
        """

        if hasattr(self, 'seq_proj'):
            seq_emb = self.seq_proj(seq_emb)
        if hasattr(self, 'struct_proj'):
            struct_emb = self.struct_proj(struct_emb)

        # ===== Sequence Stream =====
        seq_cnn_feat = self.seq_cnn(seq_emb)
        seq_trans_feat = self.seq_transformer(seq_emb, mask=seq_mask)
        seq_fused = self.seq_fusion(seq_cnn_feat, seq_trans_feat)

        # ===== Structure Stream =====
        struct_gcn_feat = self.struct_gcn(struct_emb, adj)
        struct_trans_feat = self.struct_transformer(struct_emb, mask=struct_mask)
        struct_fused = self.struct_fusion(struct_gcn_feat, struct_trans_feat)

        # ===== Final Cross-Modal Fusion (Sequence ↔ Structure) =====
        cross_out = self.cross_modal_attn(x=seq_fused, context=struct_fused)
        x = self.cross_modal_norm(seq_fused + cross_out)

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        # ===== Global Average Pooling =====
        pooled = torch.mean(x, dim=1)

        # ===== Predict Topt =====
        t_opt_pred = self.regressor(pooled)

        return t_opt_pred.squeeze(-1)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 128
    num_nodes = 128
    d_model = 1024

    seq_emb = torch.randn(batch_size, seq_len, d_model)
    struct_emb = torch.randn(batch_size, num_nodes, d_model)
    adj = torch.rand(batch_size, num_nodes, num_nodes)
    adj = (adj + adj.transpose(-1, -2)) / 2
    adj = adj / adj.sum(dim=-1, keepdim=True)

    model = SeqStructToptPredictor(d_model=d_model, num_heads=8)
    pred = model(seq_emb, struct_emb, adj)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input  - seq: {seq_emb.shape}, struct: {struct_emb.shape}, adj: {adj.shape}")
    print(f"Output - Topt predictions: {pred.shape}")
    print(f"Sample predictions: {pred.detach().numpy()}")

class SeqStructToptPredictor(nn.Module):
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()

        # --- Sequence Stream ---
        self.seq_cnn = CNNBranch(d_model, num_layers=3, kernel_size=5)
        self.seq_transformer = TransformerBranch(d_model, num_heads, num_layers=2)
        self.seq_fusion = IntraModalFusion(d_model, num_heads)

        # --- Structure Stream ---
        self.struct_gcn = GCNBranch(d_model, num_layers=3)
        self.struct_transformer = TransformerBranch(d_model, num_heads, num_layers=2)
        self.struct_fusion = IntraModalFusion(d_model, num_heads)

        # --- Final Cross-Modal Fusion (Sequence ↔ Structure) ---
        self.cross_modal_attn = CrossAttention(d_model, num_heads)
        self.cross_modal_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # --- Regression Head for Topt ---
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq_emb, struct_emb, adj, seq_mask=None, struct_mask=None):
        """
        Args:
            seq_emb:     (batch_size, seq_len, d_model)    - sequence embeddings
            struct_emb:  (batch_size, num_nodes, d_model)  - structure embeddings
            adj:         (batch_size, num_nodes, num_nodes) - adjacency matrix for GCN
            seq_mask:    optional padding mask for sequence
            struct_mask: optional padding mask for structure
        Returns:
            t_opt_pred:  (batch_size,) - predicted Topt values
        """

        if hasattr(self, 'seq_proj'):
            seq_emb = self.seq_proj(seq_emb)
        if hasattr(self, 'struct_proj'):
            struct_emb = self.struct_proj(struct_emb)

        # ===== Sequence Stream =====
        seq_cnn_feat = self.seq_cnn(seq_emb)
        seq_trans_feat = self.seq_transformer(seq_emb, mask=seq_mask)
        seq_fused = self.seq_fusion(seq_cnn_feat, seq_trans_feat)

        # ===== Structure Stream =====
        struct_gcn_feat = self.struct_gcn(struct_emb, adj)
        struct_trans_feat = self.struct_transformer(struct_emb, mask=struct_mask)
        struct_fused = self.struct_fusion(struct_gcn_feat, struct_trans_feat)

        # ===== Final Cross-Modal Fusion (Sequence ↔ Structure) =====
        cross_out = self.cross_modal_attn(x=seq_fused, context=struct_fused)
        x = self.cross_modal_norm(seq_fused + cross_out)

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        # ===== Global Average Pooling =====
        pooled = torch.mean(x, dim=1)

        # ===== Predict Topt =====
        t_opt_pred = self.regressor(pooled)

        return t_opt_pred.squeeze(-1)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 128
    num_nodes = 128
    d_model = 1024

    seq_emb = torch.randn(batch_size, seq_len, d_model)
    struct_emb = torch.randn(batch_size, num_nodes, d_model)
    adj = torch.rand(batch_size, num_nodes, num_nodes)
    adj = (adj + adj.transpose(-1, -2)) / 2
    adj = adj / adj.sum(dim=-1, keepdim=True)

    model = SeqStructToptPredictor(d_model=d_model, num_heads=8)
    pred = model(seq_emb, struct_emb, adj)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input  - seq: {seq_emb.shape}, struct: {struct_emb.shape}, adj: {adj.shape}")
    print(f"Output - Topt predictions: {pred.shape}")
    print(f"Sample predictions: {pred.detach().numpy()}")
