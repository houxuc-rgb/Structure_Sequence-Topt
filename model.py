import torch
import torch.nn as nn

try:
    from torch_geometric.utils import to_dense_batch
except ImportError:
    to_dense_batch = None

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

        self.seq_cnn = CNNBranch(d_model, num_layers=3, kernel_size=5)
        self.seq_transformer = TransformerBranch(d_model, num_heads, num_layers=2)
        self.seq_fusion = IntraModalFusion(d_model, num_heads)

        self.struct_gcn = GCNBranch(d_model, num_layers=3)
        self.struct_transformer = TransformerBranch(d_model, num_heads, num_layers=2)
        self.struct_fusion = IntraModalFusion(d_model, num_heads)

        self.cross_modal_attn = CrossAttention(d_model, num_heads)
        self.cross_modal_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _dense_sequence(self, seq_flat, seq_len, batch_size):
        seq_lengths = seq_len.view(-1).to(dtype=torch.long, device=seq_flat.device)
        seq_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=seq_flat.device),
            seq_lengths,
        )
        return to_dense_batch(seq_flat, seq_batch, batch_size=batch_size)

    def forward(self, data):
        """
        Args:
            data: PyG Batch with:
                seq:       (total_seq_tokens, d_seq)
                seq_len:   (batch_size,)
                x:         (total_structure_nodes, d_struct)
                edge_index:(2, total_edges)
                batch:     (total_structure_nodes,)
        Returns:
            t_opt_pred: (batch_size,)
        """
        if to_dense_batch is None:
            raise ImportError(
                "SeqStructToptPredictor requires torch_geometric.utils.to_dense_batch."
            )

        batch_size = data.seq_len.view(-1).numel()
        struct_batch = getattr(
            data,
            "batch",
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device),
        )

        seq_flat = data.seq
        struct_flat = data.x

        if hasattr(self, "seq_proj"):
            seq_flat = self.seq_proj(seq_flat)
        if hasattr(self, "struct_proj"):
            struct_flat = self.struct_proj(struct_flat)

        seq_emb, seq_valid = self._dense_sequence(seq_flat, data.seq_len, batch_size)
        seq_mask = ~seq_valid

        struct_emb, struct_valid = to_dense_batch(
            struct_flat,
            struct_batch,
            batch_size=batch_size,
        )
        struct_mask = ~struct_valid

        seq_cnn_feat = self.seq_cnn(seq_emb)
        seq_trans_feat = self.seq_transformer(seq_emb, mask=seq_mask)
        seq_fused = self.seq_fusion(seq_cnn_feat, seq_trans_feat)

        struct_gcn_flat = self.struct_gcn(struct_flat, data.edge_index)
        struct_gcn_feat, _ = to_dense_batch(
            struct_gcn_flat,
            struct_batch,
            batch_size=batch_size,
        )
        struct_trans_feat = self.struct_transformer(struct_emb, mask=struct_mask)
        struct_fused = self.struct_fusion(struct_gcn_feat, struct_trans_feat)

        cross_out = self.cross_modal_attn(
            x=seq_fused,
            context=struct_fused,
            mask=struct_mask,
        )
        x = self.cross_modal_norm(seq_fused + cross_out)

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        valid_mask = seq_valid.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        t_opt_pred = self.regressor(pooled)
        return t_opt_pred.squeeze(-1)
