import argparse
import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import SeqStructToptPredictor


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    def __init__(self, ids, seq_map, struct_map, y_map):
        self.ids = ids
        self.seq_map = seq_map
        self.struct_map = struct_map
        self.y_map = y_map

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        return (
            self.seq_map[pid].float(),
            self.struct_map[pid].float(),
            torch.tensor(self.y_map[pid], dtype=torch.float32),
        )


@dataclass
class Batch:
    seq: torch.Tensor         # [B, S, D_seq]
    struct: torch.Tensor      # [B, T, D_struct]
    y: torch.Tensor           # [B]
    cross_mask: torch.Tensor  # [B, 1, S, T]
    seq_valid: torch.Tensor   # [B, S]


def collate_fn(batch):
    seqs, structs, ys = zip(*batch)
    B      = len(seqs)
    D_seq  = seqs[0].shape[-1]
    D_str  = structs[0].shape[-1]
    S      = max(x.shape[0] for x in seqs)
    T      = max(x.shape[0] for x in structs)

    seq_pad      = torch.zeros(B, S, D_seq)
    struct_pad   = torch.zeros(B, T, D_str)
    seq_valid    = torch.zeros(B, S, dtype=torch.bool)
    struct_valid = torch.zeros(B, T, dtype=torch.bool)

    for i, (s, t) in enumerate(zip(seqs, structs)):
        seq_pad[i,     : s.shape[0]] = s
        struct_pad[i,  : t.shape[0]] = t
        seq_valid[i,   : s.shape[0]] = True
        struct_valid[i,: t.shape[0]] = True

    # cross_mask[b, 0, s, t] = True  iff  both positions are real tokens
    cross_mask = (seq_valid.unsqueeze(-1) & struct_valid.unsqueeze(1)).unsqueeze(1)
    return Batch(seq_pad, struct_pad, torch.stack(ys), cross_mask, seq_valid)


# ---------------------------------------------------------------------------
# Forward pass (masked mean pooling)
# ---------------------------------------------------------------------------

def forward_pass(model, seq, struct, cross_mask, seq_valid):
    """
    Applies optional input projections, cross-attention + FFN,
    masked mean pooling, then the regression head.
    """
    if hasattr(model, "seq_proj"):
        seq = model.seq_proj(seq)
    if hasattr(model, "struct_proj"):
        struct = model.struct_proj(struct)

    attn_out = model.cross_attn(x=seq, context=struct, mask=cross_mask)
    x = model.layer_norm1(seq + attn_out)
    x = model.layer_norm2(x + model.ffn(x))

    # Masked mean pool — ignore padding tokens
    seq_valid_f = seq_valid.float().unsqueeze(-1)          # [B, S, 1]
    pooled = (x * seq_valid_f).sum(1) / seq_valid_f.sum(1).clamp_min(1.0)

    return model.regressor(pooled).squeeze(-1)             # [B]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = mae_sum = mse_sum = 0.0
    n = 0

    for batch in loader:
        seq        = batch.seq.to(device)
        struct     = batch.struct.to(device)
        y          = batch.y.to(device)
        cross_mask = batch.cross_mask.to(device)
        seq_valid  = batch.seq_valid.to(device)

        pred        = forward_pass(model, seq, struct, cross_mask, seq_valid)
        loss        = criterion(pred, y)
        bs          = y.size(0)
        total_loss += loss.item() * bs
        err         = pred - y
        mae_sum    += err.abs().sum().item()
        mse_sum    += (err ** 2).sum().item()
        n          += bs

    return total_loss / n, mae_sum / n, math.sqrt(mse_sum / n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    type=str,   default="data/dataset.pt")
    parser.add_argument("--save_dir",     type=str,   default="checkpoints")
    parser.add_argument("--epochs",       type=int,   default=40)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model",      type=int,   default=1024)
    parser.add_argument("--num_heads",    type=int,   default=8)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_workers",  type=int,   default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading dataset from {args.data_path} ...")
    data       = torch.load(args.data_path, map_location="cpu")
    seq_map    = data["seq_map"]
    struct_map = data["struct_map"]
    y_map      = data["y_map"]
    train_ids  = data["train_ids"]
    val_ids    = data["val_ids"]
    test_ids   = data["test_ids"]

    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    sample_id = train_ids[0]
    d_seq     = seq_map[sample_id].shape[-1]
    d_struct  = struct_map[sample_id].shape[-1]
    print(f"  Embedding dims — seq: {d_seq}  struct: {d_struct}")

    def make_loader(ids, shuffle):
        ds = ProteinDataset(ids, seq_map, struct_map, y_map)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            persistent_workers=(args.num_workers > 0),
        )

    train_loader = make_loader(train_ids, shuffle=True)
    val_loader   = make_loader(val_ids,   shuffle=False)
    test_loader  = make_loader(test_ids,  shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = SeqStructToptPredictor(d_model=args.d_model, num_heads=args.num_heads)

    if d_seq != args.d_model:
        model.seq_proj = nn.Linear(d_seq, args.d_model)
        print(f"  Added seq_proj:    {d_seq} → {args.d_model}")
    if d_struct != args.d_model:
        model.struct_proj = nn.Linear(d_struct, args.d_model)
        print(f"  Added struct_proj: {d_struct} → {args.d_model}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer / scheduler / loss
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val   = float("inf")
    best_path  = os.path.join(args.save_dir, "best_model.pt")
    no_improve = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = n = 0

        for batch in train_loader:
            seq        = batch.seq.to(device)
            struct     = batch.struct.to(device)
            y          = batch.y.to(device)
            cross_mask = batch.cross_mask.to(device)
            seq_valid  = batch.seq_valid.to(device)

            optimizer.zero_grad()
            pred = forward_pass(model, seq, struct, cross_mask, seq_valid)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item() * y.size(0)
            n       += y.size(0)

        scheduler.step()
        train_loss = running / n
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"train_mse={train_loss:.4f} | "
            f"val_mse={val_loss:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}"
        )

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(
                {
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch":                epoch,
                    "args":                 vars(args),
                    "best_val_mse":         best_val,
                },
                best_path,
            )
            print(f"  ✓ Saved best model (val_mse={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping — no improvement for {args.patience} epochs.")
                break

    # ------------------------------------------------------------------
    # Test set evaluation with best checkpoint
    # ------------------------------------------------------------------
    print(f"\nBest val MSE: {best_val:.4f}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, criterion)
    print(f"Test | mse={test_mse:.4f} | mae={test_mae:.4f} | rmse={test_rmse:.4f}")


if __name__ == "__main__":
    main()
