import argparse
import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import SeqStructToptPredictor


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProteinDataset(Dataset):
    """
    Expects:
      ids: list[str]
      seq_map[id] -> Tensor [L_seq, d_model]
      struct_map[id] -> Tensor [L_struct, d_model]
      y_map[id] -> float
    """

    def __init__(self, ids, seq_map, struct_map, y_map):
        self.ids = ids
        self.seq_map = seq_map
        self.struct_map = struct_map
        self.y_map = y_map

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        seq = self.seq_map[pid].float()
        struct = self.struct_map[pid].float()
        y = torch.tensor(self.y_map[pid], dtype=torch.float32)
        return seq, struct, y


@dataclass
class Batch:
    seq: torch.Tensor          # [B, S, D]
    struct: torch.Tensor       # [B, T, D]
    y: torch.Tensor            # [B]
    cross_mask: torch.Tensor   # [B, 1, S, T]
    seq_valid: torch.Tensor    # [B, S]


def collate_fn(batch):
    seqs, structs, ys = zip(*batch)
    B = len(seqs)
    D = seqs[0].shape[-1]
    S = max(x.shape[0] for x in seqs)
    T = max(x.shape[0] for x in structs)

    seq_pad = torch.zeros(B, S, D, dtype=torch.float32)
    struct_pad = torch.zeros(B, T, D, dtype=torch.float32)
    seq_valid = torch.zeros(B, S, dtype=torch.bool)
    struct_valid = torch.zeros(B, T, dtype=torch.bool)

    for i, (s, t) in enumerate(zip(seqs, structs)):
        seq_pad[i, : s.shape[0]] = s
        struct_pad[i, : t.shape[0]] = t
        seq_valid[i, : s.shape[0]] = True
        struct_valid[i, : t.shape[0]] = True

    # For attention scores [B, H, S, T], broadcastable mask [B, 1, S, T]
    cross_mask = (seq_valid.unsqueeze(-1) & struct_valid.unsqueeze(1)).unsqueeze(1)
    y = torch.stack(ys)
    return Batch(seq_pad, struct_pad, y, cross_mask, seq_valid)


def forward_masked_pool(model, seq, struct, cross_mask, seq_valid):
    # Same as model.forward, but masked mean pooling so padded seq tokens do not bias prediction
    attn_out = model.cross_attn(x=seq, context=struct, mask=cross_mask)
    x = model.layer_norm1(seq + attn_out)
    ffn_out = model.ffn(x)
    x = model.layer_norm2(x + ffn_out)

    # masked mean over sequence length
    seq_valid_f = seq_valid.float().unsqueeze(-1)  # [B, S, 1]
    x_sum = (x * seq_valid_f).sum(dim=1)
    denom = seq_valid_f.sum(dim=1).clamp_min(1.0)
    pooled = x_sum / denom

    pred = model.regressor(pooled).squeeze(-1)
    return pred


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    mae_sum = 0.0
    mse_sum = 0.0

    for batch in loader:
        seq = batch.seq.to(device)
        struct = batch.struct.to(device)
        y = batch.y.to(device)
        cross_mask = batch.cross_mask.to(device)
        seq_valid = batch.seq_valid.to(device)

        pred = forward_masked_pool(model, seq, struct, cross_mask, seq_valid)
        loss = criterion(pred, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs

        err = pred - y
        mae_sum += err.abs().sum().item()
        mse_sum += (err ** 2).sum().item()

    mae = mae_sum / n
    rmse = math.sqrt(mse_sum / n)
    return total_loss / n, mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dataset.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Expected keys inside dataset.pt:
    # {
    #   "ids": list[str],
    #   "seq_map": dict[id, Tensor[L_seq, D]],
    #   "struct_map": dict[id, Tensor[L_struct, D]],
    #   "y_map": dict[id, float]
    # }
    data = torch.load(args.data_path, map_location="cpu")
    ids = data["ids"]
    seq_map = data["seq_map"]
    struct_map = data["struct_map"]
    y_map = data["y_map"]

    train_ids, temp_ids = train_test_split(ids, test_size=0.2, random_state=args.seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=args.seed)

    train_ds = ProteinDataset(train_ids, seq_map, struct_map, y_map)
    val_ds = ProteinDataset(val_ids, seq_map, struct_map, y_map)
    test_ds = ProteinDataset(test_ids, seq_map, struct_map, y_map)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    model = SeqStructToptPredictor(d_model=args.d_model, num_heads=args.num_heads).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in train_loader:
            seq = batch.seq.to(device)
            struct = batch.struct.to(device)
            y = batch.y.to(device)
            cross_mask = batch.cross_mask.to(device)
            seq_valid = batch.seq_valid.to(device)

            optimizer.zero_grad()
            pred = forward_masked_pool(model, seq, struct, cross_mask, seq_valid)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running += loss.item() * bs
            n += bs

        train_loss = running / n
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d} | "
            f"train_mse={train_loss:.4f} | "
            f"val_mse={val_loss:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_mse": best_val,
                },
                best_path,
            )

    print(f"Best val MSE: {best_val:.4f}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, criterion)
    print(f"Test  | mse={test_mse:.4f} | mae={test_mae:.4f} | rmse={test_rmse:.4f}")


if __name__ == "__main__":
    main()
