import argparse
import math
import os
import pickle
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import SeqStructToptPredictor
from utils.contact_graph import build_contact_adj_for_id


# ---------------------------------------------------------------------------
# Load embeddings from pkl directories
# ---------------------------------------------------------------------------

def load_embeddings(esmc_dir, saprot_dir):
    """Load sequence (ESMC) and structure (SaProt) embeddings, return only common IDs."""
    esmc_files  = {f.split(" ")[0]: f for f in os.listdir(esmc_dir)  if f.endswith(".pkl")}
    saprot_files = {f.replace(".pkl", ""): f for f in os.listdir(saprot_dir) if f.endswith(".pkl")}

    common_ids = sorted(set(esmc_files) & set(saprot_files))
    print(f"  ESMC={len(esmc_files)}  SaProt={len(saprot_files)}  common={len(common_ids)}")

    seq_map, struct_map, y_map = {}, {}, {}
    for uid in common_ids:
        # Parse Topt from filename: "{uid} Topt={value}.pkl"
        topt_str = esmc_files[uid].replace(uid, "").replace(" Topt=", "").replace(".pkl", "")
        y_map[uid] = float(topt_str)

        with open(os.path.join(esmc_dir, esmc_files[uid]), "rb") as f:
            seq_emb = pickle.load(f)
        with open(os.path.join(saprot_dir, saprot_files[uid]), "rb") as f:
            struct_emb = pickle.load(f)

        # Convert to torch tensors — handle numpy arrays or existing tensors
        seq_map[uid]    = torch.as_tensor(seq_emb).float()
        struct_map[uid] = torch.as_tensor(struct_emb).float()

    return common_ids, seq_map, struct_map, y_map


def build_sequential_adj(n):
    """Build a normalised tridiagonal adjacency (chain connectivity) of size n×n."""
    adj = torch.eye(n)
    if n > 1:
        idx = torch.arange(n - 1)
        adj[idx, idx + 1] = 1.0
        adj[idx + 1, idx] = 1.0
    # Row-normalise
    row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj / row_sum


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    def __init__(
        self,
        ids,
        seq_map,
        struct_map,
        y_map,
        adj_map=None,
        graph_type="contact",
        pdb_dir="data/pdbs",
        contact_cutoff=8.0,
        strict_contact_graphs=False,
    ):
        self.ids = ids
        self.seq_map = seq_map
        self.struct_map = struct_map
        self.adj_map = adj_map or {}
        self.y_map = y_map
        self.graph_type = graph_type
        self.pdb_dir = pdb_dir
        self.contact_cutoff = contact_cutoff
        self.strict_contact_graphs = strict_contact_graphs
        self._contact_adj_cache = {}
        self._fallback_warning_count = 0

    def __len__(self):
        return len(self.ids)

    def _get_adj(self, pid, num_nodes):
        if pid in self.adj_map:
            return self.adj_map[pid]

        if self.graph_type == "sequential":
            return build_sequential_adj(num_nodes)

        if pid not in self._contact_adj_cache:
            try:
                self._contact_adj_cache[pid] = build_contact_adj_for_id(
                    pid,
                    self.pdb_dir,
                    expected_len=num_nodes,
                    cutoff=self.contact_cutoff,
                )
            except Exception as exc:
                if self.strict_contact_graphs:
                    raise
                if self._fallback_warning_count < 10:
                    print(
                        f"Warning: falling back to sequential adjacency for {pid}: {exc}"
                    )
                    self._fallback_warning_count += 1
                self._contact_adj_cache[pid] = build_sequential_adj(num_nodes)

        return self._contact_adj_cache[pid]

    def __getitem__(self, idx):
        pid = self.ids[idx]
        struct = self.struct_map[pid].float()
        adj = self._get_adj(pid, struct.shape[0])
        return (
            self.seq_map[pid].float(),
            struct,
            adj,
            torch.tensor(self.y_map[pid], dtype=torch.float32),
        )


@dataclass
class Batch:
    seq: torch.Tensor          # [B, S, D_seq]
    struct: torch.Tensor       # [B, N, D_struct]
    adj: torch.Tensor          # [B, N, N]
    y: torch.Tensor            # [B]
    seq_mask: torch.Tensor     # [B, S]  True = padding (for Transformer src_key_padding_mask)
    struct_mask: torch.Tensor  # [B, N]  True = padding
    seq_valid: torch.Tensor    # [B, S]  True = real token (for masked pooling)


def collate_fn(batch):
    seqs, structs, adjs, ys = zip(*batch)
    B     = len(seqs)
    D_seq = seqs[0].shape[-1]
    D_str = structs[0].shape[-1]
    S     = max(x.shape[0] for x in seqs)
    N     = max(x.shape[0] for x in structs)

    seq_pad      = torch.zeros(B, S, D_seq)
    struct_pad   = torch.zeros(B, N, D_str)
    adj_pad      = torch.zeros(B, N, N)
    seq_valid    = torch.zeros(B, S, dtype=torch.bool)
    struct_valid = torch.zeros(B, N, dtype=torch.bool)

    for i, (s, t, a) in enumerate(zip(seqs, structs, adjs)):
        sl, nl = s.shape[0], t.shape[0]
        seq_pad[i, :sl]          = s
        struct_pad[i, :nl]       = t
        adj_pad[i, :nl, :nl]     = a
        seq_valid[i, :sl]        = True
        struct_valid[i, :nl]     = True

    # Transformer src_key_padding_mask: True = IGNORE (padding positions)
    seq_mask    = ~seq_valid
    struct_mask = ~struct_valid

    return Batch(
        seq_pad, struct_pad, adj_pad,
        torch.stack(ys),
        seq_mask, struct_mask, seq_valid,
    )


# ---------------------------------------------------------------------------
# Masked mean pooling utility
# ---------------------------------------------------------------------------

def masked_mean_pool(x, valid_mask):
    """
    x:          (B, S, D)
    valid_mask: (B, S) boolean — True for real tokens
    returns:    (B, D)
    """
    mask_f = valid_mask.float().unsqueeze(-1)          # [B, S, 1]
    return (x * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = mae_sum = mse_sum = 0.0
    n = 0

    for batch in loader:
        seq         = batch.seq.to(device)
        struct      = batch.struct.to(device)
        adj         = batch.adj.to(device)
        y           = batch.y.to(device)
        seq_mask    = batch.seq_mask.to(device)
        struct_mask = batch.struct_mask.to(device)

        pred = model(seq, struct, adj, seq_mask=seq_mask, struct_mask=struct_mask)
        loss = criterion(pred, y)
        bs   = y.size(0)

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
    parser.add_argument("--esmc_dir",     type=str,   default="embeddings/esmc_600m_features")
    parser.add_argument("--saprot_dir",   type=str,   default="embeddings/saprot_650m_features")
    parser.add_argument("--pdb_dir",      type=str,   default="data/pdbs")
    parser.add_argument("--graph_type",   type=str,   default="contact",
                        choices=["contact", "sequential"],
                        help="Adjacency for the GCN branch. Use sequential for the old baseline.")
    parser.add_argument("--contact_cutoff", type=float, default=8.0,
                        help="C-alpha distance cutoff in Angstrom for contact graph edges.")
    parser.add_argument("--strict_contact_graphs", action="store_true",
                        help="Fail instead of falling back when a contact graph cannot be built.")
    parser.add_argument("--val_split",    type=float, default=0.1)
    parser.add_argument("--test_split",   type=float, default=0.1)
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
    parser.add_argument("--gpus",         type=int,   nargs="+", default=[2,3],
                        help="GPU IDs to use, e.g. --gpus 0 1 2. Defaults to [2, 3].")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        gpu_ids = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.zeros(1).to(device)   # initialize CUDA context before any model ops
        print(f"Using GPUs: {gpu_ids}  (primary: {device})")
    else:
        gpu_ids = []
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading embeddings from:\n  seq:    {args.esmc_dir}\n  struct: {args.saprot_dir}")
    all_ids, seq_map, struct_map, y_map = load_embeddings(args.esmc_dir, args.saprot_dir)

    random.shuffle(all_ids)
    n = len(all_ids)
    n_test = max(1, int(n * args.test_split))
    n_val  = max(1, int(n * args.val_split))
    test_ids  = all_ids[:n_test]
    val_ids   = all_ids[n_test:n_test + n_val]
    train_ids = all_ids[n_test + n_val:]

    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    sample_id = train_ids[0]
    d_seq     = seq_map[sample_id].shape[-1]
    d_struct  = struct_map[sample_id].shape[-1]
    print(f"  Embedding dims — seq: {d_seq}  struct: {d_struct}")

    def make_loader(ids, shuffle):
        ds = ProteinDataset(
            ids,
            seq_map,
            struct_map,
            y_map,
            graph_type=args.graph_type,
            pdb_dir=args.pdb_dir,
            contact_cutoff=args.contact_cutoff,
            strict_contact_graphs=args.strict_contact_graphs,
        )
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
    # Add projection layers if embedding dims don't match d_model
    if d_seq != args.d_model:
        model.seq_proj = nn.Linear(d_seq, args.d_model)
        print(f"  Added seq_proj:    {d_seq} → {args.d_model}")
    if d_struct != args.d_model:
        model.struct_proj = nn.Linear(d_struct, args.d_model)
        print(f"  Added struct_proj: {d_struct} → {args.d_model}")

    model = model.to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  Wrapped with DataParallel over GPUs {gpu_ids}")
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
            seq         = batch.seq.to(device)
            struct      = batch.struct.to(device)
            adj         = batch.adj.to(device)
            y           = batch.y.to(device)
            seq_mask    = batch.seq_mask.to(device)
            struct_mask = batch.struct_mask.to(device)

            optimizer.zero_grad()
            pred = model(seq, struct, adj, seq_mask=seq_mask, struct_mask=struct_mask)
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
            _state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "model_state_dict":    _state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch":               epoch,
                    "args":                vars(args),
                    "best_val_mse":        best_val,
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
    _m = model.module if isinstance(model, nn.DataParallel) else model
    _m.load_state_dict(ckpt["model_state_dict"])
    test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, criterion)
    print(f"Test | mse={test_mse:.4f} | mae={test_mae:.4f} | rmse={test_rmse:.4f}")


if __name__ == "__main__":
    main()
