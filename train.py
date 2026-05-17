import argparse
import math
import os
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError as exc:
    raise ImportError(
        "Standard PyG training requires torch_geometric. Install PyTorch Geometric "
        "before running train.py."
    ) from exc

from model import SeqStructToptPredictor
from utils.contact_graph import build_contact_edge_index_for_id


def load_embeddings(esmc_dir, saprot_dir):
    """Load sequence and structure embeddings, returning only common protein IDs."""
    esmc_files = {f.split(" ")[0]: f for f in os.listdir(esmc_dir) if f.endswith(".pkl")}
    saprot_files = {f.replace(".pkl", ""): f for f in os.listdir(saprot_dir) if f.endswith(".pkl")}

    common_ids = sorted(set(esmc_files) & set(saprot_files))
    print(f"  ESMC={len(esmc_files)}  SaProt={len(saprot_files)}  common={len(common_ids)}")

    seq_map, struct_map, y_map = {}, {}, {}
    for uid in common_ids:
        topt_str = esmc_files[uid].replace(uid, "").replace(" Topt=", "").replace(".pkl", "")
        y_map[uid] = float(topt_str)

        with open(os.path.join(esmc_dir, esmc_files[uid]), "rb") as f:
            seq_emb = pickle.load(f)
        with open(os.path.join(saprot_dir, saprot_files[uid]), "rb") as f:
            struct_emb = pickle.load(f)

        seq_map[uid] = torch.as_tensor(seq_emb).float()
        struct_map[uid] = torch.as_tensor(struct_emb).float()

    return common_ids, seq_map, struct_map, y_map


def build_sequential_edge_index(num_nodes):
    """Build an undirected sequence-neighbor edge_index without self-loops."""
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    idx = torch.arange(num_nodes - 1, dtype=torch.long)
    forward = torch.stack([idx, idx + 1], dim=0)
    backward = torch.stack([idx + 1, idx], dim=0)
    return torch.cat([forward, backward], dim=1)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProteinDataset(Dataset):
    def __init__(
        self,
        ids,
        seq_map,
        struct_map,
        y_map,
        edge_index_map=None,
        graph_type="contact",
        pdb_dir="data/pdbs",
        contact_cutoff=8.0,
        contact_graph_backend="auto",
        strict_contact_graphs=False,
    ):
        self.ids = ids
        self.seq_map = seq_map
        self.struct_map = struct_map
        self.y_map = y_map
        self.edge_index_map = edge_index_map or {}
        self.graph_type = graph_type
        self.pdb_dir = pdb_dir
        self.contact_cutoff = contact_cutoff
        self.contact_graph_backend = contact_graph_backend
        self.strict_contact_graphs = strict_contact_graphs
        self._edge_index_cache = {}
        self._fallback_warning_count = 0

    def __len__(self):
        return len(self.ids)

    def _get_edge_index(self, pid, num_nodes):
        if pid in self.edge_index_map:
            return self.edge_index_map[pid]

        if self.graph_type == "sequential":
            return build_sequential_edge_index(num_nodes)

        if pid not in self._edge_index_cache:
            try:
                self._edge_index_cache[pid] = build_contact_edge_index_for_id(
                    pid,
                    self.pdb_dir,
                    expected_len=num_nodes,
                    cutoff=self.contact_cutoff,
                    backend=self.contact_graph_backend,
                )
            except Exception as exc:
                if self.strict_contact_graphs:
                    raise
                if self._fallback_warning_count < 10:
                    print(
                        f"Warning: falling back to sequential edge_index for {pid}: {exc}"
                    )
                    self._fallback_warning_count += 1
                self._edge_index_cache[pid] = build_sequential_edge_index(num_nodes)

        return self._edge_index_cache[pid]

    def __getitem__(self, idx):
        pid = self.ids[idx]
        seq = self.seq_map[pid].float()
        struct = self.struct_map[pid].float()
        edge_index = self._get_edge_index(pid, struct.shape[0]).long()

        return Data(
            x=struct,
            edge_index=edge_index,
            seq=seq,
            seq_len=torch.tensor([seq.shape[0]], dtype=torch.long),
            struct_len=torch.tensor([struct.shape[0]], dtype=torch.long),
            y=torch.tensor([self.y_map[pid]], dtype=torch.float32),
        )


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = mae_sum = mse_sum = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1)
        pred = model(batch)
        loss = criterion(pred, y)
        bs = y.numel()

        total_loss += loss.item() * bs
        err = pred - y
        mae_sum += err.abs().sum().item()
        mse_sum += (err ** 2).sum().item()
        n += bs

    return total_loss / n, mae_sum / n, math.sqrt(mse_sum / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esmc_dir", type=str, default="embeddings/esmc_600m_features")
    parser.add_argument("--saprot_dir", type=str, default="embeddings/saprot_650m_features")
    parser.add_argument("--pdb_dir", type=str, default="data/pdbs")
    parser.add_argument(
        "--graph_type",
        type=str,
        default="contact",
        choices=["contact", "sequential"],
        help="Edge construction for the PyG GCN branch.",
    )
    parser.add_argument(
        "--contact_cutoff",
        type=float,
        default=8.0,
        help="C-alpha distance cutoff in Angstrom for contact graph edges.",
    )
    parser.add_argument(
        "--contact_graph_backend",
        type=str,
        default="auto",
        choices=["auto", "pyg", "torch"],
        help="Backend for contact edge construction. auto uses PyG radius_graph when available.",
    )
    parser.add_argument(
        "--strict_contact_graphs",
        action="store_true",
        help="Fail instead of falling back when a contact graph cannot be built.",
    )
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[2, 3],
        help="GPU IDs to use. Standard PyG path uses the first ID as the primary device.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        gpu_ids = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.zeros(1).to(device)
        print(f"Using GPU: {gpu_ids[0]}")
        if len(gpu_ids) > 1:
            print(
                "Note: standard PyG batching is using a single primary GPU in this training script."
            )
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    print(f"Loading embeddings from:\n  seq:    {args.esmc_dir}\n  struct: {args.saprot_dir}")
    all_ids, seq_map, struct_map, y_map = load_embeddings(args.esmc_dir, args.saprot_dir)

    random.shuffle(all_ids)
    n = len(all_ids)
    n_test = max(1, int(n * args.test_split))
    n_val = max(1, int(n * args.val_split))
    test_ids = all_ids[:n_test]
    val_ids = all_ids[n_test:n_test + n_val]
    train_ids = all_ids[n_test + n_val:]

    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    sample_id = train_ids[0]
    d_seq = seq_map[sample_id].shape[-1]
    d_struct = struct_map[sample_id].shape[-1]
    print(f"  Embedding dims - seq: {d_seq}  struct: {d_struct}")

    def make_loader(ids, shuffle):
        ds = ProteinDataset(
            ids,
            seq_map,
            struct_map,
            y_map,
            graph_type=args.graph_type,
            pdb_dir=args.pdb_dir,
            contact_cutoff=args.contact_cutoff,
            contact_graph_backend=args.contact_graph_backend,
            strict_contact_graphs=args.strict_contact_graphs,
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
        )

    train_loader = make_loader(train_ids, shuffle=True)
    val_loader = make_loader(val_ids, shuffle=False)
    test_loader = make_loader(test_ids, shuffle=False)

    model = SeqStructToptPredictor(d_model=args.d_model, num_heads=args.num_heads)
    if d_seq != args.d_model:
        model.seq_proj = nn.Linear(d_seq, args.d_model)
        print(f"  Added seq_proj: {d_seq} -> {args.d_model}")
    if d_struct != args.d_model:
        model.struct_proj = nn.Linear(d_struct, args.d_model)
        print(f"  Added struct_proj: {d_struct} -> {args.d_model}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_model.pt")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(-1)

            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item() * y.numel()
            n_seen += y.numel()

        scheduler.step()
        train_loss = running / n_seen
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"train_mse={train_loss:.4f} | "
            f"val_mse={val_loss:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "best_val_mse": best_val,
                },
                best_path,
            )
            print(f"  Saved best model (val_mse={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping - no improvement for {args.patience} epochs.")
                break

    print(f"\nBest val MSE: {best_val:.4f}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, criterion)
    print(f"Test | mse={test_mse:.4f} | mae={test_mae:.4f} | rmse={test_rmse:.4f}")


if __name__ == "__main__":
    main()
