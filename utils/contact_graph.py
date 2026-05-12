from pathlib import Path

import torch


PDB_SUFFIXES = (".pdb", ".ent")


def find_pdb_file(protein_id, pdb_dir):
    """Return the first matching PDB file for a protein ID, or None."""
    pdb_root = Path(pdb_dir)
    for suffix in PDB_SUFFIXES:
        candidate = pdb_root / f"{protein_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def parse_ca_coordinates(pdb_path):
    """Parse one C-alpha coordinate per residue from the first PDB model."""
    coords = []
    seen_residues = set()

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ENDMDL") and coords:
                break
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            alt_loc = line[16].strip()
            if alt_loc not in ("", "A", "1"):
                continue

            chain_id = line[21].strip()
            residue_id = line[22:26].strip()
            insertion_code = line[26].strip()
            residue_key = (chain_id, residue_id, insertion_code)
            if residue_key in seen_residues:
                continue

            try:
                xyz = [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            except ValueError:
                continue

            seen_residues.add(residue_key)
            coords.append(xyz)

    if not coords:
        raise ValueError(f"No C-alpha coordinates found in {pdb_path}")

    return torch.tensor(coords, dtype=torch.float32)


def build_contact_adj_from_coords(coords, cutoff=8.0):
    """Build a row-normalized binary C-alpha contact adjacency matrix."""
    distances = torch.cdist(coords, coords)
    adj = (distances <= cutoff).to(dtype=torch.float32)
    adj.fill_diagonal_(1.0)
    return normalize_adjacency(adj)


def normalize_adjacency(adj):
    row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj / row_sum


def build_contact_adj_for_id(protein_id, pdb_dir, expected_len=None, cutoff=8.0):
    """Build a normalized C-alpha contact graph for one protein ID."""
    pdb_path = find_pdb_file(protein_id, pdb_dir)
    if pdb_path is None:
        raise FileNotFoundError(f"No PDB file found for {protein_id} in {pdb_dir}")

    coords = parse_ca_coordinates(pdb_path)
    if expected_len is not None and coords.shape[0] != expected_len:
        raise ValueError(
            f"Contact graph length mismatch for {protein_id}: "
            f"PDB C-alpha residues={coords.shape[0]}, embedding length={expected_len}"
        )

    return build_contact_adj_from_coords(coords, cutoff=cutoff)