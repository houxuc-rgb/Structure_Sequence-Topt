from pathlib import Path

import torch

try:
    from Bio.PDB import PDBParser
except ImportError:
    PDBParser = None

try:
    from torch_geometric.nn import radius_graph
except ImportError:
    radius_graph = None


PDB_SUFFIXES = (".pdb", ".ent")
CONTACT_GRAPH_BACKENDS = ("auto", "pyg", "torch")


def find_pdb_file(protein_id, pdb_dir):
    """Return the first matching PDB file for a protein ID, or None."""
    pdb_root = Path(pdb_dir)
    for suffix in PDB_SUFFIXES:
        candidate = pdb_root / f"{protein_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def select_atom_location(atom):
    """Select a stable altloc for a Biopython atom-like object."""
    if not atom.is_disordered():
        return atom

    for altloc in (" ", "A", "1"):
        if altloc in atom.child_dict:
            return atom.child_dict[altloc]

    atoms = atom.disordered_get_list()
    return max(atoms, key=lambda child: child.get_occupancy() or 0.0)


def parse_ca_coordinates(pdb_path):
    """Parse one C-alpha coordinate per residue from the first PDB model."""
    if PDBParser is None:
        raise ImportError(
            "Biopython is required for PDB parsing. Install it with `pip install biopython`."
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No model found in {pdb_path}")

    coords = []
    for chain in model:
        for residue in chain:
            hetero_flag = residue.id[0]
            if hetero_flag != " ":
                continue
            if "CA" not in residue:
                continue

            ca_atom = select_atom_location(residue["CA"])
            coords.append(ca_atom.get_coord().tolist())

    if not coords:
        raise ValueError(f"No C-alpha coordinates found in {pdb_path}")

    return torch.tensor(coords, dtype=torch.float32)


def validate_contact_graph_backend(backend):
    if backend not in CONTACT_GRAPH_BACKENDS:
        raise ValueError(
            f"Unknown contact graph backend '{backend}'. "
            f"Expected one of {CONTACT_GRAPH_BACKENDS}."
        )


def make_undirected(edge_index, num_nodes, add_self_loops=False):
    """Return an undirected edge index with duplicate edges removed."""
    device = edge_index.device

    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index.long(), edge_index.flip(0).long()], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    if add_self_loops:
        self_loops = torch.arange(num_nodes, device=device, dtype=torch.long).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

    edge_keys = edge_index[0] * num_nodes + edge_index[1]
    edge_keys = torch.unique(edge_keys)
    return torch.stack([edge_keys // num_nodes, edge_keys % num_nodes], dim=0)


def build_contact_edge_index_from_coords(coords, cutoff=8.0, backend="auto"):
    """Build a C-alpha contact graph as a PyG-style edge_index tensor."""
    validate_contact_graph_backend(backend)
    coords = coords.float()
    num_nodes = coords.shape[0]

    if backend in ("auto", "pyg") and radius_graph is not None:
        try:
            max_num_neighbors = max(num_nodes, 1)
            edge_index = radius_graph(
                coords,
                r=cutoff,
                batch=None,
                loop=False,
                max_num_neighbors=max_num_neighbors,
            )
            return make_undirected(edge_index, num_nodes)
        except Exception:
            if backend == "pyg":
                raise

    if backend == "pyg":
        raise ImportError(
            "torch_geometric is not available. Install PyTorch Geometric or use "
            "--contact_graph_backend auto/torch."
        )

    distances = torch.cdist(coords, coords)
    edge_index = (distances <= cutoff).nonzero(as_tuple=False).t().contiguous()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return make_undirected(edge_index, num_nodes)


def edge_index_to_dense_adj(edge_index, num_nodes, dtype=torch.float32, device=None):
    """Convert a PyG-style edge_index tensor to a legacy dense adjacency matrix."""
    if device is None:
        device = edge_index.device
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    adj.fill_diagonal_(1.0)
    return adj


def build_contact_adj_from_coords(coords, cutoff=8.0, backend="auto"):
    """Build a row-normalized binary C-alpha contact adjacency matrix."""
    edge_index = build_contact_edge_index_from_coords(
        coords,
        cutoff=cutoff,
        backend=backend,
    )
    adj = edge_index_to_dense_adj(
        edge_index,
        coords.shape[0],
        dtype=torch.float32,
        device=coords.device,
    )
    return normalize_adjacency(adj)


def build_contact_edge_index_for_id(
    protein_id,
    pdb_dir,
    expected_len=None,
    cutoff=8.0,
    backend="auto",
):
    """Build a PyG-style C-alpha contact edge_index for one protein ID."""
    pdb_path = find_pdb_file(protein_id, pdb_dir)
    if pdb_path is None:
        raise FileNotFoundError(f"No PDB file found for {protein_id} in {pdb_dir}")

    coords = parse_ca_coordinates(pdb_path)
    if expected_len is not None and coords.shape[0] != expected_len:
        raise ValueError(
            f"Contact graph length mismatch for {protein_id}: "
            f"PDB C-alpha residues={coords.shape[0]}, embedding length={expected_len}"
        )

    return build_contact_edge_index_from_coords(coords, cutoff=cutoff, backend=backend)


def normalize_adjacency(adj):
    row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj / row_sum


def build_contact_adj_for_id(
    protein_id,
    pdb_dir,
    expected_len=None,
    cutoff=8.0,
    backend="auto",
):
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

    return build_contact_adj_from_coords(coords, cutoff=cutoff, backend=backend)
