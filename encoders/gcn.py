import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None


class GCNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        if GCNConv is None:
            raise ImportError(
                "GCNBranch now uses torch_geometric.nn.GCNConv. "
                "Install PyTorch Geometric before constructing the model."
            )
        self.conv = GCNConv(d_model, d_model, add_self_loops=True, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index):
        """
        x:          (total_nodes, d_model)
        edge_index: (2, total_edges)
        """
        out = self.conv(x, edge_index)
        out = self.norm(out + x)
        return F.relu(out)


class GCNBranch(nn.Module):
    def __init__(self, d_model, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(d_model) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        """
        x:          (total_nodes, d_model)
        edge_index: (2, total_edges)
        returns:    (total_nodes, d_model)
        """
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
