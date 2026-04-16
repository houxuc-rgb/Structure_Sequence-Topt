import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        """
        x:   (batch_size, num_nodes, d_model)
        adj: (batch_size, num_nodes, num_nodes)  normalized adjacency matrix
        """
        support = self.linear(x)
        out = torch.bmm(adj, support)
        out = self.norm(out + x)  # residual
        return F.relu(out)


class GCNBranch(nn.Module):
    def __init__(self, d_model, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(d_model) for _ in range(num_layers)])

    def forward(self, x, adj):
        """
        x:   (batch_size, num_nodes, d_model)
        adj: (batch_size, num_nodes, num_nodes)
        returns: (batch_size, num_nodes, d_model)
        """
        for layer in self.layers:
            x = layer(x, adj)
        return x
