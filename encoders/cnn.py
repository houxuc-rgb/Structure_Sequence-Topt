import torch.nn as nn


class CNNBranch(nn.Module):
    def __init__(self, d_model, num_layers=1, kernel_size=5):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(d_model))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model)
        """
        x = x.transpose(1, 2)  # (B, d_model, seq_len) for Conv1d
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (B, seq_len, d_model)
        return x
