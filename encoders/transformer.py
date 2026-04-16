import torch.nn as nn


class TransformerBranch(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=1, ff_dim=None):
        super().__init__()
        if ff_dim is None:
            ff_dim = d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model)
        """
        return self.encoder(x, src_key_padding_mask=mask)
