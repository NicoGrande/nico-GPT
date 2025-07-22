"""Feed-forward module of transformer."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward module of transformer.

    The feed-forward module is a module that implements the feed-forward network.
    It is used to compute the feed-forward network of the transformer.

    Args:
        hidden_size (int): The dimension of the hidden state.
        dropout (float): The dropout rate. Defaults to 0.1.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Linear layers for up and down projection
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.activation = nn.GELU()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed-forward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.dropout(self.down_proj(self.activation(self.up_proj(x))))
        return self.layer_norm(out + x)
