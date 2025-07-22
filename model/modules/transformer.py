"""Transformer module of transformer."""

import torch
import torch.nn as nn

from model.modules.embeddings import Embeddings
from model.modules.self_attention import SelfAttention
from model.modules.feed_forward import FeedForward


class Transformer(nn.Module):
    """Transformer module of transformer.

    The transformer module is a module that implements the transformer architecture.
    It is used to encode the input tokens and generate the output tokens.

    Args:
        hidden_size (int): The dimension of the hidden state.
        vocab_size (int): The size of the vocabulary.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of transformer layers.
        dropout (float): The dropout rate. Defaults to 0.1.
        is_decoder (bool): Whether the transformer is a decoder. Defaults to True.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        is_decoder: bool = True,
    ):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Embeddings layer
        self.embeddings = Embeddings(hidden_size, vocab_size)

        # Transformer layers
        self.layers = nn.Sequential(
            [
                (
                    SelfAttention(hidden_size, num_heads, dropout, is_decoder),
                    FeedForward(hidden_size, dropout),
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Get the embeddings of the input tokens (B x N) -> (B x N x D)
        embs = self.embeddings(x)

        # Pass the embeddings through the transformer layers (B x N x D) -> (B x N x D)
        out = self.layers(embs)

        # Project the output to the vocabulary size (B x N x V)
        out = self.output_proj(out)

        # Apply the softmax function to the output (B x N x V)
        return self.softmax(out)
