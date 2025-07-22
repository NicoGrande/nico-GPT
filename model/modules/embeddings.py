"""This file contains the implementation of the embeddings layer of transformer."""

import torch
import torch.nn as nn
from enum import Enum


class PositionalEncodingType(Enum):
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"


class Embeddings(nn.Module):
    """Embeddings layer of transformer.

    The embeddings layer is a simple embedding layer that embeds the input tokens
    into a vector of dimension hidden_size. It also adds positional encoding to the embeddings.

    Args:
        hidden_size (int): The dimension of the hidden state.
        vocab_size (int): The size of the vocabulary.
        pos_embedding_type (PositionalEncodingType): The type of positional encoding.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        pos_embedding_type: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL,
    ):
        super(Embeddings, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        if pos_embedding_type == PositionalEncodingType.SINUSOIDAL:
            self.pos_embedding = self.get_sinusoidal_encoding(vocab_size, hidden_size)
        elif pos_embedding_type == PositionalEncodingType.LEARNED:
            self.pos_embedding = nn.Parameter(torch.randn(1, vocab_size, hidden_size))

    def get_sinusoidal_encoding(
        self, max_length: int, hidden_size: int
    ) -> torch.Tensor:
        """Get the sinusoidal encoding.

        Args:
            max_length (int): The maximum length of the sequence.
            hidden_size (int): The dimension of the hidden state.

        Returns:
            torch.Tensor: The sinusoidal encoding.
        """
        position = torch.arange(max_length).unsqueeze(1)
        i = torch.arange(hidden_size).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / hidden_size)
        angle_rads = position * angle_rates
        encoding = torch.zeros(max_length, hidden_size)
        encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embeddings layer.

        Args:
            x (torch.Tensor): The input tokens.

        Returns:
            torch.Tensor: The embeddings of the input tokens.
        """
        # Get the token embeddings (B x N x D)
        token_embeddings = self.embedding(x)

        # Get the positional embeddings (1 x N x D)
        pos_embeddings = self.pos_embedding[: x.size(1), :].unsqueeze(0)

        # Return the layer normalized embeddings
        return self.layer_norm(token_embeddings + pos_embeddings)
