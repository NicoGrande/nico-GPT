"""NanoGPT model."""

from dataclasses import dataclass
import torch
import torch.nn as nn

from modules.pytorch.embeddings import Embeddings
from modules.pytorch.transformer import Transformer


@dataclass
class NanoGPTConfig:
    """Configuration for the NanoGPT model.

    Args:
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The dimension of the hidden state.
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout rate.
        is_decoder (bool): Whether the transformer is a decoder.
    """

    vocab_size: int = 50257
    hidden_size: int = 258
    num_layers: int = 6
    num_heads: int = 6
    dropout: float = 0.2
    is_decoder: bool = True


class NanoGPT(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super(NanoGPT, self).__init__()
        self.embeddings = Embeddings(config.hidden_size, config.vocab_size)
        self.transformer = Transformer(
            config.hidden_size,
            config.vocab_size,
            config.num_heads,
            config.num_layers,
            config.dropout,
            config.is_decoder,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.embeddings(x))
