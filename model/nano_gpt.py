"""NanoGPT model."""

from dataclasses import dataclass
import torch
import torch.nn as nn

from model.modules.embeddings import Embeddings
from model.modules.transformer import Transformer


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


@dataclass
class TrainingConfig:
    """Configuration for the training of the NanoGPT model.

    Args:
        batch_size (int): The batch size.
        seq_len (int): The sequence length.
        max_steps (int): The maximum number of steps.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        device (str): The device to use.
        eval_interval (int): The interval to evaluate the model.
        save_interval (int): The interval to save the model.
    """

    batch_size: int = 128
    seq_len: int = 1024
    max_steps: int = 10000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    device: str = "cuda"
    eval_interval: int = 100
    save_interval: int = 1000
    save_dir: str = "checkpoints"
    save_name: str = "nano-gpt"
    save_format: str = "pt"


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
