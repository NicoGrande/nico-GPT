"""Self-attention module of transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention module of transformer.

    The self-attention module is a module that implements the self-attention mechanism.
    It is used to compute the attention between the input tokens.

    Args:
        hidden_size (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = True,
    ):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Linear layers for key, query, and value
        self.k = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        # Head dim for MHA
        if hidden_size % num_heads > 0:
            raise ValueError(
                "Hidden size should be divisible by number of attention heads."
            )

        self.head_dim = hidden_size // num_heads

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Layer normalization and linear for FFN
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Set decoder flag for auto-regressive forward
        self.is_decoder = is_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the self-attention module.

        Args:
            x (torch.Tensor): The input tokens.

        Returns:
            torch.Tensor: The output of the MHA layer.
        """
        batch_size, seq_len, _ = x.shape

        # Project the input tokens to key, query, and value
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        # Split the key, query, and value into heads (B x H x N x D_h)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the attention matrix (B x H x N x D_h) * (B x H x D_h x N) -> (B x H x N x N)
        attn_matrix = q @ k.transpose(-2, -1) / self.head_dim**0.5

        if self.is_decoder:
            # Mask of dimension (1, 1, N, N)
            mask = (
                torch.tril(torch.ones(seq_len, seq_len, device=x.device))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            attn_matrix = attn_matrix.masked_fill(mask == 0, float("-inf"))

        # Compute attention logits
        attn_logits = self.dropout(F.softmax(attn_matrix, dim=-1))

        # Compute attention output (B x H x N x N) * (B x H x N x D_h) -> (B x H x D_h x N) -> (B x N x D)
        attn_out = attn_logits @ v
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        # Project the attention output to the hidden size (B x N x D)
        attn_out = self.dropout(self.out_proj(attn_out))

        # Residual connection
        return self.layer_norm(attn_out + x)
