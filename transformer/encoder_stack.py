from torch import nn
import torch
import os
import sys

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from transformer.attention import Attention


class TransformerEncoder(nn.Module):
    """Transformer Encoder stack with multiple identical layers."""

    def __init__(
        self,
        in_channel: int,
        hidden_channel: int,
        out_channel: int,
        num_layers: int = 6,
        num_heads=8,
    ):
        super(TransformerEncoder, self).__init__()

        # Multi-head self-attention mechanism
        self.self_attention = Attention(
            in_channel=in_channel, out_channel=out_channel, num_heads=num_heads
        )
        # Position-wise feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=hidden_channel),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channel, out_features=out_channel),
        )
        # Initialize layer norm
        self.layernorm1 = nn.LayerNorm(out_channel)
        self.layernorm2 = nn.LayerNorm(out_channel)
        self.num_layers = num_layers

    def forward(self, encoder_input: torch.Tensor):
        """Process input through encoder layers.

        Args:
            encoder_input: Input tensor of shape (batch_size, seq_len, in_channel)

        Returns:
            encoder_output: Encoded representation
        """
        encoder_output = encoder_input

        # Apply each encoder layer sequentially
        for _ in range(self.num_layers):
            # 1. Multi-Head Self-Attention with residual connection and Layer Norm
            attention_output = self.self_attention(x=encoder_output)
            residual_output = attention_output + encoder_output
            layernorm_output = self.layernorm1(residual_output)

            # 2. Feed-Forward Network with residual connection
            feedforward_output = self.feedforward(layernorm_output)
            residual_output = feedforward_output + layernorm_output
            layernorm_output = self.layernorm2(residual_output)

            # Update output for next layer
            encoder_output = layernorm_output

        return encoder_output
