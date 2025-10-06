import torch
from torch import nn
import os
import sys

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from transformer.attention import Attention


class TransformerDecoder(nn.Module):
    """Transformer Decoder stack with masked self-attention and cross-attention."""

    def __init__(
        self,
        in_channel: int,
        hidden_channel: int,
        out_channel: int,
        num_layers=6,
        num_heads=8,
    ):
        super(TransformerDecoder, self).__init__()

        # Masked self-attention for decoder
        self.masked_self_attention = Attention(
            in_channel=in_channel,
            out_channel=out_channel,
            num_heads=num_heads,
        )
        # Cross-attention to attend to encoder output
        self.cross_attention = Attention(
            in_channel=in_channel,
            out_channel=out_channel,
            num_heads=num_heads,
        )

        #  Position-wise feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=hidden_channel),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channel, out_features=out_channel),
        )

        self.layernorm1 = nn.LayerNorm(out_channel)
        self.layernorm2 = nn.LayerNorm(out_channel)
        self.layernorm3 = nn.LayerNorm(out_channel)
        self.num_layers = num_layers

    def forward(self, encoder_output: torch.Tensor, decoder_input: torch.Tensor):
        """Process decoder input with attention to encoder output.

        Args:
            encoder_output: Output from encoder (batch_size, src_seq_len, in_channel)
            decoder_input: Target sequence input (batch_size, tgt_seq_len, in_channel)

        Returns:
            decoder_output: Decoded representation
        """
        # Apply each decoder layer sequentiallys
        decoder_output = decoder_input
        for _ in range(self.num_layers):
            # 1. Masked Multi-Head Self-Attention
            masked_attention_output = self.masked_self_attention(
                x=decoder_output, mask=True
            )
            residual_output = masked_attention_output + decoder_output
            layernorm_output = self.layernorm1(residual_output)

            # 2. Cross-Attention to encoder output
            cross_attention_output = self.cross_attention(
                x=encoder_output, y=layernorm_output
            )
            residual_output = cross_attention_output + layernorm_output
            layernorm_output = self.layernorm2(residual_output)

            # 3. Feed-Forward Network with residual connection
            feedforward_output = self.feedforward(layernorm_output)
            residual_output = feedforward_output + layernorm_output
            layernorm_output = self.layernorm3(residual_output)

            # Update output for next layer
            decoder_output = layernorm_output

        return decoder_output
