import torch
from torch import nn
import os
import sys

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from attention import Attention


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer."""

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads=8,
        dropout: float = 0.1,
    ):
        super(DecoderLayer, self).__init__()

        # Masked self-attention for decoder
        self.masked_self_attention = Attention(d_model=d_model, num_heads=num_heads)

        # Cross-attention to attend to encoder output
        self.cross_attention = Attention(d_model=d_model, num_heads=num_heads)

        # Position-wise feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        # Initialize layer norm
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output: torch.Tensor, decoder_input: torch.Tensor):
        """Process decoder input with attention to encoder output.

        Args:
            encoder_output: Output from encoder (batch_size, src_seq_len, in_channel)
            decoder_input: Target sequence input (batch_size, tgt_seq_len, in_channel)

        Returns:
            decoder_output: Decoded representation
        """
        # 1. Masked Multi-Head Self-Attention
        masked_attention_output = self.masked_self_attention(x=decoder_input, mask=True)
        masked_attention_output = self.dropout(masked_attention_output)
        layernorm_output = self.layernorm1(masked_attention_output + decoder_input)

        # 2. Cross-Attention to encoder output
        # Query from decoder, Key/Value from encoder
        cross_attention_output = self.cross_attention(
            x=encoder_output,  # K, V from encoder
            y=layernorm_output,  # Q from decoder
        )
        cross_attention_output = self.dropout(cross_attention_output)
        layernorm_output = self.layernorm2(cross_attention_output + layernorm_output)

        # 3. Feed-Forward Network with residual connection
        ff_output = self.feedforward(layernorm_output)
        ff_output = self.dropout(ff_output)
        decoder_output = self.layernorm3(ff_output + layernorm_output)

        return decoder_output


class DecoderStack(nn.Module):
    """Transformer Encoder stack with multiple identical layers."""

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads=8,
        dropout: float = 0.1,
    ):
        super(DecoderStack, self).__init__()
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """Process input through encoder layers.

        Args:
            decoder_input: Target sequence (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)

        Returns:
            decoder_output: Decoded representation
        """
        decoder_output = decoder_input
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_input=decoder_output,
                encoder_output=encoder_output,
            )

        return decoder_output


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 512
    num_layers = 2
    num_heads = 2

    # Create dummy input
    encoder_output = torch.rand(batch_size, seq_len, d_model)
    decoder_input = torch.rand(batch_size, seq_len, d_model)

    # Initialize encoder
    decoder = DecoderStack(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Run forward pass
    decoder_output = decoder(decoder_input, encoder_output)

    # Print results
    print(f"Encoder output shape:  {encoder_output.shape}")
    print(f"Decoder input shape:  {decoder_input.shape}")
    print(f"Output shape: {decoder_output.shape}")

    # Quick gradient check
    loss = decoder_output.mean()
    loss.backward()

    print("Backward pass successful — gradients computed.")
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad mean={param.grad.mean():.6f}")
            break  # just show one example
