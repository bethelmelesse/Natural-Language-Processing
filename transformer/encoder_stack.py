from torch import nn
import torch
import os
import sys

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from attention import Attention


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = None, num_heads: int = 8):
        super(EncoderLayer, self).__init__()
        hidden_dim = hidden_dim or 4 * d_model

        # Initialize Attention Layer
        self.self_attention = Attention(d_model=d_model, num_heads=num_heads)

        # Create FeedForward Network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Initialize layer norm
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention mechanism
        attn_output = self.self_attention(x=x)
        output = self.layernorm1(attn_output + x)

        # Position-wise feed-forward network
        ff_output = self.feedforward(output)
        output = self.layernorm2(ff_output + output)
        return output


class EncoderStack(nn.Module):
    """Transformer Encoder stack with multiple identical layers."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = None,
        num_layers: int = 6,
        num_heads=8,
    ):
        super(EncoderStack, self).__init__()
        self.num_layers = num_layers

        hidden_dim = hidden_dim or 4 * d_model

        # Multi-head self-attention mechanism
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, hidden_dim=hidden_dim, num_heads=num_heads
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, encoder_input: torch.Tensor):
        """Process input through encoder layers.

        Args:
            encoder_input: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            encoder_output: Encoded representation
        """
        encoder_output = encoder_input
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(x=encoder_output)

        return encoder_output


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 512
    num_layers = 2
    num_heads = 2

    input = torch.rand(batch_size, seq_len, d_model)

    # Create dummy input
    x = torch.rand(batch_size, seq_len, d_model)

    # Initialize encoder
    encoder = EncoderStack(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Run forward pass
    output = encoder(x)

    # Print results
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    # Quick gradient check
    loss = output.mean()
    loss.backward()

    print("Backward pass successful — gradients computed.")
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad mean={param.grad.mean():.6f}")
            break  # just show one example
