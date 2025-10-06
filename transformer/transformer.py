import torch
from torch import nn
import os
import sys

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from transformer.encoder_stack import TransformerEncoder
from transformer.decoder_stack import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, in_channel: int, hidden_channel: int, out_channel: int):
        """Complete TransoTransformer model with encoder-decoder architecture."""
        super(Transformer, self).__init__()

        # Initialize encoder and decoder stacks
        self.encoder = TransformerEncoder(
            in_channel=in_channel,
            hidden_channel=hidden_channel,
            out_channel=out_channel,
            num_layers=6,
            num_heads=8,
        )
        self.decoder = TransformerDecoder(
            in_channel=in_channel,
            hidden_channel=hidden_channel,
            out_channel=out_channel,
            num_layers=6,
            num_heads=8,
        )
        # Final linear projection to output vocabulary
        self.linear = nn.Linear(in_features=in_channel, out_features=out_channel)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, input_source: torch.Tensor, input_target: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            input_source: Source sequence tensor (e.g., English sentence)
            input_target: Target sequence tensor (e.g., shifted French sentence)

        Returns:
            logits: Output logits before softmax
        """

        # Encode the input sequence
        encoder_output = self.encoder(encoder_input=input_source)

        # Decode using encoder output and decoder input
        decoder_output = self.decoder(
            encoder_output=encoder_output, decoder_input=input_target
        )
        # Project to output vocabulary size
        logits = self.linear(decoder_output)
        return logits
