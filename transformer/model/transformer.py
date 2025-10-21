import torch
from torch import nn

from transformer.model.encoder_stack import EncoderStack
from transformer.model.decoder_stack import DecoderStack
from transformer.model.embedding import InputEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 512,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Complete TransoTransformer model with encoder-decoder architecture."""
        super(Transformer, self).__init__()

        d_ff = d_ff or 4 * d_model

        # Embeddings
        self.source_embedding = InputEmbedding(
            vocab_size=source_vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.target_embedding = InputEmbedding(
            vocab_size=target_vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Initialize encoder and decoder stacks
        self.encoder = EncoderStack(
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = DecoderStack(
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Final linear projection to output vocabulary
        self.linear = nn.Linear(in_features=d_model, out_features=target_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights (important for training stability)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor = None,
        target_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            source: Source sequence token indices (batch_size, source_seq_len)
            target: Target sequence token indices (batch_size, target_seq_len)
            source_mask: Source padding mask (batch_size, source_seq_len)
            target_mask: Target padding mask (batch_size, target_seq_len)

        Returns:
            logits: Output logits (batch_size, target_seq_len, target_vocab_size)
        """
        # Embed inputs
        source_embed = self.source_embedding(input_ids=source)
        target_embed = self.target_embedding(input_ids=target)

        # Encode source sequence
        encoder_output = self.encoder(
            encoder_input=source_embed, source_mask=source_mask
        )

        # Decode target sequence
        decoder_output = self.decoder(
            encoder_output=encoder_output,
            decoder_input=target_embed,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        # Project to output vocabulary size
        logits = self.linear(decoder_output)

        return logits


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    source_seq_size = 200
    target_seq_size = 250

    # Create dummy input
    source = torch.rand(batch_size, source_seq_size).long()
    target = torch.rand(batch_size, target_seq_size).long()

    # Initialize encoder
    model = Transformer(
        source_vocab_size=source_seq_size, target_vocab_size=target_seq_size
    )

    # Run forward pass
    logits = model(source=source, target=target)

    # Print results
    print(f"Source input shape:  {source.shape}")
    print(f"Target input shape:  {target.shape}")
    print(f"Output shape: {logits.shape}")

    # Quick gradient check
    loss = logits.mean()
    loss.backward()

    print("Backward pass successful — gradients computed.")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad mean={param.grad.mean():.6f}")
            break  # just show one example
