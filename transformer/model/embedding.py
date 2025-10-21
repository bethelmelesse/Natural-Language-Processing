import torch
from torch import nn


class InputEmbedding(nn.Module):
    """Input embedding with learned positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

        # Learned positional embedding
        self.positional_embedding = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=d_model
        )

        #  Scaling factor
        self.scale = d_model**0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings (batch_size, seq_len, d_model)
        token_embeds = self.token_embedding(input_ids) * self.scale

        # Position IDs: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)  # (1, seq_len)

        # Positional embeddings
        pos_embeds = self.positional_embedding(position_ids)

        # Combine token embeddings and positional embedding
        embeddings = self.dropout(token_embeds + pos_embeds)

        return embeddings


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    seq_length = 200
    vocab_size = 1000  # Vocabulary size should be separate from seq_length

    # Create dummy input with valid token IDs in range [0, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    input_embeddings = InputEmbedding(vocab_size=vocab_size, max_seq_len=seq_length)
    embedding = input_embeddings(input_ids=input_ids)

    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {embedding.shape}")
    print(
        f"Expected output shape: (batch_size={batch_size}, seq_len={seq_length}, d_model=512)"
    )
