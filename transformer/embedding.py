import torch
from torch import nn


class InputEmbedding(nn.Module):
    """Input embedding with learned positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        seq_len = input_ids.size(1)

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position IDs: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)  # (1, seq_len)

        # Positional embeddings
        pos_embeds = self.positional_embedding(position_ids)

        # Combine
        embeddings = token_embeds + pos_embeds

        return embeddings
