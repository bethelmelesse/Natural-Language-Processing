import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys, values, mask: bool = None):
        """
        Args:
            query: with shape [batch_size, num_heads, seq_length, dim]
            keys: with shape [batch_size, num_heads, seq_length, dim]
            values: with shape [batch_size, num_heads, seq_length, dim]
        """
        # Transpose key
        # --- Shape: (batch_size, num_heads, dim, seq_length)
        keys = torch.transpose(keys, dim0=2, dim1=3)

        #  Compute compatibility scores ((QK^T))
        # --- Shape: (batch_size, num_heads, seq_length, seq_length)
        qk_score = query @ keys

        # Apply scaling factor
        # --- divide scores by the square root of key dimension
        dk = torch.tensor(keys.shape[2])
        scaled_score = qk_score / torch.sqrt(dk).float()

        # Step 7: Apply mask if indicated
        if mask:
            ones = torch.ones_like(scaled_score)
            masking_tensor = torch.logical_not(torch.tril(input=ones).to(torch.bool))
            scaled_score.masked_fill(masking_tensor, float("-inf"))

        # Apply softmax to get attention weights
        # --- Shape: (batch_size, num_heads, seq_length, seq_length)
        attention_weights = self.softmax(scaled_score)

        # Compute weighted sum of values
        # --- Shape: (batch_size, num_heads, seq_length, dim)
        attention_output = attention_weights @ values

        return attention_output


class Attention(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, num_heads: int = 2):
        super(Attention, self).__init__()

        self.num_heads = num_heads

        self.q_proj = nn.Linear(in_features=in_channel, out_features=out_channel)
        self.k_proj = nn.Linear(in_features=in_channel, out_features=out_channel)
        self.v_proj = nn.Linear(in_features=in_channel, out_features=out_channel)

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.dim = out_channel // num_heads

        self.out_proj = nn.Linear(out_channel, out_channel)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, mask: bool = False
    ) -> torch.Tensor:
        # Check if it is self or cross attention
        y = x if y is None else y

        # Define q, k, v vectors
        # --- Shape: (batch_size, seq_length, dim)
        query = self.q_proj(y)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        # --- Shape: (batch_size, seq_length, dim) -> (batch_size, seq_length, num_heads, dim)
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        query = query.view(batch_size, seq_length, self.num_heads, self.dim)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.dim)
        values = values.view(batch_size, seq_length, self.num_heads, self.dim)

        # Transpose for multi-head attention
        # --- Shape: (seq_length, num_heads , dim) -> (num_heads, seq_length , dim)
        # --- Shape: (batch_size, seq_length, num_heads , dim) -> (batch_size, num_heads, seq_length , dim)
        query = torch.transpose(query, dim0=1, dim1=2)
        keys = torch.transpose(keys, dim0=1, dim1=2)
        values = torch.transpose(values, dim0=1, dim1=2)

        scaled_dot_product_attention_output = self.scaled_dot_product_attention(
            query=query, keys=keys, values=values, mask=mask
        )

        # Test
        official = torch.nn.functional.scaled_dot_product_attention(
            query=query, key=keys, value=values
        )
        assert torch.allclose(scaled_dot_product_attention_output, official)

        # Concatenate the dimensions
        attention_output = scaled_dot_product_attention_output.view(
            batch_size, seq_length, self.num_heads * self.dim
        )

        # Apply project output
        attention_output = self.out_proj(attention_output)

        return attention_output


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    in_dim = 6
    out_dim = 8
    num_heads = 2

    attention_fn = Attention(
        in_channel=in_dim, out_channel=out_dim, num_heads=num_heads
    )

    # Inputs
    x = torch.rand(batch_size, seq_len, in_dim)
    y = torch.rand(batch_size, seq_len, in_dim)

    print("=== Testing Self-Attention ===")
    self_attention_output = attention_fn(x=x)
    print(f"Self-attention output shape: {self_attention_output.shape}\n")

    print("=== Testing Cross-Attention ===")
    cross_attention_output = attention_fn(x=x, y=y)
    print(f"Cross-attention output shape: {cross_attention_output.shape}\n")

    print("=== Testing Masked Self-Attention ===")
    masked_attention_output = attention_fn(x=x, mask=True)
    print(f"Masked self-attention output shape: {masked_attention_output.shape}\n")

    # Gradient check
    loss = self_attention_output.mean()
    loss.backward()
    print("\nBackward pass successful — gradients computed.")
    for name, param in attention_fn.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad mean={param.grad.mean():.6f}")
            break
