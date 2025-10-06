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
    def __init__(self, d_model: int, num_heads: int = 8):
        super(Attention, self).__init__()

        self.num_heads = num_heads

        self.q_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_proj = nn.Linear(in_features=d_model, out_features=d_model)

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.dim = d_model // num_heads

        self.out_proj = nn.Linear(in_features=d_model, out_features=d_model)

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
        query_seq_length = query.shape[1]
        kv_seq_length = x.shape[1]

        query = query.view(batch_size, query_seq_length, self.num_heads, self.dim)
        keys = keys.view(batch_size, kv_seq_length, self.num_heads, self.dim)
        values = values.view(batch_size, kv_seq_length, self.num_heads, self.dim)

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
        # assert torch.allclose(scaled_dot_product_attention_output, official)

        # Concatenate the dimensions
        attention_output = scaled_dot_product_attention_output.view(
            batch_size, query_seq_length, self.num_heads * self.dim
        )

        # Apply project output
        attention_output = self.out_proj(attention_output)

        return attention_output


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 2

    # Initialize attention layer
    attention_fn = Attention(d_model=d_model, num_heads=num_heads)

    # Create dummy inputs
    x = torch.rand(batch_size, seq_len, d_model)
    y = torch.rand(batch_size, seq_len, d_model)

    print("=== Testing Self-Attention ===")
    self_attention_output = attention_fn(x=x)
    print(f"Self-attention output shape: {self_attention_output.shape}\n")

    print("=== Testing Cross-Attention ===")
    cross_attention_output = attention_fn(x=x, y=y)
    print(f"Cross-attention output shape: {cross_attention_output.shape}\n")

    print("=== Testing Masked Self-Attention ===")
    masked_attention_output = attention_fn(x=x, mask=True)
    print(f"Masked self-attention output shape: {masked_attention_output.shape}\n")

    # Quick sanity checks
    assert self_attention_output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Unexpected self-attention shape: {self_attention_output.shape}"
    assert cross_attention_output.shape == self_attention_output.shape, (
        "Cross-attention output shape mismatch"
    )
    assert masked_attention_output.shape == self_attention_output.shape, (
        "Masked attention output shape mismatch"
    )

    # Gradient check
    loss = self_attention_output.mean()
    loss.backward()
    print("Backward pass successful — gradients computed.")
    for name, param in attention_fn.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad mean={param.grad.mean():.6f}")
            break
