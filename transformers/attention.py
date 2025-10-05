import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys, values, mask: bool = None):
        """
        Args:
            query: with shape [num_heads, seq_length, dim]
            keys: with shape [num_heads, seq_length, dim]
            values: with shape [num_heads, seq_length, dim]
        """
        # Transpose key
        # --- Shape: (num_heads, dim, seq_length)
        keys = torch.transpose(keys, dim0=2, dim1=1)

        #  Compute compatibility scores ((QK^T))
        # --- Shape: (num_heads, seq_length, seq_length)
        qk_score = torch.bmm(query, keys)

        # Apply scaling factor
        # --- divide scores by the square root of key dimension
        dk = torch.tensor(keys.shape[1])
        scaled_score = qk_score / torch.sqrt(dk).float()

        # Step 7: Apply mask if indicated
        if mask:
            ones = torch.ones_like(scaled_score)
            masking_tensor = torch.logical_not(torch.tril(input=ones).to(torch.bool))
            scaled_score.masked_fill(masking_tensor, float("-inf"))

        # Apply softmax to get attention weights
        # --- Shape: (num_heads, seq_length, seq_length)
        attention_weights = self.softmax(scaled_score)

        # Compute weighted sum of values
        # --- Shape: (num_heads, seq_length, dim)
        attention_output = torch.bmm(attention_weights, values)

        return attention_output


class Attention(nn.Module):
    def __init__(self, input_features: int, output_features: int, num_heads: int = 2):
        super(Attention, self).__init__()

        self.num_heads = num_heads

        self.q_proj = nn.Linear(
            in_features=input_features, out_features=output_features
        )
        self.k_proj = nn.Linear(
            in_features=input_features, out_features=output_features
        )
        self.v_proj = nn.Linear(
            in_features=input_features, out_features=output_features
        )

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.dim = output_features // num_heads

        self.out_proj = nn.Linear(output_features, output_features)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, mask: bool = False
    ) -> torch.Tensor:
        # Check if it is self or cross attention
        y = x if y is None else y

        # Define q, k, v vectors
        # --- Shape: (seq_length, dim)
        query = self.q_proj(y)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        # --- Shape: (seq_length, num_heads, dim)
        seq_length = x.shape[0]
        query = query.view(seq_length, self.num_heads, self.dim)
        keys = keys.view(seq_length, self.num_heads, self.dim)
        values = values.view(seq_length, self.num_heads, self.dim)

        # Transpose for multi-head attention
        # --- Shape: (num_heads, seq_length , dim)
        query = torch.transpose(query, dim0=1, dim1=0)
        keys = torch.transpose(keys, dim0=1, dim1=0)
        values = torch.transpose(values, dim0=1, dim1=0)

        scaled_dot_product_attention_output = self.scaled_dot_product_attention(
            query=query, keys=keys, values=values, mask=mask
        )

        # Test
        official = torch.nn.functional.scaled_dot_product_attention(
            query=query, key=keys, value=values
        )
        # assert scaled_dot_product_attention_output == official

        # Concatenate the dimensions
        attention_output = scaled_dot_product_attention_output.view(
            seq_length, self.num_heads * self.dim
        )

        # Apply project output
        attention_output = self.out_proj(scaled_dot_product_attention_output)

        return attention_output


in_dim = 6
attention_fn = Attention(input_features=in_dim, output_features=4)


x = torch.rand(5, in_dim)  # 5 samples, 3 features each
self_attention_output = attention_fn(x=x)

x = torch.rand(5, in_dim)  # 5 samples, 3 features each
y = torch.rand(5, in_dim)  # 5 samples, 3 features each
cross_attention_output = attention_fn(x=x, y=y)


x = torch.rand(5, in_dim)  # 5 samples, 3 features each
masked_attention_output = attention_fn(x=x, mask=True)
