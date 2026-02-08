import torch
from torch import nn
from einops import einsum, rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        
        # Calculate theta_k = theta ^ (-2(k-1)/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        # Precompute frequencies for all positions up to max_seq_len
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq) # Shape: (max_seq_len, d_k / 2)
        
        # Cache cos and sin values as buffers
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)
        
        # Gather precomputed cos/sin: (..., seq_len, d_k / 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # Construct 2x2 rotation matrices for each pair: (..., seq_len, d_k / 2, 2, 2)
        R = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=-2)
        
        # Use rearrange to split the embedding dimension into pairs of size 2
        x_reshaped = rearrange(x, "... i (m d) -> ... i m d", d=2)
        
        # Rotate: (... batch, i seq_len, m pairs, l row, j col)
        out = einsum(R, x_reshaped, "... i m l j, ... i m j -> ... i m l")
        
        # Flatten back into original shape
        return rearrange(out, "... i m l -> ... i (m l)")
