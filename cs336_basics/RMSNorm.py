from torch import nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.W = nn.Parameter(torch.ones(d_model, device=device, dtype = dtype))
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(-1, keepdim=True)
        rms = (rms + self.epsilon)**0.5
        result = (x / rms) * self.W
        result = result.to(dtype)
        return result