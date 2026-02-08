from torch import nn
import torch

class SwiGLU(nn.Module):
    def __init__(self, d_model : int, d_ff : int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W3 = nn.Parameter(torch.empty(d_model, d_ff))

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.silu(x @ self.W1) * (x @ self.W3)) @ self.W2