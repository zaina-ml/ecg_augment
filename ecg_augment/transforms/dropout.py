import torch
from torch import nn

class RandomDropout(nn.Module):
    def __init__(self, max_frac=0.05):
        super().__init__()
        self.max_frac = max_frac

    def forward(self, x, severity):
        L = x.shape[-1]
        chunk_len = max(1, int(L * self.max_frac * severity))
        if chunk_len > 0:
            start = int(torch.randint(0, max(1, L - chunk_len), (1,), device=x.device))
            x[start:start + chunk_len] = 0.0
        return x