import torch
from torch import nn

class RandomShift(nn.Module):
    def __init__(self, max_shift=0.05):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x, severity):
        L = x.shape[-1]
        max_shift = int(L * self.max_shift * severity)
        if max_shift > 0:
            k = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device))
            x = torch.roll(x, shifts=k, dims=-1)
        return x