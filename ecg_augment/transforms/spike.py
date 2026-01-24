import torch
from torch import nn

class RandomSpike(nn.Module):
    def __init__(self, max_amp=0.2):
        super().__init__()
        self.max_amp = max_amp

    def forward(self, x, severity):
        L = x.shape[-1]
        amp = self.max_amp * severity * torch.std(x)
        at = int(torch.randint(0, L, (1,), device=x.device))
        l = min(int(torch.randint(1, 4, (1,), device=x.device)), L - at)
        x[at:at + l] += torch.randn(l, device=x.device) * amp
        return x
