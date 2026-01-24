import torch
from torch import nn

class AmplitudeScale(nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x, severity):
        factor = 1 + self.scale * (2 * torch.rand(1, device=x.device) - 1) * severity
        return x * factor