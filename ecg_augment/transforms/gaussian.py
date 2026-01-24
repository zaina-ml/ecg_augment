import torch
from torch import nn

class AddGaussianNoise(nn.Module):
    def __init__(self, std_factor=0.05):
        super().__init__()
        self.std_factor = std_factor

    def forward(self, x, severity):
        std = torch.std(x) * self.std_factor * severity * 10
        return x + torch.randn_like(x) * std