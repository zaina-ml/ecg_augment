import torch
from torch import nn

class TimeWarp(nn.Module):
    def __init__(self, max_stretch=0.05):
        super().__init__()
        self.max_stretch = max_stretch

    def forward(self, x, severity):
        L = x.shape[-1]

        stretch = 1 + (torch.rand(1, device=x.device) * 2 - 1) * self.max_stretch * severity
        new_len = max(2, int(L * stretch.item()))

        x_in = x.unsqueeze(0).unsqueeze(0)

        warped = F.interpolate(x_in, size=new_len, mode="linear", align_corners=False)
        resampled = F.interpolate(warped, size=L, mode="linear", align_corners=False)

        return resampled.squeeze(0).squeeze(0)