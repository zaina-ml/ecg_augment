import torch
import random
from torch import nn

class ECGAugment(nn.Module):
    def __init__(self, transforms, severity=0.5, seed=None):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.severity = severity
        self.seed = seed

    def forward(self, x):
        rng = random.Random(self.seed)

        for t in self.transforms:
            x = t(x, self.severity)
        
        return torch.clamp(x, -5, 5)


class AugmentationScheduler:
    def __init__(self, prog_epochs=10, max_severity=None):
        self.prog_epochs = prog_epochs
        self.max_severity = max_severity

    def get(self, epoch, y):
        base = min(1.0, epoch / max(1, self.prog_epochs))
        severity = base * self.max_severity

        return min(severity, self.max_severity)

