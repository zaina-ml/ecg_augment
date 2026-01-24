from ecg_augment import (
    ECGAugment,
    AugmentationScheduler,
    AddGaussianNoise,
    AmplitudeScale,
    TimeWarp,
    RandomShift,
    RandomDropout,
    RandomSpike,
)

import torch

import matplotlib.pyplot as plt

x = torch.sin(torch.linspace(0, 2 * torch.pi, 500))
aug = ECGAugment([RandomShift(), AddGaussianNoise(), RandomDropout(), RandomSpike()], severity=0.2, seed=42)
out = aug(x)

plt.plot(x, label='Original')
plt.plot(out.detach(), label='Augmented')
plt.legend()

plt.show()