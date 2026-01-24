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

x = torch.randn(12, 500)
aug = ECGAugment([AddGaussianNoise()], seed=42)
out = aug(x)
print(out.shape) # [12, 500]