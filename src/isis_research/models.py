"""Torch model constructors for the CSA-to-NASA image translator.

Lifted verbatim from ``scripts/benchmark_signal_detectors.py``.  ``torch_models``
keeps the original constructor tuple for compatibility; ``model_constructor``
is the named interface used by current training and inference code.
"""

from __future__ import annotations

import numpy as np


MODEL_INDEX = {
    "line_cnn": 0,
    "cnn_2d": 1,
    "unet": 2,
    "wide_unet": 3,
    "residual_unet": 4,
    "norm_residual_unet": 5,
    "dilated_cnn": 6,
    "hybrid_unet": 7,
}


def model_constructor(name):
    """Return a named model constructor without exposing tuple indexes."""
    if name == "coord_unet":
        name = "unet"
    try:
        index = MODEL_INDEX[name]
    except KeyError as error:
        raise ValueError(f"unsupported model: {name!r}") from error
    return torch_models()[index]


def image_features(signal, channels=1):
    """Build model input channels, optionally including normalized grid position."""
    signal = np.asarray(signal, dtype=np.float32)
    if channels == 1:
        return signal[None]
    if channels != 3:
        raise ValueError(f"unsupported input channel count: {channels}")
    height, width = signal.shape
    vertical = np.broadcast_to(
        np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None], signal.shape
    )
    horizontal = np.broadcast_to(
        np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :], signal.shape
    )
    return np.stack((signal, vertical, horizontal))


def torch_models():
    import torch
    import torch.nn as nn

    class LineCNN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, 12, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(12, 12, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(12, 1, 1),
            )

        def forward(self, values):
            batch, channels, height, width = values.shape
            lines = values.permute(0, 3, 1, 2).reshape(batch * width, channels, height)
            output = self.net(lines).reshape(batch, width, height)
            return output.permute(0, 2, 1)

    class ScanCNN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, 12, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 12, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 1, 1),
            )

        def forward(self, values):
            return self.net(values)[:, 0]

    class TinyUNet(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.left = nn.Sequential(
                nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            )
            self.down = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
            self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.out = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1)
            )

        def forward(self, values):
            left = self.left(values)
            middle = self.down(left)
            up = self.up(middle)
            return self.out(torch.cat([up, left], dim=1))[:, 0]

    class WideUNet(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.left = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            )
            self.down = nn.Sequential(
                nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            )
            self.middle = nn.Sequential(
                nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            )
            self.up_down = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.up_left = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.decode_down = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            )
            self.decode_left = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            )
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, values):
            left = self.left(values)
            down = self.down(left)
            middle = self.middle(down)
            up_down = self.up_down(middle)
            decoded_down = self.decode_down(torch.cat([up_down, down], dim=1))
            up_left = self.up_left(decoded_down)
            decoded_left = self.decode_left(torch.cat([up_left, left], dim=1))
            return self.out(decoded_left)[:, 0]

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )

        def forward(self, values):
            return nn.functional.relu(values + self.body(values))

    class ResidualUNet(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.input = nn.Conv2d(channels, 32, 3, padding=1)
            self.left = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
            self.down = nn.Sequential(
                nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                ResidualBlock(64), ResidualBlock(64),
            )
            self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.decode = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                ResidualBlock(32),
            )
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, values):
            left = self.left(self.input(values))
            down = self.down(left)
            up = self.up(down)
            return self.out(self.decode(torch.cat([up, left], dim=1)))[:, 0]

    class NormalizedResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )

        def forward(self, values):
            return nn.functional.relu(values + self.body(values))

    class NormalizedResidualUNet(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.input = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU()
            )
            self.left = nn.Sequential(
                NormalizedResidualBlock(32), NormalizedResidualBlock(32)
            )
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                NormalizedResidualBlock(64),
                NormalizedResidualBlock(64),
            )
            self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.decode = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                NormalizedResidualBlock(32),
            )
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, values):
            left = self.left(self.input(values))
            down = self.down(left)
            up = self.up(down)
            return self.out(self.decode(torch.cat([up, left], dim=1)))[:, 0]

    class DilatedCNN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=2, dilation=2),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=4, dilation=4),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1),
            )

        def forward(self, values):
            return self.net(values)[:, 0]

    class HybridUNet(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.left = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(), ResidualBlock(32)
            )
            self.down = nn.Sequential(
                nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), ResidualBlock(64)
            )
            self.middle = nn.Sequential(
                nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), ResidualBlock(128)
            )
            self.up_down = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.decode_down = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), ResidualBlock(64)
            )
            self.up_left = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.decode_left = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), ResidualBlock(32)
            )
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, values):
            left = self.left(values)
            down = self.down(left)
            middle = self.middle(down)
            up_down = self.up_down(middle)
            decoded_down = self.decode_down(torch.cat([up_down, down], dim=1))
            up_left = self.up_left(decoded_down)
            decoded_left = self.decode_left(torch.cat([up_left, left], dim=1))
            return self.out(decoded_left)[:, 0]

    return (
        LineCNN,
        ScanCNN,
        TinyUNet,
        WideUNet,
        ResidualUNet,
        NormalizedResidualUNet,
        DilatedCNN,
        HybridUNet,
    )
