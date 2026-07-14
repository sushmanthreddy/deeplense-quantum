"""D4 orbit lifting and the selected shared Model-I image encoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def norm2d(channels: int) -> nn.GroupNorm:
    """View-local normalization with identical train/evaluation behavior."""

    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def d4_transform(
    images: torch.Tensor, rotation: int, reflected: int
) -> torch.Tensor:
    """Apply ``r^rotation s^reflected`` to a batch of square images."""

    if reflected:
        images = torch.flip(images, dims=(-1,))
    return torch.rot90(images, rotation, dims=(-2, -1))


def d4_views(images: torch.Tensor) -> torch.Tensor:
    """Lift images to all eight D4 views in regular-representation order."""

    return torch.stack(
        [d4_transform(images, k, f) for f in (0, 1) for k in range(4)],
        dim=1,
    )


class MorphologyChannelBank(nn.Module):
    """Eight deterministic morphology channels for photon-count images.

    This is zero-parameter feature engineering, not a PINN.  No differential
    equation residual, lens inversion, or auxiliary loss is optimized.
    """

    output_channels = 8

    def __init__(
        self,
        log_gain: float = 20.0,
        epsilon: float = 1e-3,
        reference_pixels: int = 96,
    ) -> None:
        super().__init__()
        self.log_gain = float(log_gain)
        self.epsilon = float(epsilon)
        self.reference_pixels = int(reference_pixels)
        self.variant = "base"

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.transpose(0, 1).contiguous()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        )
        self.register_buffer(
            "sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "laplacian", laplacian.view(1, 1, 3, 3), persistent=False
        )

    @staticmethod
    def _conv_reflect(images: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(images, (1, 1, 1, 1), mode="reflect"), kernel)

    @staticmethod
    def _avg_reflect(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        return F.avg_pool2d(
            F.pad(images, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = torch.nan_to_num(
            images.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        scale = images.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        images = (images / scale).clamp(0.0, 1.0)
        log_intensity = torch.log1p(self.log_gain * images) / math.log1p(
            self.log_gain
        )

        gx = self._conv_reflect(log_intensity, self.sobel_x)
        gy = self._conv_reflect(log_intensity, self.sobel_y)
        pixel_scale = float(images.shape[-1]) / self.reference_pixels
        gradient = torch.sqrt(gx.square() + gy.square() + 1e-8) * pixel_scale
        laplacian = (
            self._conv_reflect(log_intensity, self.laplacian).abs()
            * pixel_scale**2
        )
        small_kernel = max(3, int(round(3 * pixel_scale)) | 1)
        large_kernel = max(small_kernel + 2, int(round(9 * pixel_scale)) | 1)
        dog = (
            self._avg_reflect(log_intensity, small_kernel)
            - self._avg_reflect(log_intensity, large_kernel)
        ).abs()

        height, width = images.shape[-2:]
        yy = torch.linspace(
            -1.0, 1.0, height, device=images.device, dtype=images.dtype
        )
        xx = torch.linspace(
            -1.0, 1.0, width, device=images.device, dtype=images.dtype
        )
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square()).clamp_min(1e-4)
        unit_x = (grid_x / radius).view(1, 1, height, width)
        unit_y = (grid_y / radius).view(1, 1, height, width)
        radial = (gx * unit_x + gy * unit_y).abs() * pixel_scale
        tangential = (-gx * unit_y + gy * unit_x).abs() * pixel_scale

        # The final stabilized distortion channel is retained from the selected
        # run.  It is simply a fixed mixed finite derivative.
        log_ratio_sq = torch.log(
            (1.0 + self.epsilon) / (images + self.epsilon)
        ).square()
        mixed = self._conv_reflect(
            self._conv_reflect(log_ratio_sq, self.sobel_x), self.sobel_y
        ).abs() * pixel_scale**2

        return torch.cat(
            (
                images,
                log_intensity,
                torch.tanh(2.0 * gradient),
                torch.tanh(laplacian),
                torch.tanh(4.0 * dog),
                torch.tanh(2.0 * radial),
                torch.tanh(2.0 * tangential),
                torch.tanh(mixed),
            ),
            dim=1,
        )


# Kept as a small source-compatibility alias for old imports.  The clearer
# class name above describes what the module actually does.
PhysicsChannelBank = MorphologyChannelBank


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.net(features)


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand: int,
        stride: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        hidden = in_channels * expand
        layers: list[nn.Module] = []
        if hidden != in_channels:
            layers.extend(
                (
                    nn.Conv2d(in_channels, hidden, 1, bias=False),
                    norm2d(hidden),
                    nn.SiLU(inplace=True),
                )
            )
        layers.extend(
            (
                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=hidden,
                    bias=False,
                ),
                norm2d(hidden),
                nn.SiLU(inplace=True),
                SqueezeExcite(hidden),
                nn.Conv2d(hidden, out_channels, 1, bias=False),
                norm2d(out_channels),
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.block(features)
        return features + output if self.use_residual else output


class CompactOrbitEncoder(nn.Module):
    """The selected tiny MBConv encoder, shared across all eight D4 views."""

    output_dim = 128
    variant = "tiny"

    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        specs = (
            (16, 24, 2, 2),
            (24, 24, 2, 1),
            (24, 40, 3, 2),
            (40, 40, 3, 1),
            (40, 64, 3, 2),
            (64, 64, 3, 1),
            (64, 96, 3, 2),
            (96, 96, 2, 1),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, 16, 5, stride=2, padding=2, bias=False
            ),
            norm2d(16),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *(MBConv(cin, cout, expand, stride) for cin, cout, expand, stride in specs)
        )
        self.final = nn.Sequential(
            nn.Conv2d(96, self.output_dim, 1, bias=False),
            norm2d(self.output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.final(self.blocks(self.stem(images)))
        return features.mean(dim=(-2, -1))
