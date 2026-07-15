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


class ForegroundSuppressedMorphologyChannelBank(MorphologyChannelBank):
    """Single-image Model-IV foreground suppression and SIS closure maps."""

    output_channels = 8

    def __init__(
        self,
        log_gain: float = 20.0,
        epsilon: float = 1e-3,
        reference_pixels: int = 96,
    ) -> None:
        super().__init__(
            log_gain=log_gain,
            epsilon=epsilon,
            reference_pixels=reference_pixels,
        )
        self.variant = "model_iv_sis_closure"
        axis = torch.tensor([-1.0, -2.0, 0.0, 2.0, 1.0])
        self.register_buffer(
            "mixed_derivative",
            torch.outer(axis, axis).view(1, 1, 5, 5) / 64.0,
            persistent=False,
        )

    @staticmethod
    def _geometry(
        height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if height != width:
            raise ValueError("D4 morphology requires square images")
        coordinates = torch.arange(height, device=device, dtype=torch.float32)
        coordinates = coordinates - (height - 1) / 2.0
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        return radius, torch.floor(radius).long()

    @staticmethod
    def _border_location_scale(
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if min(images.shape[-2:]) < 16:
            raise ValueError(
                "Model-IV border estimation requires at least 16 pixels"
            )
        border = torch.cat(
            (
                images[..., :8, :].flatten(-2),
                images[..., -8:, :].flatten(-2),
                images[..., :, :8].flatten(-2),
                images[..., :, -8:].flatten(-2),
            ),
            dim=-1,
        )
        location = border.median(-1, keepdim=True).values
        mad = (border - location).abs().median(-1, keepdim=True).values
        return location[..., None], (1.4826 * mad).clamp_min(1e-7)[..., None]

    @staticmethod
    def _smooth_profile(
        profile: torch.Tensor, sigma: float = 0.8
    ) -> torch.Tensor:
        radius = max(1, int(4.0 * sigma + 0.5))
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=profile.device,
            dtype=profile.dtype,
        )
        kernel = torch.exp(-0.5 * (offsets / sigma).square())
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        flat = profile.reshape(-1, 1, profile.shape[-1])
        return F.conv1d(
            F.pad(flat, (radius, radius), mode="replicate"), kernel
        ).reshape_as(profile)

    def _radial_median(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        _, radial_bin = self._geometry(height, width, features.device)
        flat = features.flatten(2)
        profile = torch.stack(
            [
                flat[..., radial_bin.flatten() == index].median(-1).values
                for index in range(int(radial_bin.max()) + 1)
            ],
            dim=-1,
        )
        profile = self._smooth_profile(profile)
        index = radial_bin.flatten().view(1, 1, -1).expand(
            batch, channels, -1
        )
        return profile.gather(2, index).reshape_as(features)

    @staticmethod
    def _normalize_map(
        features: torch.Tensor,
        radius: torch.Tensor,
        center_mask: bool = False,
    ) -> torch.Tensor:
        scale64 = features.shape[-1] / 64.0
        if center_mask:
            mask = (radius >= 3.0 * scale64) & (radius < 43.0 * scale64)
            values = features[..., mask].abs()
        else:
            values = features.flatten(2).abs()
        kth = max(1, int(math.ceil(0.99 * values.shape[-1])))
        scale = values.kthvalue(kth, dim=-1).values[..., None, None] + 1e-7
        output = (features / scale).clamp(-8.0, 8.0)
        if center_mask:
            output = output.masked_fill(
                (radius < 3.0 * scale64).view(1, 1, *radius.shape), 0.0
            )
        return output

    @staticmethod
    def _gaussian_filter(
        features: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        radius = max(1, int(4.0 * sigma + 0.5))
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=features.device,
            dtype=features.dtype,
        )
        kernel = torch.exp(-0.5 * (offsets / sigma).square())
        kernel = kernel / kernel.sum()
        channels = features.shape[1]
        kernel_x = kernel.view(1, 1, 1, -1).expand(
            channels, 1, 1, -1
        )
        kernel_y = kernel.view(1, 1, -1, 1).expand(
            channels, 1, -1, 1
        )
        output = F.conv2d(
            F.pad(features, (radius, radius, 0, 0), mode="reflect"),
            kernel_x,
            groups=channels,
        )
        return F.conv2d(
            F.pad(output, (0, 0, radius, radius), mode="reflect"),
            kernel_y,
            groups=channels,
        )

    @staticmethod
    def _estimate_einstein_radius(
        brightness: torch.Tensor, radius: torch.Tensor
    ) -> torch.Tensor:
        """Estimate one detached smooth-ring radius per supplied image.

        A soft radial peak is substantially cheaper than optimizing a lens
        model inside every training step.  The operation is label-free and
        depends on no other image in the batch.
        """

        scale = brightness.shape[-1] / 64.0
        candidates = torch.arange(
            7,
            28,
            device=brightness.device,
            dtype=brightness.dtype,
        ) * scale
        profiles = []
        positive = brightness.clamp_min(0.0)
        for candidate in candidates:
            mask = (radius >= candidate - 0.5 * scale) & (
                radius < candidate + 0.5 * scale
            )
            profiles.append(positive[..., mask].mean(-1))
        profile = torch.stack(profiles, dim=-1)
        profile = profile / profile.amax(-1, keepdim=True).clamp_min(1e-7)
        weights = torch.softmax(12.0 * profile, dim=-1)
        estimate = (weights * candidates.view(1, 1, -1)).sum(-1)
        return estimate[..., None, None].detach()

    @staticmethod
    def _sis_partner(
        features: torch.Tensor,
        einstein_radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the conjugate SIS point from the same individual image."""

        _, _, height, width = features.shape
        if height != width:
            raise ValueError("SIS closure requires square images")
        axis = torch.arange(
            height, device=features.device, dtype=features.dtype
        ) - (height - 1) / 2.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square()).clamp_min(1e-4)
        factor = 1.0 - 2.0 * einstein_radius / radius.view(
            1, 1, height, width
        )
        partner_x = factor * xx.view(1, 1, height, width)
        partner_y = factor * yy.view(1, 1, height, width)
        grid = torch.stack(
            (
                2.0 * partner_x[:, 0] / max(width - 1, 1),
                2.0 * partner_y[:, 0] / max(height - 1, 1),
            ),
            dim=-1,
        )
        valid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
        partner = F.grid_sample(
            features,
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=True,
        )
        return partner, valid[:, None]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=images.device.type, enabled=False):
            images = torch.nan_to_num(
                images.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            background, noise = self._border_location_scale(images)
            signal = images - background
            peak = signal.amax((-2, -1), keepdim=True).clamp_min(1e-7)
            current = (signal / peak).clamp(-1.0, 1.0)
            radius, _ = self._geometry(
                images.shape[-2], images.shape[-1], images.device
            )
            asinh_unscaled = torch.asinh(
                signal / (3.0 * noise).clamp_min(1e-6)
            )
            asinh_map = self._normalize_map(asinh_unscaled, radius)
            radial = self._normalize_map(
                signal - self._radial_median(signal), radius, True
            )
            asinh_radial = self._normalize_map(
                asinh_unscaled - self._radial_median(asinh_unscaled),
                radius,
                True,
            )
            size_scale = images.shape[-1] / 64.0
            smooth = tuple(
                self._gaussian_filter(asinh_unscaled, sigma * size_scale)
                for sigma in (0.8, 1.6, 3.2, 6.4)
            )
            dog_stack = torch.cat(
                tuple(
                    first - second
                    for first, second in zip(smooth[:-1], smooth[1:])
                ),
                dim=1,
            )
            einstein_radius = self._estimate_einstein_radius(
                asinh_map, radius
            )
            partner_stack, valid = self._sis_partner(
                torch.cat((dog_stack, asinh_unscaled), dim=1),
                einstein_radius,
            )
            partner_dogs = partner_stack[:, :3]
            partner_brightness = partner_stack[:, 3:4]
            union_brightness = (
                asinh_unscaled.clamp_min(0.0)
                + partner_brightness.clamp_min(0.0)
            )
            gate_scale = torch.quantile(
                union_brightness.flatten(2), 0.95, dim=-1
            )[..., None, None].clamp_min(1e-7)
            brightness_gate = (union_brightness / gate_scale).clamp(0.0, 1.0)
            offset = (
                radius.view(1, 1, *radius.shape) - einstein_radius
            ).abs()
            size_scale = images.shape[-1] / 64.0
            edge_width = max(0.5 * size_scale, 1e-3)
            closure_mask = (
                torch.sigmoid(
                    (offset - 2.0 * size_scale) / edge_width
                )
                * torch.sigmoid(
                    (8.0 * size_scale - offset) / edge_width
                )
                * valid
            )
            closure_dogs = torch.tanh(
                (dog_stack - partner_dogs) * brightness_gate
            ) * closure_mask
            unit = images / images.amax(
                (-2, -1), keepdim=True
            ).clamp_min(1e-7)
            unit = unit.clamp(0.0, 1.0)
            log_ratio_sq = torch.log(
                (1.0 + self.epsilon) / (unit + self.epsilon)
            ).square()
            mixed = F.conv2d(
                F.pad(log_ratio_sq, (2, 2, 2, 2), mode="reflect"),
                self.mixed_derivative,
            )
            mixed = mixed.abs()
            mixed = self._normalize_map(mixed, radius, True)
            return torch.cat(
                (
                    current,
                    asinh_map,
                    radial,
                    asinh_radial,
                    *closure_dogs.split(1, dim=1),
                    mixed,
                ),
                dim=1,
            )


class ModelIVPhysicsSummary(nn.Module):
    """1,395 zero-parameter D4-invariant Model-IV physics statistics."""

    output_dim = 1395
    RADIAL_EDGES = (0, 3, 5, 7, 9, 11, 13, 16, 20, 24, 29, 35, 46)
    MULTIPOLE_EDGES = (3, 6, 9, 12, 15, 19, 24, 31, 43)
    FOURIER_EDGES = (0, 1.5, 2.5, 3.5, 5, 7, 9, 12, 16, 21, 27, 34, 46)

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        )
        mixed_axis = torch.tensor([-1.0, -2.0, 0.0, 2.0, 1.0])
        self.register_buffer(
            "summary_sobel_x",
            sobel_x.view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "summary_sobel_y",
            sobel_x.T.contiguous().view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "summary_laplacian",
            laplacian.view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "summary_mixed",
            torch.outer(mixed_axis, mixed_axis).view(1, 1, 5, 5) / 64.0,
            persistent=False,
        )

    @staticmethod
    def _geometry(
        height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if height != width:
            raise ValueError("D4 summaries require square images")
        coordinates = torch.arange(height, device=device, dtype=torch.float32)
        coordinates = coordinates - (height - 1) / 2.0
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        return (
            torch.sqrt(xx.square() + yy.square()),
            torch.atan2(yy, xx),
            height / 64.0,
        )

    @staticmethod
    def _masks(
        radius: torch.Tensor, edges, scale: float
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            (radius >= lower * scale) & (radius < upper * scale)
            for lower, upper in zip(edges[:-1], edges[1:])
        )

    @staticmethod
    def _scalar_annular(
        values: torch.Tensor, masks: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        batch = values.shape[0]
        flat = values.reshape(batch, -1)
        quantiles = torch.tensor(
            (0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75,
             0.90, 0.95, 0.99, 0.999, 1.0),
            device=values.device,
            dtype=values.dtype,
        )
        quantile_values = torch.quantile(flat, quantiles, dim=-1).T
        mean = flat.mean(-1)
        centered = flat - mean[:, None]
        std = (centered.square().mean(-1) + 1e-12).sqrt()
        moments = torch.stack(
            (
                mean,
                std,
                flat.abs().mean(-1),
                (flat.square().mean(-1) + 1e-12).sqrt(),
                centered.pow(3).mean(-1) / std.clamp_min(1e-6).pow(3),
                centered.pow(4).mean(-1) / std.clamp_min(1e-6).pow(4) - 3.0,
            ),
            -1,
        )
        output = [quantile_values, moments]
        annular_quantiles = torch.tensor(
            (0.10, 0.50, 0.90),
            device=values.device,
            dtype=values.dtype,
        )
        for mask in masks:
            annulus = values[..., mask]
            quantile = torch.quantile(
                annulus, annular_quantiles, dim=-1
            ).T
            absolute_99 = torch.quantile(
                annulus.abs(), 0.99, dim=-1, keepdim=True
            )
            output.append(
                torch.cat(
                    (
                        annulus.mean(-1, keepdim=True),
                        annulus.std(-1, unbiased=False, keepdim=True),
                        annulus.abs().mean(-1, keepdim=True),
                        (
                            annulus.square().mean(-1, keepdim=True) + 1e-12
                        ).sqrt(),
                        quantile,
                        absolute_99,
                    ),
                    -1,
                )
            )
        return torch.cat(output, -1)

    @staticmethod
    def _multipoles(
        values: torch.Tensor,
        theta: torch.Tensor,
        masks: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        output = []
        for mask in masks:
            annulus = values[..., mask]
            angle = theta[mask]
            norm = annulus.abs().sum(-1).clamp_min(1e-8)
            for order in range(1, 9):
                real = (annulus * torch.cos(order * angle)).sum(-1) / norm
                imaginary = (
                    annulus * torch.sin(order * angle)
                ).sum(-1) / norm
                output.append(
                    torch.sqrt(real.square() + imaginary.square() + 1e-16)
                )
        return torch.stack(output, -1)

    @classmethod
    def _spectrum(cls, values: torch.Tensor) -> torch.Tensor:
        _, height, width = values.shape
        window_y = torch.hann_window(
            height, periodic=False, device=values.device, dtype=values.dtype
        )
        window_x = torch.hann_window(
            width, periodic=False, device=values.device, dtype=values.dtype
        )
        centered = values - values.mean((-2, -1), keepdim=True)
        power = torch.fft.fft2(
            centered * torch.outer(window_y, window_x)
        ).abs().square()
        frequency_y = torch.fft.fftfreq(height, device=values.device) * height
        frequency_x = torch.fft.fftfreq(width, device=values.device) * width
        yy, xx = torch.meshgrid(frequency_y, frequency_x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        theta = torch.atan2(yy, xx)
        scale = height / 64.0
        total = power.sum((-2, -1)).clamp_min(1e-10)
        output = [
            power[..., mask].sum(-1) / total
            for mask in cls._masks(radius, cls.FOURIER_EDGES, scale)
        ]
        for lower, upper in ((2, 6), (6, 12), (12, 22), (22, 40)):
            mask = (radius >= lower * scale) & (radius < upper * scale)
            annulus = power[..., mask]
            angle = theta[mask]
            norm = annulus.sum(-1).clamp_min(1e-10)
            for order in (2, 4, 6, 8):
                real = (annulus * torch.cos(order * angle)).sum(-1) / norm
                imaginary = (
                    annulus * torch.sin(order * angle)
                ).sum(-1) / norm
                output.append(
                    torch.sqrt(real.square() + imaginary.square() + 1e-20)
                )
        return torch.stack(output, -1)

    @staticmethod
    def _conv(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = kernel.shape[-1] // 2
        return F.conv2d(
            F.pad(values[:, None], (pad, pad, pad, pad), mode="reflect"),
            kernel,
        )[:, 0]

    def _derivatives(
        self,
        values: torch.Tensor,
        theta: torch.Tensor,
        masks: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        gradient_x = self._conv(values, self.summary_sobel_x)
        gradient_y = self._conv(values, self.summary_sobel_y)
        gradient = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1e-12
        )
        radial = gradient_x * torch.cos(theta) + gradient_y * torch.sin(theta)
        tangential = (
            -gradient_x * torch.sin(theta) + gradient_y * torch.cos(theta)
        )
        laplacian = self._conv(values, self.summary_laplacian)
        mixed = self._conv(values, self.summary_mixed)
        output = []
        for mask in masks:
            for derivative in (
                gradient,
                radial,
                tangential,
                laplacian,
                mixed,
            ):
                annulus = derivative[..., mask]
                output.extend(
                    (
                        annulus.abs().mean(-1),
                        (annulus.square().mean(-1) + 1e-12).sqrt(),
                    )
                )
        return torch.stack(output, -1)

    def forward(self, morphology: torch.Tensor) -> torch.Tensor:
        if morphology.ndim != 4 or morphology.shape[1] != 8:
            raise ValueError(
                "Model-IV physics summaries require [B, 8, H, W] maps"
            )
        with torch.autocast(device_type=morphology.device.type, enabled=False):
            morphology = morphology.float()
            radius, theta, scale = self._geometry(
                morphology.shape[-2], morphology.shape[-1], morphology.device
            )
            annuli = self._masks(radius, self.RADIAL_EDGES, scale)
            multipole_annuli = self._masks(
                radius, self.MULTIPOLE_EDGES, scale
            )
            output = []
            for channel in range(2, 7):
                values = morphology[:, channel]
                output.extend(
                    (
                        self._scalar_annular(values, annuli),
                        self._multipoles(values, theta, multipole_annuli),
                        self._spectrum(values),
                    )
                )
                if channel >= 4:
                    output.append(self._derivatives(values, theta, annuli))
            summary = torch.cat(output, -1)
            if summary.shape[-1] != self.output_dim:
                raise RuntimeError(
                    f"Model-IV summary drift: {summary.shape[-1]} != "
                    f"{self.output_dim}"
                )
            return torch.nan_to_num(
                summary, nan=0.0, posinf=1e6, neginf=-1e6
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
