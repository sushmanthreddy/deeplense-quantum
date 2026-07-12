"""Physics-conditioned D4 orbit encoder and quantum/classical bottlenecks."""

from __future__ import annotations

import math
from typing import Dict, Literal, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .quantum import (
    CAYLEY_EDGE_FAMILIES,
    D4OrbitQuantumBottleneck,
    R2S_PLAQUETTES,
    R2_EDGES,
    RS_PLAQUETTES,
    R_EDGES,
    S_EDGES,
)


def norm2d(channels: int) -> nn.GroupNorm:
    """View-local normalization with identical train/eval behavior."""

    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def d4_transform(x: torch.Tensor, rotation: int, reflected: int) -> torch.Tensor:
    """Apply ``r^rotation s^reflected`` to an image batch."""

    if reflected:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, rotation, dims=(-2, -1))


def d4_views(x: torch.Tensor) -> torch.Tensor:
    """Return all eight views in ``[(r^0..r^3), (r^0s..r^3s)]`` order."""

    return torch.stack(
        [d4_transform(x, k, f) for f in (0, 1) for k in range(4)], dim=1
    )


def spectral_morphology_summary(physics_identity: torch.Tensor) -> torch.Tensor:
    """Return 16 fixed D4-invariant radial/spectral morphology statistics.

    The input is the identity-view output of :class:`PhysicsChannelBank`.
    Channels 1, 2, and 4 are respectively stabilized log intensity,
    gradient magnitude, and absolute difference-of-Gaussians.  The summary
    was calibrated at 96x96; radii and frequency bands scale relative to that
    reference resolution.
    """

    if physics_identity.ndim != 4 or physics_identity.shape[1] < 5:
        raise ValueError(
            f"Expected (B,C,H,W) physics tensor with C>=5, got {tuple(physics_identity.shape)}"
        )
    height, width = physics_identity.shape[-2:]
    if height != width:
        raise ValueError("Spectral morphology summary requires square images")

    with torch.autocast(device_type=physics_identity.device.type, enabled=False):
        fields = physics_identity.float()
        log_intensity = fields[:, 1]
        gradient = fields[:, 2]
        dog = fields[:, 4]
        device, dtype = fields.device, fields.dtype
        epsilon = 1e-12

        coordinate = (
            torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2
        )
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        theta = torch.atan2(yy, xx)
        spatial_scale = height / 96.0

        def ring(lower: float, upper: float) -> torch.Tensor:
            return (
                (radius >= lower * spatial_scale)
                & (radius < upper * spatial_scale)
            ).to(dtype)

        ring_0_4, ring_4_8, ring_8_12 = (
            ring(0.0, 4.0),
            ring(4.0, 8.0),
            ring(8.0, 12.0),
        )

        def annular_mean(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return (field * mask).sum(dim=(-2, -1)) / mask.sum().clamp_min(1.0)

        radial = torch.stack(
            (
                annular_mean(log_intensity, ring_8_12),
                annular_mean(log_intensity, ring_4_8),
                annular_mean(gradient, ring_4_8),
                annular_mean(log_intensity, ring_0_4),
                annular_mean(dog, ring_0_4),
                annular_mean(dog, ring_4_8),
            ),
            dim=1,
        )

        hann = torch.hann_window(
            height, periodic=False, dtype=dtype, device=device
        )
        window = hann[:, None] * hann[None, :]
        weighted_mean = (
            (log_intensity * window).sum(dim=(-2, -1), keepdim=True)
            / window.sum()
        )
        centered = (log_intensity - weighted_mean) * window
        power = torch.fft.fft2(centered).abs().square()

        frequency = torch.fft.fftfreq(height, d=1.0, device=device).to(dtype)
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        frequency_radius = torch.sqrt(fx.square() + fy.square())
        frequency_scale = 96.0 / height
        edges = torch.tensor(
            (
                1 / 64,
                1 / 32,
                1 / 20,
                1 / 14,
                1 / 10,
                1 / 7,
                1 / 5,
                1 / 3,
                math.sqrt(0.5) + 1e-7,
            ),
            dtype=dtype,
            device=device,
        ) * frequency_scale
        band_masks = (
            (frequency_radius[None] >= edges[:-1, None, None])
            & (frequency_radius[None] < edges[1:, None, None])
        ).to(dtype)
        counts = band_masks.sum(dim=(-2, -1))
        if bool((counts == 0).any()):
            raise ValueError(
                f"Resolution {height} leaves an empty spectral morphology band"
            )
        band_sums = torch.einsum("bhw,khw->bk", power, band_masks)
        band_means = band_sums / counts
        total_mean = power.mean(dim=(-2, -1)).clamp_min(epsilon)
        relative_log_power = torch.log(
            band_means.clamp_min(epsilon) / total_mean[:, None]
        )
        probabilities = band_sums / band_sums.sum(
            dim=1, keepdim=True
        ).clamp_min(epsilon)
        entropy = -(
            probabilities * probabilities.clamp_min(epsilon).log()
        ).sum(dim=1) / math.log(8)
        spectrum = torch.stack(
            (
                relative_log_power[:, 2],
                relative_log_power[:, 3],
                relative_log_power[:, 1],
                relative_log_power[:, 5],
                entropy,
                relative_log_power[:, 4],
            ),
            dim=1,
        )

        angular_fields = torch.stack(
            (log_intensity, gradient, log_intensity, gradient), dim=1
        )
        angular_masks = torch.stack(
            (ring_4_8, ring_8_12, ring_4_8, ring_4_8), dim=0
        )
        orders = torch.tensor((1.0, 2.0, 2.0, 1.0), dtype=dtype, device=device)
        phases = orders[:, None, None] * theta
        weighted_fields = angular_fields * angular_masks[None]
        real = (weighted_fields * phases.cos()[None]).sum(dim=(-2, -1))
        imaginary = (weighted_fields * phases.sin()[None]).sum(dim=(-2, -1))
        denominator = weighted_fields.sum(dim=(-2, -1))
        magnitude = torch.sqrt(real.square() + imaginary.square())
        angular = torch.where(
            denominator > 1e-8,
            magnitude / denominator.clamp_min(1e-8),
            torch.zeros_like(magnitude),
        )

        summary = torch.cat((radial, spectrum, angular), dim=1)
        if summary.shape[1] != 16 or not bool(torch.isfinite(summary).all()):
            raise RuntimeError("Invalid spectral morphology summary")
        return summary


def lens_morphology_summary(physics_identity: torch.Tensor) -> torch.Tensor:
    """Return 60 fixed, D4-invariant lens morphology statistics.

    Twelve radial annuli resolve the central image, Einstein-ring region, and
    outer background.  Each annulus contributes log-intensity mean and
    standard deviation plus mean gradient and Laplacian response.  Complex
    angular moments of orders one through four add twelve phase-independent
    summaries over three annuli spanning the bright ring.  Magnitudes make
    the bank invariant to rotations and reflections, while retaining radial
    texture cues that global average pooling discards.
    """

    if physics_identity.ndim != 4 or physics_identity.shape[1] < 4:
        raise ValueError(
            f"Expected (B,C,H,W) physics tensor with C>=4, got {tuple(physics_identity.shape)}"
        )
    height, width = physics_identity.shape[-2:]
    if height != width:
        raise ValueError("Lens morphology summary requires square images")

    with torch.autocast(device_type=physics_identity.device.type, enabled=False):
        fields = physics_identity.float()
        log_intensity = fields[:, 1]
        gradient = fields[:, 2]
        laplacian = fields[:, 3]
        device, dtype = fields.device, fields.dtype

        coordinate = (
            torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2
        )
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square()) / (height / 2.0)
        theta = torch.atan2(yy, xx)
        edges = torch.tensor(
            (
                0.0,
                1 / 12,
                1 / 6,
                1 / 4,
                1 / 3,
                5 / 12,
                1 / 2,
                5 / 8,
                3 / 4,
                7 / 8,
                1.0,
                9 / 8,
                3 / 2,
            ),
            device=device,
            dtype=dtype,
        )
        annuli = torch.stack(
            [
                ((radius >= lower) & (radius < upper)).to(dtype)
                for lower, upper in zip(edges[:-1], edges[1:])
            ],
            dim=0,
        )
        counts = annuli.sum(dim=(-2, -1)).clamp_min(1.0)

        def annular_mean(field: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bhw,rhw->br", field, annuli) / counts

        log_mean = annular_mean(log_intensity)
        log_second = annular_mean(log_intensity.square())
        log_std = (log_second - log_mean.square()).clamp_min(0.0).sqrt()
        radial = torch.stack(
            (
                log_mean,
                log_std,
                annular_mean(gradient),
                annular_mean(laplacian),
            ),
            dim=-1,
        ).reshape(fields.shape[0], -1)

        # Annuli 3, 5, and 7 cover radii 12--16, 20--24, and 30--36 pixels
        # at the calibrated 96-pixel resolution.
        ring_masks = annuli[[3, 5, 7]]
        orders = torch.arange(1, 5, device=device, dtype=dtype)
        phase = orders[:, None, None] * theta
        weighted = log_intensity[:, None] * ring_masks[None]
        denominator = weighted.sum(dim=(-2, -1)).clamp_min(1e-8)
        real = torch.einsum("brhw,khw->brk", weighted, phase.cos())
        imaginary = torch.einsum("brhw,khw->brk", weighted, phase.sin())
        angular = (
            torch.sqrt(real.square() + imaginary.square())
            / denominator[:, :, None]
        ).reshape(fields.shape[0], -1)

        summary = torch.cat((radial, angular), dim=1)
        if summary.shape[1] != 60 or not bool(torch.isfinite(summary).all()):
            raise RuntimeError("Invalid lens morphology summary")
        return summary


# Frozen, label-free pruning order derived from the same-half morphology-KD
# checkpoint's classifier path magnitudes.  The full 60-vector still enters
# the orbit projection; only this compact invariant subset bypasses the
# quantum bottleneck in the annular-Haar candidate.
HAAR_MORPHOLOGY_CONTEXT_INDICES: Tuple[int, ...] = (
    20,
    8,
    17,
    25,
    24,
    16,
    48,
    21,
    9,
    12,
    15,
    53,
    22,
    13,
    10,
)


def annular_haar_scattering_summary(
    physics_views: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen 104-D localized Haar-like scattering descriptor.

    Channel one of ``physics_views`` is the stabilized log intensity.  Four
    centered finite-difference scales and a D4-complete set of axis/diagonal
    directions are passed through modulus and log compression.  Six annular
    averages produce 96 first-order coefficients; inner/outer coefficients of
    variation add eight scale-wise intermittency statistics.  At 96 pixels,
    the exact radial edges are 0, 4, 8, 12, 18, 30, and 68 pixels.  Other
    square resolutions retain the same normalized radial regions, while the
    integer difference offsets remain calibrated to Model-I's 96-pixel input.
    """

    if physics_views.ndim != 4 or physics_views.shape[1] < 2:
        raise ValueError(
            "Expected a (B,C,H,W) physics tensor with C>=2, got "
            f"{tuple(physics_views.shape)}"
        )
    height, width = physics_views.shape[-2:]
    if height != width:
        raise ValueError("Annular Haar summary requires square images")
    if height <= 7:
        raise ValueError("Annular Haar summary requires image width greater than seven")

    with torch.autocast(device_type=physics_views.device.type, enabled=False):
        log_intensity = physics_views[:, 1].float()
        device, dtype = log_intensity.device, log_intensity.dtype
        coordinate = (
            torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2
        )
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        spatial_scale = height / 96.0
        edges = torch.tensor(
            (0.0, 4.0, 8.0, 12.0, 18.0, 30.0, 68.0),
            device=device,
            dtype=dtype,
        ) * spatial_scale
        annuli = torch.stack(
            [
                ((radius >= lower) & (radius < upper)).to(dtype)
                for lower, upper in zip(edges[:-1], edges[1:])
            ],
            dim=0,
        )
        annulus_counts = annuli.sum(dim=(-2, -1))
        if bool((annulus_counts == 0).any()):
            raise ValueError(
                f"Resolution {height} leaves an empty annular Haar region"
            )
        inner = (
            (radius >= 4.0 * spatial_scale)
            & (radius < 18.0 * spatial_scale)
        ).to(dtype)
        outer = (
            (radius >= 18.0 * spatial_scale)
            & (radius < 48.0 * spatial_scale)
        ).to(dtype)
        if not bool(inner.sum() > 0) or not bool(outer.sum() > 0):
            raise ValueError(
                f"Resolution {height} leaves an empty Haar intermittency region"
            )

        annular_features = []
        intermittency_features = []
        for offset in (1, 2, 4, 7):
            padded = F.pad(
                log_intensity[:, None],
                (offset, offset, offset, offset),
                mode="reflect",
            )[:, 0]
            responses = []
            directions = (
                (0, offset),
                (offset, 0),
                (offset, offset),
                (offset, -offset),
            )
            for delta_y, delta_x in directions:
                positive = padded[
                    :,
                    offset + delta_y : offset + delta_y + height,
                    offset + delta_x : offset + delta_x + width,
                ]
                negative = padded[
                    :,
                    offset - delta_y : offset - delta_y + height,
                    offset - delta_x : offset - delta_x + width,
                ]
                responses.append(
                    torch.log1p(8.0 * (positive - negative).abs())
                )
            response = torch.stack(responses, dim=1)
            pooled = (
                torch.einsum("bohw,rhw->bor", response, annuli)
                / annulus_counts
            )
            annular_features.append(pooled.reshape(log_intensity.shape[0], -1))

            orientation_mean = response.mean(dim=1)
            for mask in (inner, outer):
                count = mask.sum()
                mean = (orientation_mean * mask).sum(dim=(-2, -1)) / count
                variance = (
                    (orientation_mean - mean[:, None, None]).square() * mask
                ).sum(dim=(-2, -1)) / count
                intermittency_features.append(
                    torch.log1p(
                        torch.sqrt(variance + 1e-8) / (mean + 1e-4)
                    )
                )

        summary = torch.cat(
            annular_features
            + [torch.stack(intermittency_features, dim=1)],
            dim=1,
        )
        if summary.shape[1] != 104 or not bool(torch.isfinite(summary).all()):
            raise RuntimeError("Invalid annular Haar scattering summary")
        return summary


def cross_scale_scattering_summary(
    physics_views: torch.Tensor,
) -> torch.Tensor:
    """Return 32 fixed second-order cross-scale scattering coefficients.

    Channel one of ``physics_views`` is the stabilized log intensity.  A
    first centered finite-difference modulus is formed for horizontal,
    vertical, and the two diagonal directions at offsets 1, 2, 4, and 7.
    Four declared paths then apply a second, coarser centered-difference
    modulus in the *same* direction: ``(1,2)``, ``(2,4)``, ``(4,7)``, and
    ``(1,7)``.  Horizontal/vertical responses and the two diagonal responses
    are pooled separately over four centered annuli, giving
    ``4 paths * 2 direction families * 4 annuli = 32`` coefficients.

    The operation is shared independently across the flattened D4 views.
    Consequently a regular permutation of the input views produces the same
    regular permutation of the output rows, while the direction-family and
    radial reductions are themselves D4 invariant.
    """

    if physics_views.ndim != 4 or physics_views.shape[1] < 2:
        raise ValueError(
            "Expected a (B,C,H,W) physics tensor with C>=2, got "
            f"{tuple(physics_views.shape)}"
        )
    height, width = physics_views.shape[-2:]
    if height != width:
        raise ValueError("Cross-scale scattering requires square images")
    if height <= 7:
        raise ValueError(
            "Cross-scale scattering requires image width greater than seven"
        )

    with torch.autocast(device_type=physics_views.device.type, enabled=False):
        log_intensity = physics_views[:, 1].float()
        device, dtype = log_intensity.device, log_intensity.dtype
        coordinate = (
            torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2
        )
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        spatial_scale = height / 96.0
        edges = torch.tensor(
            (4.0, 8.0, 12.0, 18.0, 30.0),
            device=device,
            dtype=dtype,
        ) * spatial_scale
        annuli = torch.stack(
            [
                ((radius >= lower) & (radius < upper)).to(dtype)
                for lower, upper in zip(edges[:-1], edges[1:])
            ],
            dim=0,
        )
        annulus_counts = annuli.sum(dim=(-2, -1))
        if bool((annulus_counts == 0).any()):
            raise ValueError(
                f"Resolution {height} leaves an empty cross-scale annulus"
            )

        directions = ((0, 1), (1, 0), (1, 1), (1, -1))

        def centered_modulus(
            field: torch.Tensor,
            offset: int,
            direction: Tuple[int, int],
        ) -> torch.Tensor:
            padded = F.pad(
                field[:, None],
                (offset, offset, offset, offset),
                mode="reflect",
            )[:, 0]
            delta_y = int(direction[0]) * offset
            delta_x = int(direction[1]) * offset
            positive = padded[
                :,
                offset + delta_y : offset + delta_y + height,
                offset + delta_x : offset + delta_x + width,
            ]
            negative = padded[
                :,
                offset - delta_y : offset - delta_y + height,
                offset - delta_x : offset - delta_x + width,
            ]
            return (positive - negative).abs()

        first_modulus = {
            offset: torch.stack(
                [
                    centered_modulus(log_intensity, offset, direction)
                    for direction in directions
                ],
                dim=1,
            )
            for offset in (1, 2, 4, 7)
        }
        features = []
        for finer, coarser in ((1, 2), (2, 4), (4, 7), (1, 7)):
            cascaded = torch.stack(
                [
                    centered_modulus(
                        first_modulus[finer][:, index],
                        coarser,
                        direction,
                    )
                    for index, direction in enumerate(directions)
                ],
                dim=1,
            )
            cascaded = torch.log1p(8.0 * cascaded)
            direction_families = torch.stack(
                (
                    cascaded[:, :2].mean(dim=1),
                    cascaded[:, 2:].mean(dim=1),
                ),
                dim=1,
            )
            pooled = (
                torch.einsum("bfhw,rhw->bfr", direction_families, annuli)
                / annulus_counts
            )
            features.append(pooled.reshape(log_intensity.shape[0], -1))

        summary = torch.cat(features, dim=1)
        if summary.shape[1] != 32 or not bool(torch.isfinite(summary).all()):
            raise RuntimeError("Invalid cross-scale scattering summary")
        return summary


def _normalized_sylvester_hadamard_32() -> torch.Tensor:
    """Construct the fixed normalized Sylvester Hadamard matrix of order 32."""

    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < 32:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(32.0)


def invariant_annular_haar_coefficients(
    haar_summary: torch.Tensor,
) -> torch.Tensor:
    """Reduce eight normalized Haar views to 56 canonical D4 invariants.

    Sorting before averaging gives a canonical reduction under an exact view
    permutation.  Axis and diagonal direction pairs are averaged separately:
    D4 relates horizontal to vertical and the two diagonals to one another,
    but it does not rotate an axis into a diagonal.
    """

    if haar_summary.ndim != 3 or tuple(haar_summary.shape[1:]) != (8, 104):
        raise ValueError(
            "Expected normalized Haar views with shape (B,8,104), got "
            f"{tuple(haar_summary.shape)}"
        )
    with torch.autocast(device_type=haar_summary.device.type, enabled=False):
        canonical_views = haar_summary.float().sort(dim=1).values.mean(dim=1)
        first_order = canonical_views[:, :96].reshape(-1, 4, 4, 6)
        axis = first_order[:, :, :2].mean(dim=2)
        diagonal = first_order[:, :, 2:].mean(dim=2)
        localized = torch.stack((axis, diagonal), dim=2).reshape(-1, 48)
        intermittency = canonical_views[:, 96:]
        invariant = torch.cat((localized, intermittency), dim=1)
        if invariant.shape[1] != 56 or not bool(torch.isfinite(invariant).all()):
            raise RuntimeError("Invalid invariant annular Haar coefficients")
        return invariant


class PhysicsChannelBank(nn.Module):
    """Zero-parameter, stabilized morphology channels for photon-count images.

    These are physics/morphology informed rather than a PINN: no differential
    equation residual is optimized. The last channel is a finite version of
    LensPINN's mixed derivative of squared log intensity ratio.
    """

    def __init__(
        self,
        log_gain: float = 20.0,
        epsilon: float = 1e-3,
        reference_pixels: int = 96,
        variant: Literal["base", "radial"] = "base",
    ) -> None:
        super().__init__()
        if variant not in ("base", "radial"):
            raise ValueError(f"Unknown physics channel variant: {variant}")
        self.log_gain = float(log_gain)
        self.epsilon = float(epsilon)
        self.reference_pixels = int(reference_pixels)
        self.variant = variant
        self.output_channels = 8 if variant == "base" else 10
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.transpose(0, 1).contiguous()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3), persistent=False)

    @staticmethod
    def _conv_reflect(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel)

    @staticmethod
    def _avg_reflect(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        return F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        scale = x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        x = (x / scale).clamp(0.0, 1.0)
        log_intensity = torch.log1p(self.log_gain * x) / math.log1p(self.log_gain)

        gx = self._conv_reflect(log_intensity, self.sobel_x)
        gy = self._conv_reflect(log_intensity, self.sobel_y)
        pixel_scale = float(x.shape[-1]) / self.reference_pixels
        gradient = torch.sqrt(gx.square() + gy.square() + 1e-8) * pixel_scale
        laplacian = self._conv_reflect(log_intensity, self.laplacian).abs() * pixel_scale**2
        small_kernel = max(3, int(round(3 * pixel_scale)) | 1)
        large_kernel = max(small_kernel + 2, int(round(9 * pixel_scale)) | 1)
        dog = (
            self._avg_reflect(log_intensity, small_kernel)
            - self._avg_reflect(log_intensity, large_kernel)
        ).abs()

        height, width = x.shape[-2:]
        yy = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square()).clamp_min(1e-4)
        unit_x = (grid_x / radius).view(1, 1, height, width)
        unit_y = (grid_y / radius).view(1, 1, height, width)
        radial = (gx * unit_x + gy * unit_y).abs() * pixel_scale
        tangential = (-gx * unit_y + gy * unit_x).abs() * pixel_scale

        log_ratio_sq = torch.log((1.0 + self.epsilon) / (x + self.epsilon)).square()
        mixed = self._conv_reflect(
            self._conv_reflect(log_ratio_sq, self.sobel_x), self.sobel_y
        ).abs() * pixel_scale**2

        channels = (
            x,
            log_intensity,
            torch.tanh(2.0 * gradient),
            torch.tanh(laplacian),
            torch.tanh(4.0 * dog),
            torch.tanh(2.0 * radial),
            torch.tanh(2.0 * tangential),
            torch.tanh(mixed),
        )
        if self.variant == "radial":
            mid_kernel = max(3, int(round(5 * pixel_scale)) | 1)
            outer_kernel = max(mid_kernel + 2, int(round(13 * pixel_scale)) | 1)
            signed_multiscale = self._avg_reflect(
                log_intensity, mid_kernel
            ) - self._avg_reflect(log_intensity, outer_kernel)
            central_envelope = torch.exp(-0.5 * (radius / 0.25).square()).view(
                1, 1, height, width
            )
            channels = channels + (
                torch.tanh(4.0 * signed_multiscale),
                log_intensity * central_envelope,
            )
        return torch.cat(channels, dim=1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class EfficientChannelAttention(nn.Module):
    """Five-parameter local channel attention used by the half-budget encoder."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("ECA kernel size must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights)).transpose(1, 2).unsqueeze(-1)
        return x * weights


class MBConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, expand: int, stride: int, kernel_size: int = 5
    ) -> None:
        super().__init__()
        hidden = in_channels * expand
        layers = []
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y


class MBConvECA(nn.Module):
    """MBConv with efficient channel attention instead of an SE MLP."""

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
        layers = []
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
                EfficientChannelAttention(),
                nn.Conv2d(hidden, out_channels, 1, bias=False),
                norm2d(out_channels),
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y


def paired_spatial_statistics(features: torch.Tensor) -> torch.Tensor:
    """Pool channel means, standard deviations, and adjacent-pair correlations.

    Correlations pair channels deterministically as ``(0,1), (2,3), ...``.
    Accumulation stays in fp32 under mixed precision because the weak spatial
    dispersion of sparse lens features is easily lost in bfloat16.  The
    returned descriptor has ``2*C + floor(C/2)`` entries and remains fully
    differentiable, including for spatially constant channels.
    """

    if features.ndim != 4:
        raise ValueError(
            f"Expected a (B,C,H,W) feature tensor, got {tuple(features.shape)}"
        )
    if features.shape[1] < 2:
        raise ValueError("Paired spatial statistics require at least two channels")
    with torch.autocast(device_type=features.device.type, enabled=False):
        values = features.float()
        mean = values.mean(dim=(-2, -1))
        centered = values - mean[:, :, None, None]
        variance = centered.square().mean(dim=(-2, -1)).clamp_min(0.0)
        std = (variance + 1e-8).sqrt()
        pair_count = values.shape[1] // 2
        even = centered[:, 0 : 2 * pair_count : 2]
        odd = centered[:, 1 : 2 * pair_count : 2]
        covariance = (even * odd).mean(dim=(-2, -1))
        denominator = (
            std[:, 0 : 2 * pair_count : 2]
            * std[:, 1 : 2 * pair_count : 2]
        ).clamp_min(1e-6)
        correlation = covariance / denominator
        descriptor = torch.cat((mean, std, correlation), dim=1)
    return descriptor.to(features.dtype)


class CompactOrbitEncoder(nn.Module):
    """Shared MobileNet-style image encoder; weights do not scale with |D4|."""

    def __init__(
        self,
        input_channels: int = 8,
        variant: Literal[
            "micro",
            "micro-stat",
            "deep-se",
            "deep-se-morph",
            "deep-se-haar-morph",
            "deep-se-mscorr",
            "eca",
            "tiny",
            "small",
        ] = "tiny",
        shared_late_refinement: bool = False,
    ) -> None:
        super().__init__()
        if shared_late_refinement and variant != "deep-se-haar-morph":
            raise ValueError(
                "Shared late refinement requires deep-se-haar-morph"
            )
        if variant in ("micro", "micro-stat"):
            # Retain the first three spatial stages of ``tiny`` and remove its
            # 64->96 transition/refinement pair.  A 64->96 pointwise readout
            # preserves a useful orbit-embedding width while cutting the full
            # classifier to almost exactly one half of the base parameter count.
            stem_channels = 16
            specs = (
                (16, 24, 2, 2),
                (24, 24, 2, 1),
                (24, 40, 3, 2),
                (40, 40, 3, 1),
                (40, 64, 3, 2),
                (64, 64, 3, 1),
            )
            final_channels = 64
            # ``micro-stat`` retains both the first and second spatial moments
            # of every learned channel.  This parameter-free second-order pool
            # preserves localized substructure that global averaging erases.
            self.output_dim = 192 if variant == "micro-stat" else 96
        elif variant in (
            "deep-se",
            "deep-se-morph",
            "deep-se-haar-morph",
            "deep-se-mscorr",
        ):
            # Preserve the successful micro encoder's first five SE blocks,
            # make its 64-channel refinement economical, and restore the
            # missing 64->96 transition/refinement stages.  Expansion one in
            # the late blocks keeps the complete SE topology under half budget.
            stem_channels = 16
            specs = (
                (16, 24, 2, 2),
                (24, 24, 2, 1),
                (24, 40, 3, 2),
                (40, 40, 3, 1),
                (40, 64, 3, 2),
                (64, 64, 1, 1),
                (64, 96, 1, 2),
                (96, 96, 1, 1),
            )
            final_channels = 96
            # Width 200 spends the remaining budget on the classical image
            # representation while preserving the validated four-head circuit
            # and 48->32->3 classifier shapes for same-subset warm starts.
            if variant in ("deep-se-morph", "deep-se-haar-morph"):
                self.output_dim = 192
            elif variant == "deep-se-mscorr":
                # Block-3: 40 means + 40 stds + 20 adjacent correlations.
                # Block-5: 64 means + 64 stds + 32 adjacent correlations.
                # Final: 180 means.  All 440 entries feed only the orbit core.
                self.output_dim = 440
            else:
                self.output_dim = 200
        elif variant == "eca":
            # Preserve all spatial stages and channel widths of ``tiny`` while
            # replacing parameter-heavy SE MLPs and reducing expansion ratios.
            stem_channels = 16
            specs = (
                (16, 24, 2, 2),
                (24, 24, 2, 1),
                (24, 40, 2, 2),
                (40, 40, 2, 1),
                (40, 64, 2, 2),
                (64, 64, 1, 1),
                (64, 96, 2, 2),
                (96, 96, 2, 1),
            )
            final_channels = 96
            self.output_dim = 128
        elif variant == "tiny":
            stem_channels = 16
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
            final_channels = 96
            self.output_dim = 128
        elif variant == "small":
            stem_channels = 24
            specs = (
                (24, 32, 2, 2),
                (32, 32, 2, 1),
                (32, 48, 3, 2),
                (48, 48, 3, 1),
                (48, 80, 3, 2),
                (80, 80, 3, 1),
                (80, 128, 3, 2),
                (128, 128, 3, 1),
                (128, 160, 3, 1),
            )
            final_channels = 160
            self.output_dim = 192
        else:
            raise ValueError(f"Unknown encoder variant: {variant}")
        self.variant = variant
        self.shared_late_refinement = bool(shared_late_refinement)
        self.statistical_pool = variant == "micro-stat"
        self.multiscale_correlation_pool = variant == "deep-se-mscorr"
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, 5, stride=2, padding=2, bias=False),
            norm2d(stem_channels),
            nn.SiLU(inplace=True),
        )
        block = MBConvECA if variant == "eca" else MBConv
        self.blocks = nn.Sequential(
            *(block(cin, cout, expand, stride) for cin, cout, expand, stride in specs)
        )
        final_feature_channels = (
            96
            if self.statistical_pool
            else (180 if self.multiscale_correlation_pool else self.output_dim)
        )
        self.final = nn.Sequential(
            nn.Conv2d(final_channels, final_feature_channels, 1, bias=False),
            norm2d(final_feature_channels),
            nn.SiLU(inplace=True),
        )
        if self.shared_late_refinement:
            # Two additional weight-shared applications of the 64-channel
            # refinement and two of the 96-channel refinement.  Exact zeros
            # preserve the established encoder function at initialization.
            self.shared_refinement_gates = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("shared_refinement_gates", None)

    def _apply_shared_refinement(
        self,
        features: torch.Tensor,
        block: nn.Module,
        block_index: int,
    ) -> torch.Tensor:
        """Apply two gated repetitions of a declared stride-one block."""

        if self.shared_refinement_gates is None:
            return features
        if block_index == 5:
            gates = self.shared_refinement_gates[:2]
        elif block_index == 7:
            gates = self.shared_refinement_gates[2:]
        else:
            return features
        for gate in gates:
            refined = block(features)
            delta = refined - features
            # Cast the scalar before multiplication so a zero gate does not
            # promote a bfloat16 activation and silently break exact replay.
            features = features + gate.to(features.dtype) * delta
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        block_3 = block_5 = None
        for index, block in enumerate(self.blocks):
            features = block(features)
            features = self._apply_shared_refinement(features, block, index)
            if self.multiscale_correlation_pool and index == 3:
                block_3 = features
            elif self.multiscale_correlation_pool and index == 5:
                block_5 = features
        features = self.final(features)
        mean = features.mean(dim=(-2, -1))
        if self.multiscale_correlation_pool:
            if block_3 is None or block_5 is None:
                raise RuntimeError("Missing a multiscale correlation feature tap")
            descriptor = torch.cat(
                (
                    paired_spatial_statistics(block_3),
                    paired_spatial_statistics(block_5),
                    mean,
                ),
                dim=1,
            )
            if descriptor.shape[1] != self.output_dim:
                raise RuntimeError(
                    f"Expected {self.output_dim} multiscale features, "
                    f"got {descriptor.shape[1]}"
                )
            return descriptor
        if not self.statistical_pool:
            return mean
        # Accumulate the variance in fp32 under mixed precision; sparse lens
        # activations can otherwise lose the small dispersion signal in bf16.
        std = features.float().std(dim=(-2, -1), unbiased=False).to(features.dtype)
        return torch.cat((mean, std), dim=1)

    def forward_mean_and_std(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return final-map means and population standard deviations.

        This narrow interface is used by the opt-in tied mean--dispersion
        angle projection.  It intentionally does not alter :meth:`forward`,
        so every existing encoder variant retains its exact default behavior.
        The dispersion is accumulated in fp32 because the final 3x3 feature
        maps can have small within-view variation under bfloat16 autocast.
        """

        if self.statistical_pool or self.multiscale_correlation_pool:
            raise RuntimeError(
                "Tied mean-dispersion requires a mean-pooled encoder variant"
            )
        features = self.stem(x)
        for index, block in enumerate(self.blocks):
            features = block(features)
            features = self._apply_shared_refinement(features, block, index)
        features = self.final(features)
        mean = features.mean(dim=(-2, -1))
        std = features.float().std(dim=(-2, -1), unbiased=False).to(features.dtype)
        if mean.shape[1] != self.output_dim or std.shape != mean.shape:
            raise RuntimeError("Invalid final-map mean-dispersion descriptor")
        return mean, std


def _edge_products(values: torch.Tensor, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    return torch.stack([values[..., a] * values[..., b] for a, b in edges], dim=-1)


def _motif_products(
    values: torch.Tensor, motifs: Sequence[Tuple[int, ...]]
) -> torch.Tensor:
    return torch.stack(
        [values[..., list(motif)].prod(dim=-1) for motif in motifs], dim=-1
    )


class ClassicalOrbitMixer(nn.Module):
    """Parameter-matched, elementwise Fourier control for the quantum circuit."""

    invariants_per_head = 12

    def __init__(
        self,
        heads: int,
        layers: int,
        observable_readout: Literal[
            "pair", "plaquette", "cayley-complete"
        ] = "pair",
    ) -> None:
        super().__init__()
        if observable_readout not in ("pair", "plaquette", "cayley-complete"):
            raise ValueError(f"Unknown observable readout: {observable_readout}")
        self.heads = heads
        self.layers = layers
        self.observable_readout = observable_readout
        self.invariants_per_head = {
            "pair": 12,
            "plaquette": 16,
            "cayley-complete": 28,
        }[observable_readout]
        params = torch.zeros(heads, layers, 11)
        params[..., 0] = 1.0
        params[..., 6] = 1.0
        params[..., 10] = 1.0
        params += 0.02 * torch.randn_like(params)
        self.params = nn.Parameter(params)

    @property
    def output_dim(self) -> int:
        return self.heads * self.invariants_per_head

    def forward(self, orbit_features: torch.Tensor) -> torch.Tensor:
        u, v = orbit_features[:, :, 0], orbit_features[:, :, 1]
        for layer in range(self.layers):
            p = self.params[:, layer].unsqueeze(0)
            new_u = torch.tanh(
                p[..., 0, None] * u
                + p[..., 1, None] * v
                + p[..., 2, None]
                + p[..., 3, None] * torch.sin(u)
                + p[..., 4, None] * torch.cos(v)
            )
            new_v = torch.tanh(
                p[..., 5, None] * u
                + p[..., 6, None] * v
                + p[..., 7, None]
                + p[..., 8, None] * torch.sin(v)
                + p[..., 9, None] * torch.cos(u)
            )
            residual = torch.sigmoid(p[..., 10, None])
            u = residual * u + (1.0 - residual) * new_u
            v = residual * v + (1.0 - residual) * new_v

        u_mean, v_mean = u.mean(-1), v.mean(-1)
        invariant_features = [
            u_mean,
            u.square().mean(-1) - u_mean.square(),
            v_mean,
            v.square().mean(-1) - v_mean.square(),
            _edge_products(u, R_EDGES).mean(-1),
            _edge_products(u, R2_EDGES).mean(-1),
            _edge_products(u, S_EDGES).mean(-1),
            _edge_products(v, R_EDGES).mean(-1),
            _edge_products(v, R2_EDGES).mean(-1),
            _edge_products(v, S_EDGES).mean(-1),
            (_edge_products(u, R_EDGES) - u_mean[..., None].square()).mean(-1),
            (_edge_products(v, R_EDGES) - v_mean[..., None].square()).mean(-1),
        ]
        if self.observable_readout == "cayley-complete":
            u_edges = tuple(
                _edge_products(u, edges) for edges in CAYLEY_EDGE_FAMILIES
            )
            v_edges = tuple(
                _edge_products(v, edges) for edges in CAYLEY_EDGE_FAMILIES
            )
            invariant_features = invariant_features[:4]
            invariant_features.extend(values.mean(-1) for values in u_edges)
            invariant_features.extend(values.mean(-1) for values in v_edges)
            invariant_features.extend(
                (values - u_mean[..., None].square()).mean(-1)
                for values in u_edges
            )
            invariant_features.extend(
                (values - v_mean[..., None].square()).mean(-1)
                for values in v_edges
            )
        elif self.observable_readout == "plaquette":
            invariant_features.extend(
                (
                    _motif_products(u, RS_PLAQUETTES).mean(-1),
                    _motif_products(v, RS_PLAQUETTES).mean(-1),
                    _motif_products(u, R2S_PLAQUETTES).mean(-1),
                    _motif_products(v, R2S_PLAQUETTES).mean(-1),
                )
            )
        features = torch.stack(invariant_features, dim=-1)
        return features.reshape(orbit_features.shape[0], self.output_dim)


class ParallelOrbitCores(nn.Module):
    """Joint, shared-input invariant cores with a one-scalar logit gate.

    ``quantum-classical`` is a compact hybrid with a complete classical
    bypass. ``classical-classical`` is its exactly parameter-matched control.
    The classifier applies one shared head to each invariant vector before
    mixing logits, so the two feature scales are normalized independently.
    """

    def __init__(
        self,
        heads: int,
        layers: int,
        architecture: Literal["quantum-classical", "classical-classical"],
        quantum_encoding: Literal["angle", "boltzmann", "gibbs"] = "angle",
        observable_readout: Literal[
            "pair", "plaquette", "cayley-complete"
        ] = "pair",
    ) -> None:
        super().__init__()
        if architecture == "quantum-classical":
            self.branch_a: nn.Module = D4OrbitQuantumBottleneck(
                heads=heads,
                reuploads=layers,
                input_encoding=quantum_encoding,
                observable_readout=observable_readout,
            )
        elif architecture == "classical-classical":
            if quantum_encoding != "angle":
                raise ValueError(
                    "Classical-classical fusion requires quantum_encoding='angle'"
                )
            self.branch_a = ClassicalOrbitMixer(
                heads=heads,
                layers=layers,
                observable_readout=observable_readout,
            )
        else:
            raise ValueError(f"Unknown parallel-core architecture: {architecture}")
        self.branch_b = ClassicalOrbitMixer(
            heads=heads,
            layers=layers,
            observable_readout=observable_readout,
        )
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.architecture = architecture
        self.observable_readout = observable_readout

    @property
    def output_dim(self) -> int:
        return self.branch_a.output_dim

    @property
    def mixing_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.mix_logit)

    def forward(self, orbit_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.branch_a(orbit_features), self.branch_b(orbit_features)


class GaugeFixedThreeClassLinear(nn.Module):
    """Three-logit linear map with the common bias gauge fixed to zero.

    A three-class softmax is unchanged by adding the same scalar to every
    logit, so one of the three output biases is unidentifiable.  Keeping only
    the first two biases and fixing the third to zero removes exactly that one
    redundant parameter without constraining the classifier weights.
    """

    __constants__ = ("in_features", "out_features")

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = 3
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features - 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        gauge_fixed_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.linear(features, self.weight, gauge_fixed_bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features=3, bias_gauge=2"


class MorphologyFusionHead(nn.Module):
    """Fuse normalized quantum invariants with fixed morphology context.

    Bottleneck invariants retain their own affine normalization instead of
    being renormalized jointly with the physically scaled summary bank.
    """

    def __init__(
        self,
        input_dim: int,
        invariant_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        gauge_fixed_output: bool = False,
    ) -> None:
        super().__init__()
        if not 0 < invariant_dim <= input_dim:
            raise ValueError("Invalid bottleneck invariant width for fusion head")
        self.invariant_dim = invariant_dim
        self.invariant_norm = nn.LayerNorm(invariant_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if gauge_fixed_output:
            if num_classes != 3:
                raise ValueError("Gauge-fixed output requires exactly three classes")
            self.classifier: nn.Module = GaugeFixedThreeClassLinear(hidden_dim)
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        invariant = self.invariant_norm(features[:, : self.invariant_dim])
        fused = torch.cat((invariant, features[:, self.invariant_dim :]), dim=1)
        return self.classifier(
            self.dropout(self.activation(self.projection(fused)))
        )


class HaarSubtypeResidual(nn.Module):
    """Fifteen-parameter invariant axion-versus-CDM logit correction."""

    feature_count = 15

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(self.feature_count))
        self.register_buffer(
            "selected_indices",
            torch.arange(self.feature_count, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "center", torch.zeros(self.feature_count), persistent=True
        )
        self.register_buffer(
            "scale", torch.ones(self.feature_count), persistent=True
        )

    def set_selection(
        self,
        selected_indices: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        if tuple(selected_indices.shape) != (self.feature_count,):
            raise ValueError("Haar subtype selection must contain 15 indices")
        if selected_indices.dtype != torch.long:
            raise ValueError("Haar subtype indices must use torch.long")
        if len(torch.unique(selected_indices)) != self.feature_count or not bool(
            ((selected_indices >= 0) & (selected_indices < 56)).all()
        ):
            raise ValueError("Haar subtype indices must be unique values in [0,56)")
        if tuple(center.shape) != (self.feature_count,) or tuple(scale.shape) != (
            self.feature_count,
        ):
            raise ValueError("Haar subtype normalization has the wrong shape")
        if not bool(torch.isfinite(center).all()) or not bool(
            torch.isfinite(scale).all()
        ):
            raise ValueError("Haar subtype normalization must be finite")
        if not bool((scale > 0).all()):
            raise ValueError("Haar subtype scales must be positive")
        self.selected_indices.copy_(selected_indices.to(self.selected_indices))
        self.center.copy_(center.to(self.center))
        self.scale.copy_(scale.to(self.scale))

    def forward(
        self, haar_summary: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        invariant = invariant_annular_haar_coefficients(haar_summary)
        selected = invariant.index_select(1, self.selected_indices)
        standardized = (selected - self.center[None]) / self.scale[None]
        delta = (
            standardized * self.weight.to(standardized.dtype)[None]
        ).sum(dim=1) / math.sqrt(self.feature_count)
        return delta, standardized


def max_preserving_subtype_envelope(
    logits: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Change only the subtype gap while preserving its parent envelope.

    Classes are ordered ``[axion, cdm, no_sub]``.  The winning subtype is
    assigned the original ``max(axion, cdm)`` value explicitly, and the other
    subtype is clamped below it.  This makes the parent envelope exact in the
    storage dtype (including bfloat16), rather than relying on cancellation in
    ``m - max(a + d, c - d)``.  Consequently the ``no_sub`` argmax indicator
    is identical before and after the correction, including ties under
    PyTorch's first-index argmax rule.
    """

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("Max-preserving subtype logits must have shape [B,3]")
    if delta.ndim != 1 or delta.shape[0] != logits.shape[0]:
        raise ValueError("Max-preserving subtype delta must have shape [B]")
    if not logits.is_floating_point() or not delta.is_floating_point():
        raise ValueError("Max-preserving subtype inputs must be floating point")
    delta = delta.to(device=logits.device, dtype=logits.dtype)
    if not bool(torch.isfinite(logits).all()) or not bool(
        torch.isfinite(delta).all()
    ):
        raise RuntimeError("Max-preserving subtype inputs must be finite")

    axion, cdm, no_sub = logits.unbind(dim=1)
    shifted_axion = axion + delta
    shifted_cdm = cdm - delta
    parent_envelope = torch.maximum(axion, cdm)
    axion_wins = shifted_axion >= shifted_cdm
    shifted_winner = torch.where(axion_wins, shifted_axion, shifted_cdm)
    common_shift = parent_envelope - shifted_winner
    shifted_axion = shifted_axion + common_shift
    shifted_cdm = shifted_cdm + common_shift

    # Explicit winner assignment provides an operational finite-precision
    # guarantee.  The minimum is a guard against a one-ulp overshoot in the
    # losing branch.  With delta==0, common_shift is exact zero and all three
    # original logits replay bitwise.
    adjusted_axion = torch.where(
        axion_wins,
        parent_envelope,
        torch.minimum(shifted_axion, parent_envelope),
    )
    adjusted_cdm = torch.where(
        axion_wins,
        torch.minimum(shifted_cdm, parent_envelope),
        parent_envelope,
    )
    adjusted = torch.stack((adjusted_axion, adjusted_cdm, no_sub), dim=1)
    if not torch.equal(
        adjusted[:, :2].amax(dim=1), parent_envelope
    ):  # pragma: no cover - guarded algebra, retained fail closed
        raise RuntimeError("Max-preserving subtype envelope drifted numerically")
    return adjusted


class D4OrbitClassifier(nn.Module):
    """Compact physics-conditioned D4 orbit classifier."""

    def __init__(
        self,
        num_classes: int = 3,
        heads: int = 4,
        reuploads: int = 2,
        core: Literal[
            "quantum", "classical", "hybrid", "classical-fusion"
        ] = "quantum",
        include_context: bool = False,
        dropout: float = 0.10,
        encoder_variant: Literal[
            "micro",
            "micro-stat",
            "deep-se",
            "deep-se-morph",
            "deep-se-haar-morph",
            "deep-se-mscorr",
            "eca",
            "tiny",
            "small",
        ] = "tiny",
        physics_variant: Literal["base", "radial"] = "base",
        physics_summary: Literal[
            "none",
            "moments",
            "moments-spectral",
            "moments-morphology",
            "moments-morphology-haar",
        ] = "none",
        quantum_encoding: Literal["angle", "boltzmann", "gibbs"] = "angle",
        observable_readout: Literal[
            "pair", "plaquette", "cayley-complete"
        ] = "pair",
        tied_mean_dispersion: bool = False,
        haar_subtype_residual: bool = False,
        shared_late_refinement: bool = False,
        haar_subtype_max_envelope: bool = False,
        r2_entanglers: bool = False,
        equatorial_readout: bool = False,
        meridional_readout: bool = False,
        cross_scale_reupload: bool = False,
    ) -> None:
        super().__init__()
        if physics_summary not in (
            "none",
            "moments",
            "moments-spectral",
            "moments-morphology",
            "moments-morphology-haar",
        ):
            raise ValueError(f"Unknown physics summary: {physics_summary}")
        if encoder_variant == "deep-se-mscorr":
            if include_context:
                raise ValueError(
                    "deep-se-mscorr is quantum-clean and forbids a context bypass"
                )
            if physics_summary in (
                "moments-morphology",
                "moments-morphology-haar",
            ):
                raise ValueError(
                    "deep-se-mscorr forbids post-core morphology fusion"
                )
        if encoder_variant == "deep-se-haar-morph":
            if physics_summary != "moments-morphology-haar":
                raise ValueError(
                    "deep-se-haar-morph requires moments-morphology-haar"
                )
            if include_context:
                raise ValueError(
                    "deep-se-haar-morph uses only its frozen compact morphology bypass"
                )
        elif physics_summary == "moments-morphology-haar":
            raise ValueError(
                "moments-morphology-haar requires deep-se-haar-morph"
            )
        if tied_mean_dispersion and encoder_variant != "deep-se-haar-morph":
            raise ValueError(
                "Tied mean-dispersion requires the deep-se-haar-morph candidate"
            )
        if tied_mean_dispersion and (
            heads != 4
            or reuploads != 2
            or core not in ("quantum", "classical")
            or quantum_encoding != "angle"
            or observable_readout != "pair"
        ):
            raise ValueError(
                "Tied mean-dispersion requires four heads, two reuploads, "
                "an angle-encoded pair readout, and a quantum or classical core"
            )
        if haar_subtype_residual and tied_mean_dispersion:
            raise ValueError(
                "Haar subtype residual and tied mean-dispersion are mutually exclusive"
            )
        if haar_subtype_max_envelope and not haar_subtype_residual:
            raise ValueError(
                "The max-preserving subtype envelope requires the Haar subtype residual"
            )
        if shared_late_refinement and (
            tied_mean_dispersion or haar_subtype_residual
        ):
            raise ValueError(
                "Shared late refinement, tied mean-dispersion, and Haar "
                "subtype residual are mutually exclusive"
            )
        if cross_scale_reupload and (
            num_classes != 3
            or heads != 4
            or reuploads != 2
            or core not in ("quantum", "classical")
            or include_context
            or encoder_variant != "deep-se-haar-morph"
            or physics_variant != "base"
            or physics_summary != "moments-morphology-haar"
            or quantum_encoding != "angle"
            or observable_readout != "pair"
            or tied_mean_dispersion
            or haar_subtype_residual
            or haar_subtype_max_envelope
            or shared_late_refinement
            or r2_entanglers
            or equatorial_readout
            or meridional_readout
        ):
            raise ValueError(
                "Cross-scale reupload requires the exact three-class, "
                "four-head/two-reupload quantum or classical annular-Haar "
                "architecture without another architecture extension"
            )
        if haar_subtype_residual and (
            encoder_variant != "deep-se-haar-morph"
            or physics_summary != "moments-morphology-haar"
            or include_context
            or num_classes != 3
            or heads != 4
            or reuploads != 2
            or core not in ("quantum", "classical")
            or quantum_encoding != "angle"
            or observable_readout != "pair"
        ):
            raise ValueError(
                "Haar subtype residual requires the exact three-class, "
                "four-head/two-reupload annular-Haar quantum or classical model"
            )
        if shared_late_refinement and (
            encoder_variant != "deep-se-haar-morph"
            or physics_summary != "moments-morphology-haar"
            or include_context
            or heads != 4
            or reuploads != 2
            or core not in ("quantum", "classical")
            or quantum_encoding != "angle"
            or observable_readout != "pair"
        ):
            raise ValueError(
                "Shared late refinement requires the exact four-head, "
                "two-reupload annular-Haar quantum or classical model"
            )
        if r2_entanglers and (
            num_classes != 3
            or heads != 4
            or reuploads != 2
            or core != "quantum"
            or include_context
            or encoder_variant != "deep-se-haar-morph"
            or physics_variant != "base"
            or physics_summary != "moments-morphology-haar"
            or quantum_encoding != "angle"
            or observable_readout != "pair"
            or tied_mean_dispersion
            or haar_subtype_residual
            or shared_late_refinement
            or equatorial_readout
            or meridional_readout
        ):
            raise ValueError(
                "R2 entanglers require the exact three-class, four-head/"
                "two-reupload quantum annular-Haar architecture without "
                "another architecture extension"
            )
        if equatorial_readout and (
            num_classes != 3
            or heads != 4
            or reuploads != 2
            or core != "quantum"
            or include_context
            or encoder_variant != "deep-se-haar-morph"
            or physics_variant != "base"
            or physics_summary != "moments-morphology-haar"
            or quantum_encoding != "angle"
            or observable_readout != "pair"
            or tied_mean_dispersion
            or haar_subtype_residual
            or shared_late_refinement
            or r2_entanglers
            or meridional_readout
        ):
            raise ValueError(
                "Equatorial readout requires the exact three-class, four-head/"
                "two-reupload quantum annular-Haar architecture without "
                "another architecture extension"
            )
        if meridional_readout and (
            num_classes != 3
            or heads != 4
            or reuploads != 2
            or core != "quantum"
            or include_context
            or encoder_variant != "deep-se-haar-morph"
            or physics_variant != "base"
            or physics_summary != "moments-morphology-haar"
            or quantum_encoding != "angle"
            or observable_readout != "pair"
            or tied_mean_dispersion
            or haar_subtype_residual
            or shared_late_refinement
            or r2_entanglers
            or equatorial_readout
        ):
            raise ValueError(
                "Meridional readout requires the exact three-class, four-head/"
                "two-reupload quantum annular-Haar architecture without "
                "another architecture extension"
            )
        self.heads = heads
        self.include_context = include_context
        self.physics_summary = physics_summary
        self.quantum_encoding = quantum_encoding
        self.observable_readout = observable_readout
        self.tied_mean_dispersion = bool(tied_mean_dispersion)
        self.haar_subtype_residual_enabled = bool(haar_subtype_residual)
        self.haar_subtype_max_envelope = bool(haar_subtype_max_envelope)
        self.shared_late_refinement = bool(shared_late_refinement)
        self.r2_entanglers = bool(r2_entanglers)
        self.equatorial_readout = bool(equatorial_readout)
        self.meridional_readout = bool(meridional_readout)
        self.cross_scale_reupload = bool(cross_scale_reupload)
        self.physics = PhysicsChannelBank(variant=physics_variant)
        self.encoder = CompactOrbitEncoder(
            input_channels=self.physics.output_channels,
            variant=encoder_variant,
            shared_late_refinement=self.shared_late_refinement,
        )
        moment_dim = (
            2 * self.physics.output_channels
            if physics_summary
            in (
                "moments",
                "moments-spectral",
                "moments-morphology",
                "moments-morphology-haar",
            )
            else 0
        )
        spectral_dim = 16 if physics_summary == "moments-spectral" else 0
        morphology_dim = (
            60
            if physics_summary
            in ("moments-morphology", "moments-morphology-haar")
            else 0
        )
        haar_dim = 104 if physics_summary == "moments-morphology-haar" else 0
        self.physics_summary_dim = (
            moment_dim + spectral_dim + morphology_dim + haar_dim
        )
        self.morphology_feature_dim = morphology_dim
        self.morphology_context_dim = (
            len(HAAR_MORPHOLOGY_CONTEXT_INDICES)
            if physics_summary == "moments-morphology-haar"
            else morphology_dim
        )
        self.haar_summary_dim = haar_dim
        if morphology_dim:
            self.register_buffer(
                "morphology_mean", torch.zeros(morphology_dim), persistent=True
            )
            self.register_buffer(
                "morphology_scale", torch.ones(morphology_dim), persistent=True
            )
        else:
            self.morphology_mean = None
            self.morphology_scale = None
        if haar_dim:
            self.register_buffer(
                "haar_mean", torch.zeros(haar_dim), persistent=True
            )
            self.register_buffer(
                "haar_scale", torch.ones(haar_dim), persistent=True
            )
            self.register_buffer(
                "morphology_context_indices",
                torch.tensor(
                    HAAR_MORPHOLOGY_CONTEXT_INDICES, dtype=torch.long
                ),
                persistent=True,
            )
        else:
            self.haar_mean = None
            self.haar_scale = None
            self.morphology_context_indices = None
        if self.cross_scale_reupload:
            self.register_buffer(
                "cross_scale_mean", torch.zeros(32), persistent=True
            )
            self.register_buffer(
                "cross_scale_scale", torch.ones(32), persistent=True
            )
            self.register_buffer(
                "cross_scale_walsh",
                _normalized_sylvester_hadamard_32()[:8].contiguous(),
                persistent=True,
            )
        else:
            self.cross_scale_mean = None
            self.cross_scale_scale = None
            self.cross_scale_walsh = None
        self.orbit_projection = nn.Linear(
            self.encoder.output_dim + self.physics_summary_dim, heads * 2
        )
        if self.tied_mean_dispersion:
            # One gate per projected angle.  Reusing the existing encoder
            # columns for both mean and standard deviation adds only eight
            # trainable values for the four-head candidate.  Zero is an exact
            # functional warm start of the established annular-Haar model.
            self.dispersion_gates = nn.Parameter(torch.zeros(heads * 2))
        else:
            self.register_parameter("dispersion_gates", None)
        if self.cross_scale_reupload:
            # One scalar per head controls both angle coordinates and all D4
            # views.  Exact zeros replay the established annular-Haar model.
            self.cross_scale_reupload_gates = nn.Parameter(torch.zeros(heads))
        else:
            self.register_parameter("cross_scale_reupload_gates", None)
        if core == "quantum":
            self.core: nn.Module = D4OrbitQuantumBottleneck(
                heads=heads,
                reuploads=reuploads,
                input_encoding=quantum_encoding,
                observable_readout=observable_readout,
                r2_entanglers=self.r2_entanglers,
                equatorial_readout=self.equatorial_readout,
                meridional_readout=self.meridional_readout,
            )
        elif core == "classical":
            if quantum_encoding != "angle":
                raise ValueError("Classical core requires quantum_encoding='angle'")
            self.core = ClassicalOrbitMixer(
                heads=heads,
                layers=reuploads,
                observable_readout=observable_readout,
            )
        elif core in ("hybrid", "classical-fusion"):
            architecture = (
                "quantum-classical" if core == "hybrid" else "classical-classical"
            )
            self.core = ParallelOrbitCores(
                heads=heads,
                layers=reuploads,
                architecture=architecture,
                quantum_encoding=quantum_encoding,
                observable_readout=observable_readout,
            )
        else:
            raise ValueError(f"Unknown core: {core}")
        self.core_name = core

        context_dim = self.encoder.output_dim * 3 if include_context else 0
        if include_context:
            self.context_projection = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, 64),
                nn.SiLU(inplace=True),
            )
            context_dim = 64
        else:
            self.context_projection = None
        head_input = self.core.output_dim + context_dim + self.morphology_context_dim
        # Statistical pooling spends the budget on second-order learned image
        # features and a deeper quantum core rather than a wide classical head.
        head_hidden = (
            18
            if encoder_variant in ("deep-se-morph", "deep-se-haar-morph")
            else (
                19
                if encoder_variant == "micro-stat"
                else (
                    14
                    if observable_readout == "cayley-complete"
                    else (24 if observable_readout == "plaquette" else 32)
                )
            )
        )
        if self.morphology_context_dim:
            self.head: nn.Module = MorphologyFusionHead(
                input_dim=head_input,
                invariant_dim=self.core.output_dim,
                hidden_dim=head_hidden,
                num_classes=num_classes,
                dropout=dropout,
                gauge_fixed_output=(
                    self.r2_entanglers
                    or self.equatorial_readout
                    or self.meridional_readout
                ),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(head_input),
                nn.Linear(head_input, head_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )
        self.haar_subtype_residual = (
            HaarSubtypeResidual() if self.haar_subtype_residual_enabled else None
        )

    def set_morphology_normalization(
        self, mean: torch.Tensor, scale: torch.Tensor
    ) -> None:
        """Install train-subset-only fixed normalization for morphology features."""

        if self.morphology_feature_dim == 0:
            raise RuntimeError("Model has no morphology features")
        if tuple(mean.shape) != (self.morphology_feature_dim,) or tuple(
            scale.shape
        ) != (self.morphology_feature_dim,):
            raise ValueError("Morphology normalization has the wrong shape")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("Morphology normalization must be finite")
        if not bool((scale > 0).all()):
            raise ValueError("Morphology scales must be positive")
        self.morphology_mean.copy_(mean.to(self.morphology_mean))
        self.morphology_scale.copy_(scale.to(self.morphology_scale))

    def set_haar_normalization(
        self, mean: torch.Tensor, scale: torch.Tensor
    ) -> None:
        """Install train-subset-only fixed normalization for Haar features."""

        if self.haar_summary_dim == 0:
            raise RuntimeError("Model has no annular Haar features")
        if tuple(mean.shape) != (self.haar_summary_dim,) or tuple(
            scale.shape
        ) != (self.haar_summary_dim,):
            raise ValueError("Haar normalization has the wrong shape")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("Haar normalization must be finite")
        if not bool((scale > 0).all()):
            raise ValueError("Haar scales must be positive")
        self.haar_mean.copy_(mean.to(self.haar_mean))
        self.haar_scale.copy_(scale.to(self.haar_scale))

    def set_cross_scale_normalization(
        self, mean: torch.Tensor, scale: torch.Tensor
    ) -> None:
        """Install train-subset-only normalization for the 32 CSSR features."""

        if not self.cross_scale_reupload:
            raise RuntimeError("Model has no cross-scale reupload features")
        if tuple(mean.shape) != (32,) or tuple(scale.shape) != (32,):
            raise ValueError("Cross-scale normalization has the wrong shape")
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(scale).all()
        ):
            raise ValueError("Cross-scale normalization must be finite")
        if not bool((scale > 0).all()):
            raise ValueError("Cross-scale scales must be positive")
        self.cross_scale_mean.copy_(mean.to(self.cross_scale_mean))
        self.cross_scale_scale.copy_(scale.to(self.cross_scale_scale))

    def _orbit_encode_with_context(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        views = d4_views(x)
        batch, group, channels, height, width = views.shape
        flat = views.reshape(batch * group, channels, height, width)
        physics = self.physics(flat)
        cross_scale_summary = cross_scale_tokens = None
        if self.cross_scale_reupload:
            cross_scale_flat = cross_scale_scattering_summary(physics)
            standardized = (
                cross_scale_flat - self.cross_scale_mean[None]
            ) / self.cross_scale_scale[None]
            token_flat = F.linear(standardized, self.cross_scale_walsh)
            cross_scale_summary = cross_scale_flat.reshape(batch, group, 32)
            cross_scale_tokens = token_flat.reshape(
                batch, group, self.heads, 2
            ).permute(0, 2, 3, 1)
        encoded_std_flat = None
        if self.tied_mean_dispersion:
            encoded_flat, encoded_std_flat = self.encoder.forward_mean_and_std(
                physics
            )
        else:
            encoded_flat = self.encoder(physics)
        projection_features = [encoded_flat]
        if self.physics_summary in (
            "moments",
            "moments-spectral",
            "moments-morphology",
            "moments-morphology-haar",
        ):
            # Per-view global moments expose stable morphology directly to the
            # angle map.  They have no trainable parameters and still transform
            # by regular permutation because they are computed independently
            # and identically for every D4 view.
            mean = physics.mean(dim=(-2, -1))
            std = physics.std(dim=(-2, -1), unbiased=False)
            projection_features.extend((mean, std))
        if self.physics_summary == "moments-spectral":
            physics_views = physics.reshape(
                batch, group, self.physics.output_channels, height, width
            )
            spectral = spectral_morphology_summary(physics_views[:, 0])
            spectral_flat = spectral[:, None, :].expand(-1, group, -1).reshape(
                batch * group, -1
            )
            projection_features.append(spectral_flat)
        morphology_context = None
        if self.physics_summary in (
            "moments-morphology",
            "moments-morphology-haar",
        ):
            physics_views = physics.reshape(
                batch, group, self.physics.output_channels, height, width
            )
            morphology = lens_morphology_summary(physics_views[:, 0])
            morphology = (
                morphology - self.morphology_mean[None]
            ) / self.morphology_scale[None]
            morphology_flat = morphology[:, None, :].expand(
                -1, group, -1
            ).reshape(batch * group, -1)
            projection_features.append(morphology_flat)
            if self.morphology_context_indices is None:
                morphology_context = morphology
            else:
                morphology_context = morphology.index_select(
                    1, self.morphology_context_indices
                )
        haar_summary = None
        if self.physics_summary == "moments-morphology-haar":
            haar_flat = annular_haar_scattering_summary(physics)
            haar_flat = (
                haar_flat - self.haar_mean[None]
            ) / self.haar_scale[None]
            projection_features.append(haar_flat)
            haar_summary = haar_flat.reshape(batch, group, -1)
        projection_features = torch.cat(projection_features, dim=1)
        encoded = encoded_flat.reshape(batch, group, -1)
        projected = self.orbit_projection(projection_features)
        if encoded_std_flat is not None:
            # Tie dispersion to the learned mean-column directions rather than
            # allocating a second 192x8 matrix.  This term affects only the
            # quantum/classical-core angles; ``encoded`` and every classifier
            # context path continue to use the established mean descriptor.
            mean_columns = self.orbit_projection.weight[
                :, : self.encoder.output_dim
            ]
            dispersion = F.linear(encoded_std_flat, mean_columns, bias=None)
            # Keep the zero extension in the projection's autocast dtype.  A
            # float32 gate multiplied by a bfloat16 projection would promote
            # the addition and break bitwise replay despite a numerical zero.
            dispersion_term = self.dispersion_gates.to(projected.dtype) * (
                dispersion.to(projected.dtype)
            )
            projected = projected + dispersion_term
        projected = projected.reshape(batch, group, -1)
        projected = projected.reshape(batch, group, self.heads, 2).permute(0, 2, 3, 1)
        angles = math.pi * torch.tanh(projected)
        return (
            encoded,
            angles,
            morphology_context,
            haar_summary,
            cross_scale_summary,
            cross_scale_tokens,
        )

    def orbit_encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return learned orbit embeddings and the quantum angle tensor."""

        encoded, angles, _, _, _, _ = self._orbit_encode_with_context(x)
        return encoded, angles

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        (
            encoded,
            angles,
            morphology_context,
            haar_summary,
            cross_scale_summary,
            cross_scale_tokens,
        ) = (
            self._orbit_encode_with_context(x)
        )
        context_embedding = None
        if self.context_projection is not None:
            context = torch.cat(
                (
                    encoded.mean(dim=1),
                    encoded.std(dim=1, unbiased=False),
                    encoded.amax(dim=1),
                ),
                dim=1,
            )
            context_embedding = self.context_projection(context)

        def classify(invariant_features: torch.Tensor) -> torch.Tensor:
            features = [invariant_features]
            if context_embedding is not None:
                features.append(context_embedding)
            if morphology_context is not None:
                features.append(morphology_context)
            return self.head(torch.cat(features, dim=1))

        branch_invariants = branch_logits = mixing_weight = None
        base_invariant = perturbed_invariant = cross_scale_delta = None
        if self.core_name in ("hybrid", "classical-fusion"):
            if return_aux and self.core_name == "hybrid":
                branch_a, equivariant = self.core.branch_a(
                    angles, return_equivariant=True
                )
                branch_b = self.core.branch_b(angles)
            else:
                branch_a, branch_b = self.core(angles)
                equivariant = None
            logits_a, logits_b = classify(branch_a), classify(branch_b)
            mixing_weight = self.core.mixing_weight
            logits = mixing_weight * logits_a + (1.0 - mixing_weight) * logits_b
            invariant = mixing_weight * branch_a + (1.0 - mixing_weight) * branch_b
            branch_invariants = (branch_a, branch_b)
            branch_logits = (logits_a, logits_b)
        else:
            if self.cross_scale_reupload:
                if (
                    cross_scale_tokens is None
                    or self.cross_scale_reupload_gates is None
                ):
                    raise RuntimeError("Missing cross-scale reupload state")
                tokens = cross_scale_tokens.to(dtype=angles.dtype)
                gates = torch.tanh(
                    self.cross_scale_reupload_gates.to(dtype=angles.dtype)
                )[None, :, None, None]
                cross_scale_delta = (
                    (math.pi / 4.0) * gates * torch.tanh(tokens)
                )
                if return_aux and self.core_name == "quantum":
                    base_invariant, equivariant = self.core(
                        angles, return_equivariant=True
                    )
                else:
                    base_invariant = self.core(angles)
                    equivariant = None
                perturbed_invariant = self.core(angles + cross_scale_delta)
                invariant = base_invariant + 0.5 * (
                    perturbed_invariant - base_invariant
                )
            elif return_aux and self.core_name == "quantum":
                invariant, equivariant = self.core(angles, return_equivariant=True)
            else:
                invariant = self.core(angles)
                equivariant = None
            logits = classify(invariant)
        haar_subtype_delta = haar_subtype_features = None
        haar_subtype_base_logits = None
        if self.haar_subtype_residual is not None:
            if haar_summary is None:
                raise RuntimeError("Haar subtype residual requires Haar summaries")
            haar_subtype_delta, haar_subtype_features = (
                self.haar_subtype_residual(haar_summary)
            )
            delta = haar_subtype_delta.to(logits.dtype)
            haar_subtype_base_logits = logits
            if self.haar_subtype_max_envelope:
                logits = max_preserving_subtype_envelope(logits, delta)
            else:
                logits = logits + torch.stack(
                    (delta, -delta, torch.zeros_like(delta)), dim=1
                )
        if return_aux:
            return logits, {
                "encoded": encoded,
                "angles": angles,
                "invariants": invariant,
                "equivariant": equivariant,
                "branch_invariants": branch_invariants,
                "branch_logits": branch_logits,
                "mixing_weight": mixing_weight,
                "morphology_context": morphology_context,
                "haar_summary": haar_summary,
                "haar_subtype_delta": haar_subtype_delta,
                "haar_subtype_features": haar_subtype_features,
                "haar_subtype_base_logits": haar_subtype_base_logits,
                "cross_scale_summary": cross_scale_summary,
                "cross_scale_tokens": cross_scale_tokens,
                "cross_scale_delta": cross_scale_delta,
                "base_invariants": base_invariant,
                "perturbed_invariants": perturbed_invariant,
            }
        return logits

    def parameter_report(self) -> Dict[str, int | str]:
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        report = {
            "total": count(self),
            "physics": count(self.physics),
            "encoder": count(self.encoder),
            "orbit_projection": count(self.orbit_projection),
            "core": count(self.core),
            "head_and_context": count(self.head)
            + (count(self.context_projection) if self.context_projection is not None else 0),
        }
        if self.core_name == "quantum":
            report["quantum"] = report["core"]
            report["parallel_classical"] = 0
            report["mixture_trainable"] = 0
        elif self.core_name == "hybrid":
            report["quantum"] = count(self.core.branch_a)
            report["parallel_classical"] = count(self.core.branch_b)
            report["mixture_trainable"] = self.core.mix_logit.numel()
        elif self.core_name == "classical-fusion":
            report["quantum"] = 0
            report["parallel_classical"] = count(self.core.branch_a) + count(
                self.core.branch_b
            )
            report["mixture_trainable"] = self.core.mix_logit.numel()
        else:
            report["quantum"] = 0
            report["parallel_classical"] = report["core"]
            report["mixture_trainable"] = 0
        report["encoder_variant"] = self.encoder.variant
        report["encoder_output_dim"] = self.encoder.output_dim
        report["encoder_final_channels"] = self.encoder.final[0].out_channels
        report["encoder_multiscale_correlation_pool"] = bool(
            self.encoder.multiscale_correlation_pool
        )
        report["physics_variant"] = self.physics.variant
        report["physics_summary"] = self.physics_summary
        report["physics_summary_dim"] = self.physics_summary_dim
        report["morphology_feature_dim"] = self.morphology_feature_dim
        report["morphology_context_dim"] = self.morphology_context_dim
        report["haar_summary_dim"] = self.haar_summary_dim
        report["tied_mean_dispersion"] = self.tied_mean_dispersion
        report["dispersion_gate_trainable"] = (
            self.dispersion_gates.numel()
            if self.dispersion_gates is not None
            else 0
        )
        report["haar_subtype_residual"] = self.haar_subtype_residual_enabled
        report["haar_subtype_max_envelope"] = self.haar_subtype_max_envelope
        report["haar_subtype_residual_trainable"] = (
            count(self.haar_subtype_residual)
            if self.haar_subtype_residual is not None
            else 0
        )
        report["shared_late_refinement"] = self.shared_late_refinement
        report["shared_late_refinement_gate_trainable"] = (
            self.encoder.shared_refinement_gates.numel()
            if self.encoder.shared_refinement_gates is not None
            else 0
        )
        report["cross_scale_reupload"] = self.cross_scale_reupload
        report["cross_scale_reupload_gate_trainable"] = (
            self.cross_scale_reupload_gates.numel()
            if self.cross_scale_reupload_gates is not None
            else 0
        )
        report["cross_scale_scattering_dim"] = (
            32 if self.cross_scale_reupload else 0
        )
        report["cross_scale_walsh_channels"] = (
            8 if self.cross_scale_reupload else 0
        )
        r2_params = getattr(self.core, "r2_params", None)
        report["r2_entanglers"] = self.r2_entanglers
        report["r2_entangler_trainable"] = (
            r2_params.numel() if r2_params is not None else 0
        )
        readout_phases = getattr(self.core, "readout_phases", None)
        report["equatorial_readout"] = self.equatorial_readout
        report["equatorial_readout_trainable"] = (
            readout_phases.numel() if readout_phases is not None else 0
        )
        meridional_phases = getattr(self.core, "meridional_phases", None)
        report["meridional_readout"] = self.meridional_readout
        report["meridional_readout_trainable"] = (
            meridional_phases.numel() if meridional_phases is not None else 0
        )
        report["quantum_state_preparation_trainable"] = (
            report["quantum"]
            - report["equatorial_readout_trainable"]
            - report["meridional_readout_trainable"]
            if self.core_name == "quantum"
            else 0
        )
        output_classifier = (
            self.head.classifier
            if isinstance(self.head, MorphologyFusionHead)
            else self.head[-1]
        )
        classifier_bias = getattr(output_classifier, "bias", None)
        report["classifier_bias_trainable"] = (
            classifier_bias.numel() if classifier_bias is not None else 0
        )
        report["classifier_bias_gauge_degrees"] = (
            2
            if (
                self.r2_entanglers
                or self.equatorial_readout
                or self.meridional_readout
            )
            else report["classifier_bias_trainable"]
        )
        report["quantum_encoding"] = self.quantum_encoding
        report["observable_readout"] = self.observable_readout
        report["core_architecture"] = self.core_name
        return report
