"""Standalone CPU inference definition for the winning seed-2 D4-ORQB.

This temporary source is intentionally restricted to the checkpoint's exact
base/tiny/angle/pair path.  It has no dependency on the repository package and
preserves every persistent ``state_dict`` key used by the 245,221-parameter
checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_CHECKPOINT = Path(
    "/home/jovyan/susmered-datavol-1/outputs/deeplense-quantum/"
    "d4-orqb/model-i/fixed-backbone-replication/seed-2/quantum/best.pt"
)
CLASS_NAMES = ("axion", "cdm", "no_sub")

R_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 3),
    (1, 2),
    (2, 3),
    (4, 5),
    (4, 7),
    (5, 6),
    (6, 7),
)
R2_EDGES: tuple[tuple[int, int], ...] = ((0, 2), (1, 3), (4, 6), (5, 7))
S_EDGES: tuple[tuple[int, int], ...] = ((0, 4), (1, 7), (2, 6), (3, 5))


def norm2d(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def d4_transform(
    images: torch.Tensor, rotation: int, reflected: int
) -> torch.Tensor:
    if reflected:
        images = torch.flip(images, dims=(-1,))
    return torch.rot90(images, rotation, dims=(-2, -1))


def d4_views(images: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [d4_transform(images, rotation, reflected) for reflected in (0, 1) for rotation in range(4)],
        dim=1,
    )


class PhysicsChannelBank(nn.Module):
    """The exact zero-parameter eight-channel base physics bank."""

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
        self.output_channels = 8
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

        gradient_x = self._conv_reflect(log_intensity, self.sobel_x)
        gradient_y = self._conv_reflect(log_intensity, self.sobel_y)
        pixel_scale = float(images.shape[-1]) / self.reference_pixels
        gradient = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1e-8
        ) * pixel_scale
        laplacian = (
            self._conv_reflect(log_intensity, self.laplacian).abs()
            * pixel_scale**2
        )
        small_kernel = max(3, int(round(3 * pixel_scale)) | 1)
        large_kernel = max(
            small_kernel + 2, int(round(9 * pixel_scale)) | 1
        )
        difference_of_averages = (
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
        radial = (
            gradient_x * unit_x + gradient_y * unit_y
        ).abs() * pixel_scale
        tangential = (
            -gradient_x * unit_y + gradient_y * unit_x
        ).abs() * pixel_scale

        log_ratio_squared = torch.log(
            (1.0 + self.epsilon) / (images + self.epsilon)
        ).square()
        mixed = self._conv_reflect(
            self._conv_reflect(log_ratio_squared, self.sobel_x), self.sobel_y
        ).abs() * pixel_scale**2
        return torch.cat(
            (
                images,
                log_intensity,
                torch.tanh(2.0 * gradient),
                torch.tanh(laplacian),
                torch.tanh(4.0 * difference_of_averages),
                torch.tanh(2.0 * radial),
                torch.tanh(2.0 * tangential),
                torch.tanh(mixed),
            ),
            dim=1,
        )


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


class TinyOrbitEncoder(nn.Module):
    """The exact tiny encoder topology and checkpoint module names."""

    output_dim = 128

    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, 16, 5, stride=2, padding=2, bias=False
            ),
            norm2d(16),
            nn.SiLU(inplace=True),
        )
        specifications = (
            (16, 24, 2, 2),
            (24, 24, 2, 1),
            (24, 40, 3, 2),
            (40, 40, 3, 1),
            (40, 64, 3, 2),
            (64, 64, 3, 1),
            (64, 96, 3, 2),
            (96, 96, 2, 1),
        )
        self.blocks = nn.Sequential(
            *(
                MBConv(in_channels, out_channels, expand, stride)
                for in_channels, out_channels, expand, stride in specifications
            )
        )
        self.final = nn.Sequential(
            nn.Conv2d(96, self.output_dim, 1, bias=False),
            norm2d(self.output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.stem(features)
        features = self.blocks(features)
        features = self.final(features)
        return features.mean(dim=(-2, -1))


def _bit_mask(qubit: int, n_qubits: int) -> int:
    return 1 << (n_qubits - qubit - 1)


def _apply_ry(
    state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int
) -> torch.Tensor:
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    view = state.reshape(state.shape[0], left, 2, right)
    amplitude_0, amplitude_1 = view[:, :, 0, :], view[:, :, 1, :]
    cosine = torch.cos(theta / 2).view(-1, 1, 1)
    sine = torch.sin(theta / 2).view(-1, 1, 1)
    output = torch.stack(
        (
            cosine * amplitude_0 - sine * amplitude_1,
            sine * amplitude_0 + cosine * amplitude_1,
        ),
        dim=2,
    )
    return output.reshape_as(state)


def _apply_rx(
    state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int
) -> torch.Tensor:
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    view = state.reshape(state.shape[0], left, 2, right)
    amplitude_0, amplitude_1 = view[:, :, 0, :], view[:, :, 1, :]
    cosine = torch.cos(theta / 2).view(-1, 1, 1)
    sine = (-1j * torch.sin(theta / 2)).view(-1, 1, 1)
    output = torch.stack(
        (
            cosine * amplitude_0 + sine * amplitude_1,
            sine * amplitude_0 + cosine * amplitude_1,
        ),
        dim=2,
    )
    return output.reshape_as(state)


def _apply_rz(
    state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int
) -> torch.Tensor:
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    view = state.reshape(state.shape[0], left, 2, right)
    phase_0 = torch.exp(-0.5j * theta).view(-1, 1, 1)
    phase_1 = torch.exp(0.5j * theta).view(-1, 1, 1)
    output = torch.stack(
        (phase_0 * view[:, :, 0, :], phase_1 * view[:, :, 1, :]), dim=2
    )
    return output.reshape_as(state)


def _apply_pair_rotation(
    state: torch.Tensor,
    theta: torch.Tensor,
    qubit_0: int,
    qubit_1: int,
    pauli: str,
    z_signs: torch.Tensor,
) -> torch.Tensor:
    cosine = torch.cos(theta / 2).view(-1, 1)
    sine = (-1j * torch.sin(theta / 2)).view(-1, 1)
    if pauli == "z":
        transformed = state * (
            z_signs[qubit_0] * z_signs[qubit_1]
        ).view(1, -1)
    elif pauli == "x":
        mask = _bit_mask(qubit_0, z_signs.shape[0]) | _bit_mask(
            qubit_1, z_signs.shape[0]
        )
        permutation = torch.arange(state.shape[1], device=state.device) ^ mask
        transformed = state.index_select(1, permutation)
    else:
        raise ValueError(f"Unsupported Pauli operator: {pauli}")
    return cosine * state + sine * transformed


def _edge_expectation_z(
    probabilities: torch.Tensor,
    z_signs: torch.Tensor,
    edges: Sequence[tuple[int, int]],
) -> torch.Tensor:
    observables = torch.stack([z_signs[a] * z_signs[b] for a, b in edges])
    return probabilities @ observables.transpose(0, 1)


def _expectation_x(
    state: torch.Tensor,
    qubits: Sequence[tuple[int, ...]],
    n_qubits: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    basis = torch.arange(state.shape[1], device=state.device)
    for wires in qubits:
        mask = 0
        for wire in wires:
            mask |= _bit_mask(wire, n_qubits)
        flipped = state.index_select(1, basis ^ mask)
        values.append((state.conj() * flipped).sum(dim=1).real)
    return torch.stack(values, dim=1)


class D4OrbitQuantumBottleneck(nn.Module):
    """The exact four-head/two-reupload angle/pair quantum core."""

    parameters_per_layer = 11
    invariants_per_head = 12

    def __init__(self) -> None:
        super().__init__()
        self.heads = 4
        self.reuploads = 2
        self.n_qubits = 8
        parameters = torch.zeros(
            self.heads, self.reuploads, self.parameters_per_layer
        )
        parameters[..., 0] = 1.0
        parameters[..., 2] = 1.0
        parameters[..., 4:] = 0.02 * torch.randn_like(parameters[..., 4:])
        self.params = nn.Parameter(parameters)

        basis = torch.arange(1 << self.n_qubits)
        signs = []
        for qubit in range(self.n_qubits):
            bit = (basis & _bit_mask(qubit, self.n_qubits)) != 0
            signs.append(
                torch.where(
                    bit, -torch.ones_like(basis), torch.ones_like(basis)
                )
            )
        self.register_buffer(
            "z_signs", torch.stack(signs).float(), persistent=False
        )
        self.register_buffer(
            "r_edges", torch.tensor(R_EDGES, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "r2_edges",
            torch.tensor(R2_EDGES, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "s_edges", torch.tensor(S_EDGES, dtype=torch.long), persistent=False
        )

    @property
    def output_dim(self) -> int:
        return self.heads * self.invariants_per_head

    def _edge_rotation(
        self,
        state: torch.Tensor,
        theta: torch.Tensor,
        edges: torch.Tensor,
        pauli: str,
    ) -> torch.Tensor:
        for qubit_0, qubit_1 in edges.tolist():
            state = _apply_pair_rotation(
                state,
                theta,
                qubit_0,
                qubit_1,
                pauli,
                self.z_signs,
            )
        return state

    def _run_statevector(self, orbit_features: torch.Tensor) -> torch.Tensor:
        batch = orbit_features.shape[0]
        features = orbit_features.reshape(
            batch * self.heads, 2, self.n_qubits
        ).float()
        parameters = self.params.unsqueeze(0).expand(batch, -1, -1, -1)
        parameters = parameters.reshape(
            batch * self.heads, self.reuploads, self.parameters_per_layer
        ).float()
        state = torch.zeros(
            features.shape[0],
            1 << self.n_qubits,
            dtype=torch.complex64,
            device=features.device,
        )
        state[:, 0] = 1.0 + 0.0j

        for layer in range(self.reuploads):
            layer_parameters = parameters[:, layer]
            for qubit in range(self.n_qubits):
                ry_angle = (
                    layer_parameters[:, 0] * features[:, 0, qubit]
                    + layer_parameters[:, 1]
                )
                rz_angle = (
                    layer_parameters[:, 2] * features[:, 1, qubit]
                    + layer_parameters[:, 3]
                )
                state = _apply_ry(state, ry_angle, qubit, self.n_qubits)
                state = _apply_rz(state, rz_angle, qubit, self.n_qubits)
            for qubit in range(self.n_qubits):
                state = _apply_rx(
                    state, layer_parameters[:, 4], qubit, self.n_qubits
                )
                state = _apply_ry(
                    state, layer_parameters[:, 5], qubit, self.n_qubits
                )
                state = _apply_rz(
                    state, layer_parameters[:, 6], qubit, self.n_qubits
                )
            state = self._edge_rotation(
                state, layer_parameters[:, 7], self.r_edges, "z"
            )
            state = self._edge_rotation(
                state, layer_parameters[:, 8], self.s_edges, "z"
            )
            state = self._edge_rotation(
                state, layer_parameters[:, 9], self.r_edges, "x"
            )
            state = self._edge_rotation(
                state, layer_parameters[:, 10], self.s_edges, "x"
            )
        return state

    def forward(
        self, orbit_features: torch.Tensor, return_equivariant: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_shape = (self.heads, 2, self.n_qubits)
        if orbit_features.ndim != 4 or orbit_features.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected (B,{self.heads},2,{self.n_qubits}), "
                f"got {tuple(orbit_features.shape)}"
            )
        with torch.autocast(
            device_type=orbit_features.device.type, enabled=False
        ):
            state = self._run_statevector(orbit_features)
            probabilities = state.abs().square()
            z_values = probabilities @ self.z_signs.transpose(0, 1)
            x_values = _expectation_x(
                state,
                [(qubit,) for qubit in range(self.n_qubits)],
                self.n_qubits,
            )
            edge_families = (R_EDGES, R2_EDGES, S_EDGES)
            zz_values = tuple(
                _edge_expectation_z(probabilities, self.z_signs, edges)
                for edges in edge_families
            )
            xx_values = tuple(
                _expectation_x(state, edges, self.n_qubits)
                for edges in edge_families
            )

            def edge_products(
                values: torch.Tensor, edges: Sequence[tuple[int, int]]
            ) -> torch.Tensor:
                return torch.stack(
                    [values[:, a] * values[:, b] for a, b in edges], dim=1
                )

            z_mean = z_values.mean(dim=1)
            x_mean = x_values.mean(dim=1)
            invariants = torch.stack(
                (
                    z_mean,
                    z_values.square().mean(dim=1) - z_mean.square(),
                    x_mean,
                    x_values.square().mean(dim=1) - x_mean.square(),
                    *(values.mean(dim=1) for values in zz_values),
                    *(values.mean(dim=1) for values in xx_values),
                    (
                        zz_values[0] - edge_products(z_values, R_EDGES)
                    ).mean(dim=1),
                    (
                        xx_values[0] - edge_products(x_values, R_EDGES)
                    ).mean(dim=1),
                ),
                dim=1,
            ).reshape(orbit_features.shape[0], self.output_dim)
        if not return_equivariant:
            return invariants
        return invariants, {
            "z": z_values.reshape(
                orbit_features.shape[0], self.heads, self.n_qubits
            ),
            "x": x_values.reshape(
                orbit_features.shape[0], self.heads, self.n_qubits
            ),
        }


class BestD4ORQB(nn.Module):
    """Exact 245,221-parameter seed-2 base/tiny D4-ORQB architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.physics = PhysicsChannelBank()
        self.encoder = TinyOrbitEncoder(input_channels=8)
        self.orbit_projection = nn.Linear(128, 8)
        self.core = D4OrbitQuantumBottleneck()
        self.head = nn.Sequential(
            nn.LayerNorm(48),
            nn.Linear(48, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(32, 3),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def orbit_encode(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        views = d4_views(images)
        batch, group, channels, height, width = views.shape
        flattened_views = views.reshape(
            batch * group, channels, height, width
        )
        physics = self.physics(flattened_views)
        encoded_flat = self.encoder(physics)
        encoded = encoded_flat.reshape(batch, group, 128)
        projected = self.orbit_projection(encoded_flat)
        projected = projected.reshape(batch, group, 4, 2).permute(0, 2, 3, 1)
        angles = math.pi * torch.tanh(projected)
        return encoded, angles

    def forward(
        self, images: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        encoded, angles = self.orbit_encode(images)
        if return_aux:
            invariants, equivariant = self.core(
                angles, return_equivariant=True
            )
        else:
            invariants = self.core(angles)
            equivariant = None
        logits = self.head(invariants)
        if not return_aux:
            return logits
        return logits, {
            "encoded": encoded,
            "angles": angles,
            "invariants": invariants,
            "equivariant": equivariant,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["BestD4ORQB", dict[str, Any]]:
        checkpoint_path = Path(checkpoint_path)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:  # Compatibility with older PyTorch releases.
            payload = torch.load(checkpoint_path, map_location=map_location)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(
                "Expected a checkpoint wrapper containing the 'model' state_dict"
            )
        model = cls()
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        return model, payload


def _self_test(checkpoint_path: str | Path) -> None:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, payload = BestD4ORQB.from_checkpoint(checkpoint_path)
    assert model.parameter_count == 245_221, model.parameter_count
    assert tuple(model.core.params.shape) == (4, 2, 11)
    image = torch.linspace(0.0, 1.0, 96 * 96, dtype=torch.float32).reshape(
        1, 1, 96, 96
    )
    with torch.inference_mode():
        logits, auxiliary = model(image, return_aux=True)
    assert tuple(logits.shape) == (1, 3)
    assert tuple(auxiliary["angles"].shape) == (1, 4, 2, 8)
    assert tuple(auxiliary["invariants"].shape) == (1, 48)
    assert bool(torch.isfinite(logits).all())
    print(
        {
            "status": "passed",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "parameters": model.parameter_count,
            "logits_shape": list(logits.shape),
            "angles_shape": list(auxiliary["angles"].shape),
            "invariants_shape": list(auxiliary["invariants"].shape),
            "probabilities": torch.softmax(logits, dim=1).tolist(),
        }
    )


if __name__ == "__main__":
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    _self_test(checkpoint)
