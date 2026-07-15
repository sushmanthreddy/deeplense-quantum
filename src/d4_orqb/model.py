"""Composition of the selected shared encoder and D4 orbit bottlenecks."""

from __future__ import annotations

import math
from typing import Dict, Literal, Sequence, Tuple

import torch
from torch import nn

from .config import Config
from .encoder import (
    CompactOrbitEncoder,
    ForegroundSuppressedMorphologyChannelBank,
    ModelIVPhysicsSummary,
    MorphologyChannelBank,
    d4_views,
)
from .quantum import D4OrbitQuantumBottleneck, R2_EDGES, R_EDGES, S_EDGES


def _edge_products(
    values: torch.Tensor, edges: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    return torch.stack(
        [values[..., first] * values[..., second] for first, second in edges],
        dim=-1,
    )


class ClassicalOrbitMixer(nn.Module):
    """Parameter-matched classical scaffold used only for backbone pretraining."""

    invariants_per_head = 12

    def __init__(self, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.layers = layers
        parameters = torch.zeros(heads, layers, 11)
        parameters[..., 0] = 1.0
        parameters[..., 6] = 1.0
        parameters[..., 10] = 1.0
        parameters += 0.02 * torch.randn_like(parameters)
        self.params = nn.Parameter(parameters)

    @property
    def output_dim(self) -> int:
        return self.heads * self.invariants_per_head

    def forward(self, orbit_features: torch.Tensor) -> torch.Tensor:
        first, second = orbit_features[:, :, 0], orbit_features[:, :, 1]
        for layer in range(self.layers):
            parameters = self.params[:, layer].unsqueeze(0)
            new_first = torch.tanh(
                parameters[..., 0, None] * first
                + parameters[..., 1, None] * second
                + parameters[..., 2, None]
                + parameters[..., 3, None] * torch.sin(first)
                + parameters[..., 4, None] * torch.cos(second)
            )
            new_second = torch.tanh(
                parameters[..., 5, None] * first
                + parameters[..., 6, None] * second
                + parameters[..., 7, None]
                + parameters[..., 8, None] * torch.sin(second)
                + parameters[..., 9, None] * torch.cos(first)
            )
            residual = torch.sigmoid(parameters[..., 10, None])
            first = residual * first + (1.0 - residual) * new_first
            second = residual * second + (1.0 - residual) * new_second

        first_mean = first.mean(-1)
        second_mean = second.mean(-1)
        invariant_features = [
            first_mean,
            first.square().mean(-1) - first_mean.square(),
            second_mean,
            second.square().mean(-1) - second_mean.square(),
            _edge_products(first, R_EDGES).mean(-1),
            _edge_products(first, R2_EDGES).mean(-1),
            _edge_products(first, S_EDGES).mean(-1),
            _edge_products(second, R_EDGES).mean(-1),
            _edge_products(second, R2_EDGES).mean(-1),
            _edge_products(second, S_EDGES).mean(-1),
            (
                _edge_products(first, R_EDGES)
                - first_mean[..., None].square()
            ).mean(-1),
            (
                _edge_products(second, R_EDGES)
                - second_mean[..., None].square()
            ).mean(-1),
        ]
        features = torch.stack(invariant_features, dim=-1)
        return features.reshape(orbit_features.shape[0], self.output_dim)


class D4OrbitClassifier(nn.Module):
    """Selected eight-view D4-ORQB Model-I classifier."""

    def __init__(
        self,
        num_classes: int = 3,
        heads: int = 4,
        reuploads: int = 2,
        core: Literal["quantum", "classical"] = "quantum",
        include_context: bool = False,
        dropout: float = 0.10,
        foreground_suppressed: bool = False,
    ) -> None:
        super().__init__()
        if num_classes != 3:
            raise ValueError("The selected Model-I classifier has three classes")
        self.heads = heads
        self.include_context = include_context
        self.core_name = core

        # These remain top-level modules so the historical backbone-prefix
        # initialization contract stays stable.
        self.physics = (
            ForegroundSuppressedMorphologyChannelBank()
            if foreground_suppressed
            else MorphologyChannelBank()
        )
        if foreground_suppressed:
            self.physics_summary: nn.Module | None = ModelIVPhysicsSummary()
            self.physics_summary_norm: nn.Module | None = nn.LayerNorm(
                ModelIVPhysicsSummary.output_dim,
                elementwise_affine=False,
            )
            self.physics_summary_head: nn.Module | None = nn.Linear(
                ModelIVPhysicsSummary.output_dim, num_classes
            )
            nn.init.zeros_(self.physics_summary_head.weight)
            nn.init.zeros_(self.physics_summary_head.bias)
        else:
            self.physics_summary = None
            self.physics_summary_norm = None
            self.physics_summary_head = None
        self.encoder = CompactOrbitEncoder(
            input_channels=self.physics.output_channels
        )
        self.orbit_projection = nn.Linear(self.encoder.output_dim, heads * 2)

        if core == "quantum":
            self.core: nn.Module = D4OrbitQuantumBottleneck(
                heads=heads, reuploads=reuploads
            )
        elif core == "classical":
            self.core = ClassicalOrbitMixer(heads=heads, layers=reuploads)
        else:
            raise ValueError(f"Unknown core: {core}")

        context_dim = self.encoder.output_dim * 3 if include_context else 0
        if include_context:
            self.context_projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, 64),
                nn.SiLU(inplace=True),
            )
            context_dim = 64
        else:
            self.context_projection = None

        head_input = self.core.output_dim + context_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def orbit_encode(
        self,
        images: torch.Tensor,
        morphology: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return learned orbit embeddings and circuit angle features."""

        if self.physics.variant.startswith("model_iv_"):
            # The deterministic bank is D4-equivariant, so lifting its output
            # is exactly equivalent and avoids evaluating it eight times.
            if morphology is None:
                morphology = self.physics(images)
            morphology_views = d4_views(morphology)
            batch, group, channels, height, width = morphology_views.shape
            morphology = morphology_views.reshape(
                batch * group, channels, height, width
            )
        else:
            views = d4_views(images)
            batch, group, channels, height, width = views.shape
            flat_views = views.reshape(batch * group, channels, height, width)
            morphology = self.physics(flat_views)
        flat_encoded = self.encoder(morphology)
        encoded = flat_encoded.reshape(batch, group, -1)
        projected = self.orbit_projection(flat_encoded)
        projected = projected.reshape(batch, group, self.heads, 2).permute(
            0, 2, 3, 1
        )
        angles = math.pi * torch.tanh(projected)
        return encoded, angles

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        # Canonicalize D4-transformed views before deterministic physics and
        # convolution kernels. Odd torch.rot90 actions otherwise retain a
        # transposed stride layout that can select numerically different CUDA
        # kernels and amplify roundoff after feature standardization. Plain
        # contiguous format is unambiguous for the singleton input channel.
        images = images.contiguous()
        morphology = None
        summary = None
        if self.physics_summary is not None:
            morphology = self.physics(images)
            summary = self.physics_summary(morphology)
        encoded, angles = self.orbit_encode(images, morphology=morphology)
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

        if return_aux and self.core_name == "quantum":
            invariants, equivariant = self.core(
                angles, return_equivariant=True
            )
        else:
            invariants = self.core(angles)
            equivariant = None
        features = [invariants]
        if context_embedding is not None:
            features.append(context_embedding)
        logits = self.head(torch.cat(features, dim=1))
        summary_logits = None
        if summary is not None:
            assert self.physics_summary_norm is not None
            assert self.physics_summary_head is not None
            summary_logits = self.physics_summary_head(
                self.physics_summary_norm(summary)
            )
            logits = logits + summary_logits

        if return_aux:
            return logits, {
                "encoded": encoded,
                "angles": angles,
                "invariants": invariants,
                "equivariant": equivariant,
                "physics_summary": summary,
                "physics_summary_logits": summary_logits,
            }
        return logits

    def parameter_report(self) -> Dict[str, int | str]:
        def count(module: nn.Module) -> int:
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )

        context = (
            count(self.context_projection)
            if self.context_projection is not None
            else 0
        )
        summary_parameters = (
            count(self.physics_summary_head)
            if self.physics_summary_head is not None
            else 0
        )
        return {
            "total": count(self),
            "morphology_channels": count(self.physics),
            "morphology_variant": self.physics.variant,
            "physics_summary_dim": (
                ModelIVPhysicsSummary.output_dim
                if self.physics_summary is not None
                else 0
            ),
            "physics_summary_head": summary_parameters,
            "encoder": count(self.encoder),
            "orbit_projection": count(self.orbit_projection),
            "core": count(self.core),
            "head_and_context": count(self.head) + context,
            "core_architecture": self.core_name,
            "encoder_variant": "tiny",
            "encoder_output_dim": self.encoder.output_dim,
            "input_channels": self.physics.output_channels,
            "quantum_encoding": "angle",
            "observable_readout": "pair",
            "execution_backend": (
                "torchquantum" if self.core_name == "quantum" else "classical"
            ),
        }


def build_model(
    config: Config,
    core: Literal["quantum", "classical"] = "quantum",
    include_context: bool = False,
) -> D4OrbitClassifier:
    """Build one of the two fixed stages without embedding device policy."""

    return D4OrbitClassifier(
        num_classes=3,
        heads=config.heads,
        reuploads=config.reuploads,
        core=core,
        include_context=include_context,
        dropout=config.dropout,
        foreground_suppressed=config.dataset_id == "model_iv",
    )


def parameter_summary(model: D4OrbitClassifier) -> Dict[str, int | str]:
    return model.parameter_report()
