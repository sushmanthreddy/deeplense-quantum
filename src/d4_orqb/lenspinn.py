"""Clean, vectorized adaptation of the archived LensPINN-small baseline.

The architecture follows the local NeurIPS ML4PS notebook: shifted patch
tokenization, one LSA transformer that predicts a scalar Einstein radius,
integer scatter-based source reconstruction, two MobileNetV3-small-0.5 feature
branches, and a 2000->64->64->3 classifier.  The implementation removes Python
loops over the batch and returns logits (rather than applying Softmax before
CrossEntropyLoss), without changing trainable parameter count.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from torch import nn


def _integer_translate(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    shifted = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0
    elif dy < 0:
        shifted[..., dy:, :] = 0
    if dx > 0:
        shifted[..., :, :dx] = 0
    elif dx < 0:
        shifted[..., :, dx:] = 0
    return shifted


class ShiftPatchTokenizer(nn.Module):
    def __init__(self, image_size: int, embedding_dim: int, patch_size: int) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.shift = patch_size // 2
        self.tokenizer = nn.Conv2d(5, embedding_dim, patch_size, stride=patch_size)
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.positional = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        s = self.shift
        shifted = torch.cat(
            (
                image,
                _integer_translate(image, -s, -s),
                _integer_translate(image, s, -s),
                _integer_translate(image, -s, s),
                _integer_translate(image, s, s),
            ),
            dim=1,
        )
        patches = self.tokenizer(shifted).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(image.shape[0], -1, -1)
        return torch.cat((cls, patches), dim=1) + self.positional


class MultiLocallySelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, patches: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim, heads, dropout=dropout, batch_first=True
        )
        self.register_buffer(
            "self_mask", torch.eye(patches + 1, dtype=torch.bool), persistent=False
        )

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(query, key, value, attn_mask=self.self_mask)
        return output


class FeedForwardRegression(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLSABlock(nn.Module):
    def __init__(
        self, embedding_dim: int, heads: int, patches: int, hidden: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attention = MultiLocallySelfAttention(embedding_dim, heads, patches, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForwardRegression(embedding_dim, hidden, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1))
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        query = self.query(key) / self.temperature
        value = self.value(key)
        value = self.norm1(value + self.attention(key, query, value))
        value = self.norm2(value + self.feedforward(value))
        return self.dropout(value)


class ScalarLensInversion(nn.Module):
    def __init__(
        self,
        image_size: int,
        embedding_dim: int,
        patches: int,
        heads: int,
        blocks: int = 1,
        hidden: int = 64,
        dropout: float = 0.1,
        min_angle: float = -3.323,
        max_angle: float = 3.232,
        reconstruction: Literal["archived-hard", "differentiable"] = "archived-hard",
        min_einstein_radius: float = 0.8,
        max_einstein_radius: float = 1.2,
    ) -> None:
        super().__init__()
        if reconstruction not in ("archived-hard", "differentiable"):
            raise ValueError(f"Unknown reconstruction mode: {reconstruction}")
        self.image_size = image_size
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.reconstruction = reconstruction
        self.min_einstein_radius = float(min_einstein_radius)
        self.max_einstein_radius = float(max_einstein_radius)
        self.blocks = nn.ModuleList(
            TransformerLSABlock(embedding_dim, heads, patches, hidden, dropout)
            for _ in range(blocks)
        )
        self.radius = nn.Linear((patches + 1) * embedding_dim, 1)

    def forward(self, image: torch.Tensor, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = patches
        for block in self.blocks:
            encoded = block(encoded)
        raw_radius = self.radius(encoded.flatten(1))
        if self.reconstruction == "differentiable":
            width = self.max_einstein_radius - self.min_einstein_radius
            einstein_radius = self.min_einstein_radius + width * torch.sigmoid(raw_radius)
        else:
            einstein_radius = raw_radius
        return einstein_radius, self.image_to_source(image, einstein_radius)

    def image_to_source(self, image: torch.Tensor, einstein_radius: torch.Tensor) -> torch.Tensor:
        if self.reconstruction == "differentiable":
            return self._image_to_source_bilinear(image, einstein_radius)
        return self._image_to_source_archived(image, einstein_radius)

    def _lens_coordinates(
        self,
        image: torch.Tensor,
        einstein_radius: torch.Tensor,
        *,
        archived_offset: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = image.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError("Unexpected image size")
        pixel_width = (self.max_angle - self.min_angle) / height
        centre_x, centre_y = height // 2, width // 2
        offset_x = centre_x - 1 if archived_offset else centre_x
        offset_y = centre_y - 1 if archived_offset else centre_y
        range_x = torch.arange(-offset_x, height - offset_x, device=image.device)
        range_y = torch.arange(-offset_y, width - offset_y, device=image.device)
        x, y = torch.meshgrid(range_x, range_y, indexing="ij")
        x = x.float() * pixel_width
        y = y.float() * pixel_width
        radius = torch.sqrt(x.square() + y.square())
        radius = torch.where(radius == 0, torch.ones_like(radius), radius)
        scale = einstein_radius.float().view(batch, 1, 1)
        bx = (x.unsqueeze(0) - scale * x.unsqueeze(0) / radius) / pixel_width
        by = (y.unsqueeze(0) - scale * y.unsqueeze(0) / radius) / pixel_width
        return (
            (bx + centre_x).clamp(0, height - 1),
            (by + centre_y).clamp(0, width - 1),
        )

    def _image_to_source_archived(
        self, image: torch.Tensor, einstein_radius: torch.Tensor
    ) -> torch.Tensor:
        """Faithful hard splat, including the archived off-by-one coordinate grid.

        Casting the predicted coordinates to integers severs the gradient to the
        inversion transformer. This mode exists only for an explicit reproduction.
        """

        batch, _, height, width = image.shape
        bx, by = self._lens_coordinates(
            image, einstein_radius, archived_offset=True
        )
        bx, by = bx.long(), by.long()
        indices = (bx * width + by).reshape(batch, -1)
        source = torch.zeros(batch, height * width, dtype=image.dtype, device=image.device)
        counts = torch.zeros_like(source)
        source.scatter_add_(1, indices, image.reshape(batch, -1))
        counts.scatter_add_(1, indices, torch.ones_like(image).reshape(batch, -1))
        source = source / counts.clamp_min(1)
        return source.reshape(batch, 1, height, width)

    def _image_to_source_bilinear(
        self, image: torch.Tensor, einstein_radius: torch.Tensor
    ) -> torch.Tensor:
        """Centered differentiable forward soft-splat into the source plane."""

        batch, _, height, width = image.shape
        bx, by = self._lens_coordinates(
            image, einstein_radius, archived_offset=False
        )
        x0, y0 = torch.floor(bx), torch.floor(by)
        x1 = (x0 + 1).clamp_max(height - 1)
        y1 = (y0 + 1).clamp_max(width - 1)
        wx, wy = bx - x0, by - y0

        image_flat = image.float().reshape(batch, -1)
        numerator = torch.zeros(
            batch, height * width, dtype=torch.float32, device=image.device
        )
        denominator = torch.zeros_like(numerator)
        neighbors = (
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y0, wx * (1.0 - wy)),
            (x1, y1, wx * wy),
        )
        for x_index, y_index, weight in neighbors:
            index = (x_index.long() * width + y_index.long()).reshape(batch, -1)
            weight_flat = weight.reshape(batch, -1)
            numerator = numerator.scatter_add(1, index, image_flat * weight_flat)
            denominator = denominator.scatter_add(1, index, weight_flat)
        source = numerator / denominator.clamp_min(1e-6)
        return source.reshape(batch, 1, height, width)


class LensPINNSmall(nn.Module):
    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 32,
        embedding_dim: int = 384,
        heads: int = 16,
        hidden: int = 64,
        pretrained: bool = True,
        logits_fix: bool = True,
        reconstruction: Literal["archived-hard", "differentiable"] = "archived-hard",
        retain_archived_unused_block: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError("LensPINN requires timm; install timm==0.9.16") from error

        self.image_size = image_size
        self.logits_fix = logits_fix
        self.reconstruction = reconstruction
        self.tokenizer = ShiftPatchTokenizer(image_size, embedding_dim, patch_size)
        self.inversion = ScalarLensInversion(
            image_size,
            embedding_dim,
            self.tokenizer.num_patches,
            heads,
            blocks=1,
            hidden=hidden,
            reconstruction=reconstruction,
        )
        # The archived Decoder registers this second LSA block but never calls
        # it. Retaining it reproduces the published 7,173,654 nominal count.
        self.archived_unused_transformer = (
            TransformerLSABlock(
                embedding_dim, heads, self.tokenizer.num_patches, hidden, dropout=0.1
            )
            if retain_archived_unused_block
            else None
        )
        self.decoder_observed_source = timm.create_model(
            "mobilenetv3_small_050", pretrained=pretrained
        )
        self.decoder_distortion = timm.create_model(
            "mobilenetv3_small_050", pretrained=pretrained
        )
        for decoder in (self.decoder_observed_source, self.decoder_distortion):
            decoder.conv_stem = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(2000, hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 3),
        )

    def forward(
        self, image: torch.Tensor, distortion: torch.Tensor, return_aux: bool = False
    ):
        patches = self.tokenizer(image)
        einstein_radius, source = self.inversion(image, patches)
        if self.reconstruction == "differentiable" and self.training:
            # One shared call gives observed/source examples equal influence on
            # BatchNorm statistics and is faster than two sequential passes.
            paired_features = self.decoder_observed_source(torch.cat((image, source), dim=0))
            observed_features, source_features = paired_features.chunk(2, dim=0)
        else:
            observed_features = self.decoder_observed_source(image)
            source_features = self.decoder_observed_source(source)
        distortion_features = self.decoder_distortion(distortion)
        logits = self.classifier(
            torch.cat((observed_features - source_features, distortion_features), dim=1)
        )
        if return_aux:
            return logits, {"einstein_radius": einstein_radius, "source": source}
        return logits


def lenspinn_distortion(image: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Stable version of abs(tanh(d_y d_x [log(Imax/I)]^2))."""

    image = image.float().clamp_min(0)
    image = image / image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    log_ratio_sq = torch.log((1.0 + epsilon) / (image + epsilon)).square()
    derivative_y = torch.gradient(log_ratio_sq, dim=-2)[0]
    mixed = torch.gradient(derivative_y, dim=-1)[0]
    return torch.tanh(mixed).abs()
