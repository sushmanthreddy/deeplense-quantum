"""Development-only finite-shot evaluation for a trained D4-ORQB model.

This module deliberately has no test-dataset argument.  It replays the exact
saved Model-I validation ordering, obtains the final eight-qubit statevector,
and samples *joint* computational-basis outcomes after either no basis change
(Z) or an all-qubit Hadamard basis change (X).  The 256-shot result is always a
prefix of the corresponding 1024-shot draw.

The default plug-in estimator mirrors the analytic feature map.  An optional
U-statistic estimator removes the finite-sample bias in the quadratic terms by
forming products from distinct shots.  It does not alter the linear Pauli
expectation estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Sequence, Tuple

import numpy as np
import torch

from .data import CachedNPYDataset, make_loader
from .metrics import classification_metrics
from .model import D4OrbitClassifier
from .quantum import R2_EDGES, R_EDGES, S_EDGES


MODEL_I_CLASSES = ("axion", "cdm", "no_sub")
MODEL_I_DEVELOPMENT_SAMPLES = 87_525
MODEL_I_VALIDATION_SAMPLES = 17_504
MODEL_I_IMAGE_SIZE = 96
SHOT_COUNTS = (256, 1024)
MAX_SHOTS = max(SHOT_COUNTS)
# These are protocol constants, not command-line tuning knobs.
SHOT_SEEDS = (17, 42, 314_159)
BOOTSTRAP_SEED = 20_260_711
BOOTSTRAP_RESAMPLES = 10_000
NONINFERIOR_MARGIN = -0.005
SAVED_METRIC_ATOL = 1e-5
REPLAY_MIN_CLASS_AGREEMENT = 0.999
REPLAY_MAX_PROBABILITY_MAE = 0.003
REPLAY_MAX_ACCURACY_DELTA = 0.001
SCALAR_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auc_ovr",
    "nll",
    "brier",
    "ece_15",
)
Estimator = Literal["plugin", "ustat"]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read valid JSON from {path}") from error


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _atomic_json(path: Path, value: Mapping) -> None:
    temporary = path.with_name(f"{path.name}.building-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f"{path.name}.building-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _normalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError("Measurement probabilities must have shape (states, outcomes)")
    probabilities = probabilities.float()
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("Measurement probabilities contain NaN or infinity")
    minimum = float(probabilities.min())
    if minimum < -1e-7:
        raise RuntimeError(f"Measurement probabilities contain a negative value: {minimum}")
    probabilities = probabilities.clamp_min(0.0)
    normalizer = probabilities.sum(dim=1, keepdim=True)
    if bool((normalizer <= 0.0).any()):
        raise RuntimeError("A state has no finite measurement probability")
    return probabilities / normalizer


def hadamard_all(state: torch.Tensor) -> torch.Tensor:
    """Apply an in-place-free normalized Walsh-Hadamard transform by rows."""

    if state.ndim != 2 or state.shape[1] <= 0:
        raise ValueError("State must have shape (states, 2**qubits)")
    dimension = state.shape[1]
    if dimension & (dimension - 1):
        raise ValueError("Statevector dimension must be a power of two")
    transformed = state
    width = 1
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    while width < dimension:
        paired = transformed.reshape(transformed.shape[0], -1, 2, width)
        lower, upper = paired[:, :, 0], paired[:, :, 1]
        transformed = (
            torch.stack((lower + upper, lower - upper), dim=2).reshape_as(transformed)
            * inverse_sqrt_two
        )
        width *= 2
    return transformed


def z_and_x_probabilities(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized full-register outcome distributions in Z and X."""

    z_probabilities = _normalize_probabilities(state.abs().square())
    x_state = hadamard_all(state)
    x_probabilities = _normalize_probabilities(x_state.abs().square())
    return z_probabilities, x_probabilities


def sample_joint_bitstrings(
    probabilities: torch.Tensor,
    shots: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one categorical full-register outcome per shot and state."""

    if shots < 1:
        raise ValueError("shots must be positive")
    probabilities = _normalize_probabilities(probabilities)
    return torch.multinomial(
        probabilities, num_samples=shots, replacement=True, generator=generator
    )


def outcomes_to_signs(outcomes: torch.Tensor, z_signs: torch.Tensor) -> torch.Tensor:
    """Decode integer bitstrings to simultaneous Pauli signs for every qubit."""

    if outcomes.ndim != 2 or z_signs.ndim != 2:
        raise ValueError("Outcomes and sign table must both be rank two")
    if outcomes.numel() and (
        int(outcomes.min()) < 0 or int(outcomes.max()) >= z_signs.shape[1]
    ):
        raise ValueError("Outcome is outside the supplied bitstring sign table")
    table = z_signs.transpose(0, 1).to(device=outcomes.device, dtype=torch.float32)
    return table[outcomes]


def _edge_correlations(
    signs: torch.Tensor, edges: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    return torch.stack(
        [(signs[..., a] * signs[..., b]).mean(dim=1) for a, b in edges], dim=1
    )


def _u_products(
    signs: torch.Tensor, edges: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    """Unbiased products E[S_a]E[S_b] using ordered distinct shots."""

    shots = signs.shape[1]
    if shots < 2:
        raise ValueError("U-statistic features require at least two shots")
    sums = signs.sum(dim=1)
    denominator = float(shots * (shots - 1))
    return torch.stack(
        [
            (
                sums[:, a] * sums[:, b]
                - (signs[..., a] * signs[..., b]).sum(dim=1)
            )
            / denominator
            for a, b in edges
        ],
        dim=1,
    )


def _basis_measurement_features(
    signs: torch.Tensor,
    estimator: Estimator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, orbit variance, and the three edge-correlation fields."""

    if signs.ndim != 3 or signs.shape[2] != 8:
        raise ValueError("Joint signs must have shape (states, shots, 8)")
    means = signs.mean(dim=1)
    orbit_mean = means.mean(dim=1)
    if estimator == "plugin":
        orbit_variance = means.square().mean(dim=1) - orbit_mean.square()
    elif estimator == "ustat":
        shots = signs.shape[1]
        if shots < 2:
            raise ValueError("U-statistic features require at least two shots")
        denominator = float(shots * (shots - 1))
        sums = signs.sum(dim=1)
        mean_squared = ((sums.square() - shots) / denominator).mean(dim=1)
        shot_orbit_means = signs.mean(dim=2)
        orbit_sum = shot_orbit_means.sum(dim=1)
        orbit_mean_squared = (
            orbit_sum.square() - shot_orbit_means.square().sum(dim=1)
        ) / denominator
        orbit_variance = mean_squared - orbit_mean_squared
    else:
        raise ValueError(f"Unknown finite-shot estimator: {estimator}")
    return (
        orbit_mean,
        orbit_variance,
        _edge_correlations(signs, R_EDGES),
        _edge_correlations(signs, R2_EDGES),
        _edge_correlations(signs, S_EDGES),
    )


def finite_shot_invariants(
    z_sign_samples: torch.Tensor,
    x_sign_samples: torch.Tensor,
    heads: int,
    estimator: Estimator = "plugin",
) -> torch.Tensor:
    """Reconstruct the classifier's 12 invariants per head from joint shots."""

    if z_sign_samples.shape != x_sign_samples.shape:
        raise ValueError("Z and X samples must have identical shape")
    states = z_sign_samples.shape[0]
    if heads < 1 or states % heads:
        raise ValueError("State count must be divisible by the number of heads")
    z_mean, z_variance, zz_r, zz_r2, zz_s = _basis_measurement_features(
        z_sign_samples, estimator
    )
    x_mean, x_variance, xx_r, xx_r2, xx_s = _basis_measurement_features(
        x_sign_samples, estimator
    )
    if estimator == "plugin":
        z_product_r = torch.stack(
            [
                z_sign_samples[..., a].mean(1) * z_sign_samples[..., b].mean(1)
                for a, b in R_EDGES
            ],
            dim=1,
        )
        x_product_r = torch.stack(
            [
                x_sign_samples[..., a].mean(1) * x_sign_samples[..., b].mean(1)
                for a, b in R_EDGES
            ],
            dim=1,
        )
    else:
        z_product_r = _u_products(z_sign_samples, R_EDGES)
        x_product_r = _u_products(x_sign_samples, R_EDGES)
    invariant = torch.stack(
        (
            z_mean,
            z_variance,
            x_mean,
            x_variance,
            zz_r.mean(dim=1),
            zz_r2.mean(dim=1),
            zz_s.mean(dim=1),
            xx_r.mean(dim=1),
            xx_r2.mean(dim=1),
            xx_s.mean(dim=1),
            (zz_r - z_product_r).mean(dim=1),
            (xx_r - x_product_r).mean(dim=1),
        ),
        dim=1,
    )
    return invariant.reshape(states // heads, heads * invariant.shape[1])


def _analytic_basis_features(
    probabilities: torch.Tensor,
    sign_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means = probabilities @ sign_table.transpose(0, 1)

    def correlations(edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
        observables = torch.stack([sign_table[a] * sign_table[b] for a, b in edges])
        return probabilities @ observables.transpose(0, 1)

    orbit_mean = means.mean(dim=1)
    return (
        orbit_mean,
        means.square().mean(dim=1) - orbit_mean.square(),
        correlations(R_EDGES),
        correlations(R2_EDGES),
        correlations(S_EDGES),
    )


def analytic_invariants_from_probabilities(
    z_probabilities: torch.Tensor,
    x_probabilities: torch.Tensor,
    z_signs: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    """Compute the exact feature map from the same two basis distributions."""

    if z_probabilities.shape != x_probabilities.shape:
        raise ValueError("Z and X probability arrays must have identical shape")
    states = z_probabilities.shape[0]
    if states % heads:
        raise ValueError("State count must be divisible by heads")
    sign_table = z_signs.to(device=z_probabilities.device, dtype=z_probabilities.dtype)
    z_mean, z_variance, zz_r, zz_r2, zz_s = _analytic_basis_features(
        z_probabilities, sign_table
    )
    x_mean, x_variance, xx_r, xx_r2, xx_s = _analytic_basis_features(
        x_probabilities, sign_table
    )
    z_single = z_probabilities @ sign_table.transpose(0, 1)
    x_single = x_probabilities @ sign_table.transpose(0, 1)
    z_product_r = torch.stack([z_single[:, a] * z_single[:, b] for a, b in R_EDGES], 1)
    x_product_r = torch.stack([x_single[:, a] * x_single[:, b] for a, b in R_EDGES], 1)
    invariant = torch.stack(
        (
            z_mean,
            z_variance,
            x_mean,
            x_variance,
            zz_r.mean(1),
            zz_r2.mean(1),
            zz_s.mean(1),
            xx_r.mean(1),
            xx_r2.mean(1),
            xx_s.mean(1),
            (zz_r - z_product_r).mean(1),
            (xx_r - x_product_r).mean(1),
        ),
        1,
    )
    return invariant.reshape(states // heads, heads * invariant.shape[1])


def logits_from_invariants(
    model: D4OrbitClassifier,
    invariants: torch.Tensor,
    encoded: torch.Tensor,
) -> torch.Tensor:
    """Apply the frozen classical head to an alternate bottleneck estimate."""

    features = [invariants]
    if model.context_projection is not None:
        context = torch.cat(
            (
                encoded.mean(dim=1),
                encoded.std(dim=1, unbiased=False),
                encoded.amax(dim=1),
            ),
            dim=1,
        )
        features.append(model.context_projection(context))
    return model.head(torch.cat(features, dim=1))


def _infer_architecture(config: Mapping, state: Mapping[str, torch.Tensor]) -> Dict:
    if bool(config.get("haar_subtype_max_envelope", False)):
        raise RuntimeError(
            "Finite-shot replay does not support the derived max-preserving "
            "Haar subtype envelope"
        )
    if bool(config.get("shared_late_refinement", False)) or (
        "encoder.shared_refinement_gates" in state
    ):
        raise RuntimeError(
            "Finite-shot replay does not yet support shared late refinement"
        )
    if bool(config.get("haar_subtype_residual", False)) or any(
        str(key).startswith("haar_subtype_residual.") for key in state
    ):
        raise RuntimeError(
            "Finite-shot replay does not yet support the image-derived "
            "Haar subtype residual"
        )
    core_parameters = state.get("core.params")
    stem = state.get("encoder.stem.0.weight")
    final = state.get("encoder.final.0.weight")
    classifier = state.get("head.4.weight")
    if any(value is None for value in (core_parameters, stem, final, classifier)):
        raise RuntimeError("Checkpoint is missing architecture-defining tensors")
    if core_parameters.ndim != 3 or core_parameters.shape[2] != 11:
        raise RuntimeError("Checkpoint does not contain a supported D4-ORQB circuit")
    heads, reuploads = map(int, core_parameters.shape[:2])
    if int(config.get("heads", -1)) != heads or int(config.get("reuploads", -1)) != reuploads:
        raise RuntimeError("Checkpoint circuit shape disagrees with config.json")
    input_channels = int(stem.shape[1])
    inferred_physics = {8: "base", 10: "radial"}.get(input_channels)
    if inferred_physics is None:
        raise RuntimeError(f"Unsupported physics-bank input channels: {input_channels}")
    physics_variant = str(config.get("physics_variant", inferred_physics))
    if physics_variant != inferred_physics:
        raise RuntimeError("Checkpoint physics variant disagrees with config.json")
    output_dim = int(final.shape[0])
    inferred_encoder = {128: "tiny", 192: "small"}.get(output_dim)
    if inferred_encoder is None:
        raise RuntimeError(f"Unsupported encoder output width: {output_dim}")
    encoder_variant = str(config.get("encoder_variant", inferred_encoder))
    if encoder_variant != inferred_encoder:
        raise RuntimeError("Checkpoint encoder variant disagrees with config.json")
    context_present = any(str(key).startswith("context_projection.") for key in state)
    if bool(config.get("include_context", False)) != context_present:
        raise RuntimeError("Checkpoint context branch disagrees with config.json")
    if int(classifier.shape[0]) != len(MODEL_I_CLASSES):
        raise RuntimeError("Checkpoint classifier is not three-class Model I")
    return {
        "heads": heads,
        "reuploads": reuploads,
        "physics_variant": physics_variant,
        "encoder_variant": encoder_variant,
        "include_context": context_present,
    }


def load_completed_quantum_run_strict(
    run_dir: Path,
) -> Tuple[D4OrbitClassifier, Dict, Dict, np.ndarray, Mapping[str, np.ndarray]]:
    """Strict-load a completed, development-only quantum training artifact."""

    required = (
        "config.json",
        "summary.json",
        "history.json",
        "data_report.json",
        "best.pt",
        "split_indices.npz",
        "best_validation_predictions.npz",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete training run; missing {missing}")
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    history = _read_json(run_dir / "history.json")
    data_report = _read_json(run_dir / "data_report.json")
    if bool(config.get("haar_subtype_residual", False)):
        raise RuntimeError(
            "Finite-shot replay does not yet support the image-derived "
            "Haar subtype residual"
        )
    if config.get("core") != "quantum":
        raise RuntimeError("Finite-shot evaluation requires core=quantum")
    if int(config.get("split_seed", -1)) != 42 or not math.isclose(
        float(config.get("val_fraction", -1.0)), 0.20, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Run does not use the predeclared Model-I validation split protocol")
    if config.get("evaluate_test") is not False:
        raise RuntimeError("Training config must record evaluate_test=false")
    if summary.get("official_test_evaluated") is not False:
        raise RuntimeError("Training summary does not preserve the official test lock")
    if data_report.get("official_test_locked_during_selection") is not True:
        raise RuntimeError("Data report does not assert the official test lock")
    if data_report.get("official_test_cache_opened", False) is not False:
        raise RuntimeError("Data report says the official test cache was opened")
    report_development = data_report.get("development")
    if not isinstance(report_development, dict) or (
        tuple(report_development.get("classes", ())) != MODEL_I_CLASSES
        or int(report_development.get("samples", -1)) != MODEL_I_DEVELOPMENT_SAMPLES
        or int(report_development.get("image_size", -1)) != MODEL_I_IMAGE_SIZE
    ):
        raise RuntimeError("Data report does not identify canonical Model-I development data")
    disjoint = data_report.get("digest_disjoint")
    if not isinstance(disjoint, dict) or int(disjoint.get("intersection", -1)) != 0:
        raise RuntimeError("Data report does not establish development/test digest disjointness")
    forbidden_test_artifacts = [
        name
        for name in ("test_predictions.npz", "official_test_predictions.npz")
        if (run_dir / name).exists()
    ]
    if forbidden_test_artifacts:
        raise RuntimeError(
            f"Training run contains forbidden test prediction artifacts: {forbidden_test_artifacts}"
        )
    if not isinstance(history, list) or not history:
        raise RuntimeError("Completed run has no training history")
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch < 1 or not any(int(item.get("epoch", -1)) == best_epoch for item in history):
        raise RuntimeError("Best epoch is absent from run history")
    checkpoint = _torch_load(run_dir / "best.pt")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("best.pt does not contain a model state dictionary")
    if int(checkpoint.get("epoch", -1)) != best_epoch:
        raise RuntimeError("best.pt epoch disagrees with summary.json")
    architecture = _infer_architecture(config, checkpoint["model"])
    model = D4OrbitClassifier(
        num_classes=len(MODEL_I_CLASSES),
        heads=architecture["heads"],
        reuploads=architecture["reuploads"],
        core="quantum",
        include_context=architecture["include_context"],
        encoder_variant=architecture["encoder_variant"],
        physics_variant=architecture["physics_variant"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    expected_parameters = summary.get("parameters")
    if not isinstance(expected_parameters, dict):
        raise RuntimeError("Training summary has no parameter report")
    actual_parameters = model.parameter_report()
    for name in ("total", "encoder", "orbit_projection", "core", "quantum"):
        if name not in expected_parameters:
            raise RuntimeError(f"Training parameter report lacks {name}")
        if int(expected_parameters[name]) != int(actual_parameters[name]):
            raise RuntimeError(f"Parameter report mismatch for {name}")
    with np.load(run_dir / "split_indices.npz") as split_file:
        if set(split_file.files) < {"train", "val"}:
            raise RuntimeError("split_indices.npz lacks train or val")
        train_indices = np.asarray(split_file["train"], dtype=np.int64)
        validation_indices = np.asarray(split_file["val"], dtype=np.int64)
    if (
        train_indices.ndim != 1
        or validation_indices.ndim != 1
        or len(validation_indices) != MODEL_I_VALIDATION_SAMPLES
        or len(np.unique(train_indices)) != len(train_indices)
        or len(np.unique(validation_indices)) != len(validation_indices)
        or np.intersect1d(train_indices, validation_indices).size
        or not np.array_equal(
            np.sort(np.concatenate((train_indices, validation_indices))),
            np.arange(MODEL_I_DEVELOPMENT_SAMPLES),
        )
    ):
        raise RuntimeError("Run does not contain the complete canonical-sized development split")
    with np.load(run_dir / "best_validation_predictions.npz") as prediction_file:
        prediction_artifact = {
            name: np.asarray(prediction_file[name]) for name in prediction_file.files
        }
    required_prediction_fields = {"indices", "labels", "logits", "probabilities"}
    if not required_prediction_fields.issubset(prediction_artifact):
        raise RuntimeError("Saved validation predictions are incomplete")
    if not np.array_equal(
        np.asarray(prediction_artifact["indices"], dtype=np.int64), validation_indices
    ):
        raise RuntimeError("Saved predictions do not preserve validation split ordering")
    saved_labels = np.asarray(prediction_artifact["labels"], dtype=np.int64)
    saved_logits = np.asarray(prediction_artifact["logits"])
    saved_probabilities = np.asarray(prediction_artifact["probabilities"])
    if (
        saved_labels.shape != (MODEL_I_VALIDATION_SAMPLES,)
        or saved_logits.shape != (MODEL_I_VALIDATION_SAMPLES, len(MODEL_I_CLASSES))
        or saved_probabilities.shape
        != (MODEL_I_VALIDATION_SAMPLES, len(MODEL_I_CLASSES))
        or not np.isfinite(saved_logits).all()
        or not np.isfinite(saved_probabilities).all()
        or (saved_labels < 0).any()
        or (saved_labels >= len(MODEL_I_CLASSES)).any()
    ):
        raise RuntimeError("Saved validation predictions have invalid shapes or values")
    reconstructed_probabilities = _softmax(saved_logits)
    if not np.allclose(
        reconstructed_probabilities,
        saved_probabilities,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError("Saved probabilities disagree with saved logits")
    summary_validation = summary.get("validation")
    if not isinstance(summary_validation, dict):
        raise RuntimeError("Training summary has no validation metrics")
    replayed_saved_metrics = classification_metrics(
        saved_labels, saved_logits, list(MODEL_I_CLASSES)
    )
    for name in SCALAR_METRICS:
        if name not in summary_validation or abs(
            float(replayed_saved_metrics[name]) - float(summary_validation[name])
        ) > SAVED_METRIC_ATOL:
            raise RuntimeError(f"Saved validation metric disagrees with summary: {name}")
    return model, config, summary, validation_indices, prediction_artifact


def validate_development_cache(
    cache_dir: Path, validation_indices: np.ndarray, saved_labels: np.ndarray
) -> Tuple[Tuple[str, ...], np.ndarray]:
    """Validate only the canonical Model-I development cache."""

    required = ("metadata.json", "images.npy", "labels.npy", "manifest.csv")
    missing = [name for name in required if not (cache_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Development cache is incomplete; missing {missing}")
    metadata = _read_json(cache_dir / "metadata.json")
    classes = tuple(metadata.get("classes", ()))
    if (
        metadata.get("complete") is not True
        or classes != MODEL_I_CLASSES
        or int(metadata.get("samples", -1)) != MODEL_I_DEVELOPMENT_SAMPLES
        or int(metadata.get("image_size", -1)) != MODEL_I_IMAGE_SIZE
    ):
        raise RuntimeError("Cache is not the canonical-sized Model-I development cache")
    source_name = Path(str(metadata.get("source_root", ""))).name
    if source_name != "Model_I":
        raise RuntimeError("Cache metadata does not identify the Model_I development source")
    labels = np.load(cache_dir / "labels.npy", mmap_mode="r")
    images = np.load(cache_dir / "images.npy", mmap_mode="r")
    if labels.shape != (MODEL_I_DEVELOPMENT_SAMPLES,) or images.shape != (
        MODEL_I_DEVELOPMENT_SAMPLES,
        MODEL_I_IMAGE_SIZE,
        MODEL_I_IMAGE_SIZE,
    ):
        raise RuntimeError("Development cache arrays have unexpected shapes")
    validation_labels = np.asarray(labels[validation_indices], dtype=np.int64)
    if not np.array_equal(validation_labels, np.asarray(saved_labels, dtype=np.int64)):
        raise RuntimeError("Run labels disagree with the development cache")
    return classes, validation_labels


def _metric_delta(actual: Mapping, reference: Mapping) -> Dict[str, float]:
    return {
        name: float(actual[name]) - float(reference[name])
        for name in SCALAR_METRICS
    }


def _prediction_agreement(
    labels: np.ndarray,
    finite_probabilities: np.ndarray,
    analytic_probabilities: np.ndarray,
    finite_logits: np.ndarray | None = None,
    analytic_logits: np.ndarray | None = None,
) -> Dict[str, float | int]:
    finite_predictions = finite_probabilities.argmax(axis=1)
    analytic_predictions = analytic_probabilities.argmax(axis=1)
    finite_correct = finite_predictions == labels
    analytic_correct = analytic_predictions == labels
    difference = finite_probabilities - analytic_probabilities
    result = {
        "class_agreement": float((finite_predictions == analytic_predictions).mean()),
        "changed_predictions": int((finite_predictions != analytic_predictions).sum()),
        "analytic_correct_finite_wrong": int((analytic_correct & ~finite_correct).sum()),
        "analytic_wrong_finite_correct": int((~analytic_correct & finite_correct).sum()),
        "probability_mae": float(np.abs(difference).mean()),
        "probability_rmse": float(np.sqrt(np.square(difference).mean())),
        "probability_max_abs": float(np.abs(difference).max()),
    }
    if finite_logits is not None or analytic_logits is not None:
        if finite_logits is None or analytic_logits is None:
            raise ValueError("Both finite and analytic logits are required for logit agreement")
        logit_difference = np.asarray(finite_logits, dtype=np.float64) - np.asarray(
            analytic_logits, dtype=np.float64
        )
        result.update(
            {
                "logit_mae": float(np.abs(logit_difference).mean()),
                "logit_rmse": float(np.sqrt(np.square(logit_difference).mean())),
                "logit_max_abs": float(np.abs(logit_difference).max()),
            }
        )
    return result


def stratified_paired_accuracy_bootstrap(
    labels: np.ndarray,
    finite_predictions: np.ndarray,
    analytic_predictions: np.ndarray,
    samples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    noninferior_margin: float = NONINFERIOR_MARGIN,
) -> Dict[str, object]:
    """Paired bootstrap of finite-minus-analytic accuracy, stratified by class.

    Within each true class, resampling sample indices is exactly equivalent to
    drawing the observed paired correctness differences from their empirical
    three-point distribution.  Using multinomial counts implements that same
    bootstrap without materializing a ``samples x validation_size`` matrix.
    """

    labels = np.asarray(labels, dtype=np.int64)
    finite_predictions = np.asarray(finite_predictions, dtype=np.int64)
    analytic_predictions = np.asarray(analytic_predictions, dtype=np.int64)
    if (
        labels.ndim != 1
        or finite_predictions.shape != labels.shape
        or analytic_predictions.shape != labels.shape
        or len(labels) == 0
    ):
        raise ValueError("Labels and paired predictions must be nonempty aligned vectors")
    if samples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    paired_difference = (
        (finite_predictions == labels).astype(np.int8)
        - (analytic_predictions == labels).astype(np.int8)
    )
    generator = np.random.default_rng(seed)
    replicates = np.zeros(samples, dtype=np.float64)
    strata = np.unique(labels)
    stratum_sizes = {}
    for label in strata:
        values = paired_difference[labels == label]
        support = len(values)
        stratum_sizes[str(int(label))] = int(support)
        outcomes, counts = np.unique(values, return_counts=True)
        probabilities = counts.astype(np.float64) / support
        draws = generator.multinomial(support, probabilities, size=samples)
        replicates += (draws @ outcomes.astype(np.float64)) / len(labels)
    lower_two_sided, upper_two_sided = np.quantile(replicates, (0.025, 0.975))
    lower_one_sided = float(np.quantile(replicates, 0.05))
    return {
        "difference_finite_minus_analytic": float(paired_difference.mean()),
        "two_sided_95_ci_low": float(lower_two_sided),
        "two_sided_95_ci_high": float(upper_two_sided),
        "one_sided_95_lower_bound": lower_one_sided,
        "noninferior_margin": float(noninferior_margin),
        "noninferior_at_one_sided_95": bool(lower_one_sided > noninferior_margin),
        "resamples": int(samples),
        "analysis_seed": int(seed),
        "stratified_by": "true_class",
        "stratum_sizes": stratum_sizes,
    }


def _feature_error_update(
    accumulator: Dict[str, np.ndarray | float | int],
    finite: torch.Tensor,
    analytic: torch.Tensor,
) -> None:
    difference = (finite.float() - analytic.float()).detach().cpu().numpy().astype(np.float64)
    absolute = np.abs(difference)
    accumulator["elements"] += int(difference.size)
    accumulator["sum_abs"] += float(absolute.sum())
    accumulator["sum_square"] += float(np.square(difference).sum())
    accumulator["max_abs"] = max(float(accumulator["max_abs"]), float(absolute.max()))
    accumulator["per_feature_sum_abs"] += absolute.sum(axis=0)
    accumulator["per_feature_sum_square"] += np.square(difference).sum(axis=0)
    accumulator["per_feature_max_abs"] = np.maximum(
        accumulator["per_feature_max_abs"], absolute.max(axis=0)
    )
    accumulator["samples"] += int(difference.shape[0])


def _feature_error_finalize(accumulator: Mapping) -> Dict:
    elements = int(accumulator["elements"])
    samples = int(accumulator["samples"])
    return {
        "mae": float(accumulator["sum_abs"]) / elements,
        "rmse": math.sqrt(float(accumulator["sum_square"]) / elements),
        "max_abs": float(accumulator["max_abs"]),
        "per_feature_mae": (accumulator["per_feature_sum_abs"] / samples).tolist(),
        "per_feature_rmse": np.sqrt(
            accumulator["per_feature_sum_square"] / samples
        ).tolist(),
        "per_feature_max_abs": accumulator["per_feature_max_abs"].tolist(),
    }


def _new_feature_accumulator(feature_count: int) -> Dict:
    return {
        "elements": 0,
        "samples": 0,
        "sum_abs": 0.0,
        "sum_square": 0.0,
        "max_abs": 0.0,
        "per_feature_sum_abs": np.zeros(feature_count, dtype=np.float64),
        "per_feature_sum_square": np.zeros(feature_count, dtype=np.float64),
        "per_feature_max_abs": np.zeros(feature_count, dtype=np.float64),
    }


def _aggregate_records(records: Sequence[Mapping]):
    grouped = {}
    for shots in SHOT_COUNTS:
        estimators = sorted({record["estimator"] for record in records})
        for estimator in estimators:
            selected = [
                record
                for record in records
                if record["shots"] == shots and record["estimator"] == estimator
            ]
            metric_summary = {}
            for name in SCALAR_METRICS:
                values = np.asarray([item["metrics"][name] for item in selected], dtype=float)
                metric_summary[name] = {
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            grouped[f"shots_{shots}_{estimator}"] = {
                "replicates": len(selected),
                "metrics_across_seeds": metric_summary,
                "seed_averaging_performed": False,
                "note": (
                    "Each seed is an independent simulated measurement run; predictions are "
                    "not ensembled because that would triple the effective shot budget."
                ),
            }
    return grouped


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def evaluate_finite_shots(args: argparse.Namespace) -> Dict:
    run_dir = Path(args.run_dir).expanduser().resolve(strict=True)
    development_cache = Path(args.development_cache).expanduser().resolve(strict=True)
    # Resolve existing symlinked parents and ``..`` before containment checks.
    # The final directory must not exist, so strict resolution is impossible.
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise RuntimeError("Finite-shot output must be outside the immutable training run")
    if output_dir == development_cache or development_cache in output_dir.parents:
        raise RuntimeError("Finite-shot output must be outside the development cache")
    if output_dir.exists():
        raise FileExistsError(f"Finite-shot output already exists: {output_dir}")
    model, config, summary, validation_indices, saved_predictions = (
        load_completed_quantum_run_strict(run_dir)
    )
    training_data_report = _read_json(run_dir / "data_report.json")
    class_names, labels = validate_development_cache(
        development_cache, validation_indices, saved_predictions["labels"]
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use a GPU PyTorchJob")
    model.to(device).eval()
    dataset = CachedNPYDataset(development_cache, validation_indices)
    loader = make_loader(dataset, args.batch_size, False, args.workers, SHOT_SEEDS[0])
    estimators: Tuple[Estimator, ...] = (
        ("plugin", "ustat") if args.include_ustat else ("plugin",)
    )
    generators = {
        seed: {
            "z": _generator(device, seed),
            "x": _generator(device, seed + 1_000_003),
        }
        for seed in SHOT_SEEDS
    }
    keys = [
        (seed, shots, estimator)
        for seed in SHOT_SEEDS
        for shots in SHOT_COUNTS
        for estimator in estimators
    ]
    logits_chunks = {key: [] for key in keys}
    feature_errors = {
        key: _new_feature_accumulator(model.core.output_dim) for key in keys
    }
    labels_chunks, indices_chunks, analytic_logits_chunks = [], [], []
    offset = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for images, batch_labels, batch_indices in loader:
            expected = validation_indices[offset : offset + len(batch_indices)]
            actual_indices = batch_indices.numpy().astype(np.int64, copy=False)
            if not np.array_equal(actual_indices, expected):
                raise RuntimeError("DataLoader changed the saved validation sample ordering")
            offset += len(batch_indices)
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            autocast_enabled = device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=autocast_enabled,
            ):
                encoded, angles = model.orbit_encode(images)
            with torch.autocast(device_type=device.type, enabled=False):
                state = model.core._run_statevector(angles)
                z_probabilities, x_probabilities = z_and_x_probabilities(state)
                analytic_invariants = analytic_invariants_from_probabilities(
                    z_probabilities,
                    x_probabilities,
                    model.core.z_signs,
                    model.heads,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=autocast_enabled,
            ):
                analytic_logits = logits_from_invariants(model, analytic_invariants, encoded)
            analytic_logits_chunks.append(analytic_logits.float().cpu().numpy())
            labels_chunks.append(batch_labels.numpy())
            indices_chunks.append(actual_indices)
            for seed in SHOT_SEEDS:
                # Exactly one maximum-shot draw per basis makes 256 a strict prefix.
                z_outcomes = sample_joint_bitstrings(
                    z_probabilities, MAX_SHOTS, generators[seed]["z"]
                )
                x_outcomes = sample_joint_bitstrings(
                    x_probabilities, MAX_SHOTS, generators[seed]["x"]
                )
                z_sign_samples = outcomes_to_signs(z_outcomes, model.core.z_signs)
                x_sign_samples = outcomes_to_signs(x_outcomes, model.core.z_signs)
                for shots in SHOT_COUNTS:
                    nested_z = z_sign_samples[:, :shots]
                    nested_x = x_sign_samples[:, :shots]
                    for estimator in estimators:
                        key = (seed, shots, estimator)
                        finite_invariants = finite_shot_invariants(
                            nested_z, nested_x, model.heads, estimator
                        )
                        _feature_error_update(
                            feature_errors[key], finite_invariants, analytic_invariants
                        )
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                            enabled=autocast_enabled,
                        ):
                            finite_logits = logits_from_invariants(
                                model, finite_invariants, encoded
                            )
                        logits_chunks[key].append(finite_logits.float().cpu().numpy())
    if offset != len(validation_indices):
        raise RuntimeError("Validation replay did not cover every saved sample")
    indices = np.concatenate(indices_chunks).astype(np.int64, copy=False)
    labels = np.concatenate(labels_chunks).astype(np.int64, copy=False)
    if not np.array_equal(indices, validation_indices):
        raise RuntimeError("Final prediction ordering differs from the saved validation split")
    analytic_logits = np.concatenate(analytic_logits_chunks).astype(np.float32, copy=False)
    analytic_probabilities = _softmax(analytic_logits)
    analytic_metrics = classification_metrics(labels, analytic_logits, list(class_names))
    saved_probabilities = np.asarray(saved_predictions["probabilities"], dtype=np.float64)
    saved_replay_agreement = _prediction_agreement(
        labels,
        saved_probabilities,
        analytic_probabilities,
        np.asarray(saved_predictions["logits"]),
        analytic_logits,
    )
    saved_replay_agreement["saved_probability_sha256"] = _array_sha256(saved_probabilities)
    saved_replay_agreement["replay_probability_sha256"] = _array_sha256(
        analytic_probabilities
    )
    accuracy_delta = abs(
        float(analytic_metrics["accuracy"]) - float(summary["validation"]["accuracy"])
    )
    if (
        float(saved_replay_agreement["class_agreement"])
        < REPLAY_MIN_CLASS_AGREEMENT
        or float(saved_replay_agreement["probability_mae"])
        > REPLAY_MAX_PROBABILITY_MAE
        or accuracy_delta > REPLAY_MAX_ACCURACY_DELTA
    ):
        raise RuntimeError(
            "Current analytic checkpoint replay does not agree with the saved validation "
            f"prediction artifact: {saved_replay_agreement}, accuracy_delta={accuracy_delta}"
        )
    saved_replay_agreement["strict_replay_passed"] = True
    saved_replay_agreement["accuracy_abs_delta_from_summary"] = accuracy_delta
    arrays: Dict[str, np.ndarray] = {
        "indices": indices,
        "labels": labels,
        "analytic_logits": analytic_logits,
        "analytic_probabilities": analytic_probabilities.astype(np.float32),
    }
    records = []
    for seed, shots, estimator in keys:
        logits = np.concatenate(logits_chunks[(seed, shots, estimator)]).astype(
            np.float32, copy=False
        )
        probabilities = _softmax(logits)
        metrics = classification_metrics(labels, logits, list(class_names))
        key_name = f"seed_{seed}_shots_{shots}_{estimator}"
        arrays[f"{key_name}_logits"] = logits
        arrays[f"{key_name}_probabilities"] = probabilities.astype(np.float32)
        records.append(
            {
                "seed": seed,
                "shots": shots,
                "estimator": estimator,
                "metrics": metrics,
                "delta_from_analytic": _metric_delta(metrics, analytic_metrics),
                "analytic_prediction_agreement": _prediction_agreement(
                    labels,
                    probabilities,
                    analytic_probabilities,
                    logits,
                    analytic_logits,
                ),
                "paired_accuracy_bootstrap": stratified_paired_accuracy_bootstrap(
                    labels,
                    probabilities.argmax(axis=1),
                    analytic_probabilities.argmax(axis=1),
                ),
                "analytic_feature_agreement": _feature_error_finalize(
                    feature_errors[(seed, shots, estimator)]
                ),
            }
        )
    aggregate = _aggregate_records(records)
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_npz(output_dir / "predictions.npz", arrays)
    result = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "Model_I development validation only",
            "official_test_cache_touched": False,
            "test_isolation": {
                "this_finite_shot_evaluator": (
                    "has no test-cache/test-root argument and opened development validation only"
                ),
                "training_config_evaluate_test": config["evaluate_test"],
                "training_summary_official_test_evaluated": summary[
                    "official_test_evaluated"
                ],
                "training_data_report_official_test_locked": training_data_report[
                    "official_test_locked_during_selection"
                ],
                "training_data_report_test_cache_opened_record": training_data_report.get(
                    "official_test_cache_opened", "legacy_field_not_recorded"
                ),
                "legacy_training_report_contains_test_metadata": isinstance(
                    training_data_report.get("test"), dict
                ),
                "legacy_metadata_interpretation": (
                    "The training report may contain test-cache metadata/digest-disjointness from "
                    "the legacy cache-preparation path; config, summary, and absence of a test "
                    "prediction artifact establish that no official-test model inference was run."
                ),
            },
            "samples": int(len(labels)),
            "class_names": list(class_names),
            "sample_order_sha256": _array_sha256(indices),
            "canonical_output_dir": str(output_dir),
        },
        "protocol": {
            "measurement": "joint full 8-bit strings",
            "bases": ["Z", "X (H on all 8 qubits before Z measurement)"],
            "shots": list(SHOT_COUNTS),
            "nested": "the 256-shot estimate is the first 256 outcomes of each 1024-shot draw",
            "seeds": list(SHOT_SEEDS),
            "x_basis_seed_offset": 1_000_003,
            "estimators": list(estimators),
            "primary_estimator": "plugin",
            "ustat_role": "bias-corrected sensitivity analysis",
            "seed_role": (
                "independent simulated measurement runs; never averaged for a primary result"
            ),
            "equivalent_circuit_executions": {
                "formula_per_image": f"{model.heads} heads * 2 measurement bases * shots",
                "per_image_at_256_shots": model.heads * 2 * 256,
                "per_image_at_1024_shots": model.heads * 2 * 1024,
                "per_validation_seed_at_256_shots": (
                    MODEL_I_VALIDATION_SAMPLES * model.heads * 2 * 256
                ),
                "per_validation_seed_at_1024_shots": (
                    MODEL_I_VALIDATION_SAMPLES * model.heads * 2 * 1024
                ),
                "three_seeds_at_1024_shots": (
                    len(SHOT_SEEDS)
                    * MODEL_I_VALIDATION_SAMPLES
                    * model.heads
                    * 2
                    * 1024
                ),
            },
            "paired_accuracy_endpoint": {
                "contrast": "finite-shot minus analytic accuracy",
                "resampling": "paired within true-class strata",
                "resamples": BOOTSTRAP_RESAMPLES,
                "analysis_seed": BOOTSTRAP_SEED,
                "two_sided_confidence_level": 0.95,
                "one_sided_lower_confidence_level": 0.95,
                "noninferior_margin": NONINFERIOR_MARGIN,
                "decision_rule": "one-sided 95% lower bound > noninferior margin",
            },
            "checkpoint_replay_acceptance": {
                "minimum_saved_class_agreement": REPLAY_MIN_CLASS_AGREEMENT,
                "maximum_probability_mae": REPLAY_MAX_PROBABILITY_MAE,
                "maximum_accuracy_abs_delta_from_summary": REPLAY_MAX_ACCURACY_DELTA,
            },
            "plugin": "sample means substituted into all analytic features",
            "ustat": (
                "quadratic products use ordered distinct-shot U-statistics; linear Pauli "
                "means use all shots"
                if args.include_ustat
                else "not requested"
            ),
            "limitations": (
                "Ideal finite-shot sampling from statevectors; no device, compilation, readout, "
                "decoherence, or gate-noise model. D4 invariance holds in distribution, not for "
                "each independent finite-shot realization."
            ),
        },
        "inputs": {
            "run_dir": str(run_dir),
            "development_cache": str(development_cache),
            "checkpoint_sha256": _sha256(run_dir / "best.pt"),
            "split_sha256": _sha256(run_dir / "split_indices.npz"),
            "development_manifest_sha256": _sha256(development_cache / "manifest.csv"),
            "best_epoch": int(summary["best_epoch"]),
            "resolved_model": {
                "heads": model.heads,
                "reuploads": model.core.reuploads,
                "encoder_variant": model.encoder.variant,
                "physics_variant": model.physics.variant,
                "include_context": model.include_context,
                "parameters": model.parameter_report(),
            },
        },
        "analytic": {
            "metrics": analytic_metrics,
            "saved_prediction_replay_agreement": saved_replay_agreement,
        },
        "finite_shot_replicates": records,
        "aggregate": aggregate,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
        },
        "artifacts": {
            "predictions": "predictions.npz",
            "prediction_arrays": sorted(arrays),
        },
    }
    _atomic_json(output_dir / "results.json", result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate finite-shot D4-ORQB predictions on Model-I validation only"
    )
    parser.add_argument("--run-dir", required=True, help="Completed quantum training run")
    parser.add_argument(
        "--development-cache",
        required=True,
        help="Canonical Model-I development cache (there is intentionally no test argument)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--include-ustat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report bias-corrected quadratic feature estimates",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate_finite_shots(args)
    concise = {
        "status": result["status"],
        "samples": result["scope"]["samples"],
        "analytic_accuracy": result["analytic"]["metrics"]["accuracy"],
        "aggregate": result["aggregate"],
    }
    print(f"FINITE_SHOT_SUMMARY {json.dumps(concise, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
