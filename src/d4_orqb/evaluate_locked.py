"""Audited, two-stage evaluation for the locked Model-I test set.

The ``freeze`` stage validates completed development-only runs and creates an
atomic, tamper-evident snapshot.  It deliberately has no test-cache argument.
The ``run-test`` stage verifies that snapshot, replays every validation
prediction, writes a durable test-access marker, and only then opens the test
cache.  Ensembles average probabilities (never logits), and all comparisons
are fixed in the frozen analysis plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from .data import CachedNPYDataset, make_loader
from .metrics import classification_metrics


SCHEMA_VERSION = 2
CLASSIFIER_KINDS = ("d4-orqb", "lenspinn-repaired")
RUN_ARTIFACTS = (
    "config.json",
    "data_report.json",
    "history.json",
    "parameter_report.json",
    "split_indices.npz",
    "last.pt",
    "best.pt",
    "best_validation_predictions.npz",
    "summary.json",
)
CACHE_ARTIFACTS = ("metadata.json", "images.npy", "labels.npy", "manifest.csv")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
Z_95 = 1.959963984540054
MAX_VALIDATION_REPLAY_PROBABILITY_ATOL = 5e-3
MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL = 2e-5
MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL = 3e-4
MAX_VALIDATION_REPLAY_METRIC_ATOL = 2e-4
MODEL_I_CLASSES = ["axion", "cdm", "no_sub"]
MODEL_I_DEVELOPMENT_COUNTS = {"axion": 28_897, "cdm": 29_772, "no_sub": 28_856}
MODEL_I_DEVELOPMENT_SAMPLES = 87_525
MODEL_I_TEST_SAMPLES = 15_000
MODEL_I_IMAGE_SIZE = 96
MODEL_I_SPLIT_SEED = 42
MODEL_I_VAL_FRACTION = 0.20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> Dict[str, int | str]:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"Expected a regular, non-symlink file: {path}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"Expected a regular, non-symlink file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return {
            "bytes": int(file_stat.st_size),
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def fingerprint_named_files(root: str | Path, names: Iterable[str]) -> Dict[str, Dict]:
    root = Path(root)
    return {name: file_fingerprint(root / name) for name in names}


def assert_fingerprints_equal(actual: Mapping, expected: Mapping, context: str) -> None:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{context} artifact set changed: actual={sorted(actual)} "
            f"expected={sorted(expected)}"
        )
    for name in sorted(expected):
        if actual[name] != expected[name]:
            raise RuntimeError(
                f"{context} artifact changed for {name}: "
                f"actual={actual[name]} expected={expected[name]}"
            )


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def fsync_directory(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assert_no_symlink_components(path: str | Path, context: str) -> Path:
    """Reject lexical aliases and every existing symlink in an absolute path.

    This uses ``lstat`` component by component, so checking an output path does
    not follow an attacker-controlled parent symlink into a protected cache.
    It is intentionally never called on the official-test path before the
    durable test-access marker.
    """

    path = Path(path).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{context} path must be absolute")
    normalized = Path(os.path.abspath(str(path)))
    if path != normalized:
        raise RuntimeError(f"{context} path must not contain lexical aliases: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{context} path contains a symlink component: {current}")
    return path


def assert_canonical_directory(path: str | Path, context: str) -> Path:
    path = assert_no_symlink_components(path, context)
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{context} is not a regular directory: {path}")
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"{context} directory is not canonical: {path}")
    return path


def atomic_json(path: str | Path, value, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise RuntimeError(f"Refusing to replace existing marker: {path}") from error
            temporary.unlink()
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.building-{os.getpid()}-{uuid.uuid4().hex}.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def validate_probabilities(probabilities: np.ndarray, *, name: str = "probabilities") -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or not len(probabilities):
        raise ValueError(f"{name} must be a non-empty [samples, classes] matrix")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} contains non-finite values")
    if (probabilities < -1e-12).any() or (probabilities > 1.0 + 1e-12).any():
        raise ValueError(f"{name} contains values outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6):
        raise ValueError(f"{name} rows do not sum to one")
    return probabilities


def probability_replay_diagnostics(
    actual_probabilities: np.ndarray, expected_probabilities: np.ndarray
) -> Dict[str, float | bool]:
    """Summarize pointwise replay drift without hiding rare large deviations."""

    actual = validate_probabilities(actual_probabilities, name="replayed probabilities")
    expected = validate_probabilities(expected_probabilities, name="frozen probabilities")
    if actual.shape != expected.shape:
        raise ValueError("Replayed and frozen probability shapes differ")
    absolute_difference = np.abs(actual - expected)
    return {
        "max_probability_absolute_difference": float(absolute_difference.max()),
        "mean_probability_absolute_difference": float(absolute_difference.mean()),
        "p99_probability_absolute_difference": float(
            np.quantile(absolute_difference, 0.99)
        ),
        "predicted_classes_exact": bool(
            np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
        ),
    }


def uniform_probability_ensemble(probability_matrices: Sequence[np.ndarray]) -> np.ndarray:
    if len(probability_matrices) < 2:
        raise ValueError("A frozen ensemble requires at least two members")
    checked = [
        validate_probabilities(value, name=f"ensemble member {index}")
        for index, value in enumerate(probability_matrices)
    ]
    shape = checked[0].shape
    if any(value.shape != shape for value in checked[1:]):
        raise ValueError("Ensemble probability shapes differ")
    result = np.mean(np.stack(checked, axis=0), axis=0, dtype=np.float64)
    return validate_probabilities(result, name="uniform ensemble")


def wilson_interval(successes: int, samples: int, z: float = Z_95) -> Dict[str, float | int]:
    successes, samples = int(successes), int(samples)
    if samples <= 0 or successes < 0 or successes > samples:
        raise ValueError("Wilson interval requires 0 <= successes <= samples and samples > 0")
    proportion = successes / samples
    denominator = 1.0 + z * z / samples
    centre = (proportion + z * z / (2.0 * samples)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / samples + z * z / (4.0 * samples * samples)
        )
        / denominator
    )
    low = max(0.0, centre - radius)
    high = min(1.0, centre + radius)
    if successes == 0:
        low = 0.0
    if successes == samples:
        high = 1.0
    return {
        "successes": successes,
        "samples": samples,
        "confidence": 0.95,
        "low": low,
        "high": high,
    }


def metrics_from_probabilities(
    labels: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str]
) -> Dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = validate_probabilities(probabilities)
    if probabilities.shape != (len(labels), len(class_names)):
        raise ValueError("Label/probability/class dimensions disagree")
    logits = np.log(np.clip(probabilities, 1e-300, 1.0))
    metrics = classification_metrics(labels, logits, list(class_names))
    successes = int((probabilities.argmax(axis=1) == labels).sum())
    metrics["accuracy_wilson95"] = wilson_interval(successes, len(labels))
    return metrics


def mcnemar_exact(
    labels: np.ndarray, probabilities_a: np.ndarray, probabilities_b: np.ndarray
) -> Dict[str, int | float]:
    labels = np.asarray(labels, dtype=np.int64)
    a = validate_probabilities(probabilities_a, name="McNemar A").argmax(axis=1)
    b = validate_probabilities(probabilities_b, name="McNemar B").argmax(axis=1)
    if len(a) != len(labels) or len(b) != len(labels):
        raise ValueError("McNemar arrays have different lengths")
    correct_a, correct_b = a == labels, b == labels
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_only, b_only)
        if 2 * lower >= discordant:
            p_value = 1.0
        else:
            # Exact Binomial(n, 1/2) lower tail, accumulated relative to
            # P(X=lower).  The log-domain anchor avoids both enormous integer
            # conversion and underflow for the 15k-example official test set.
            log_probability_at_lower = (
                math.lgamma(discordant + 1)
                - math.lgamma(lower + 1)
                - math.lgamma(discordant - lower + 1)
                - discordant * math.log(2.0)
            )
            relative_term = 1.0
            relative_sum = 1.0
            for index in range(lower, 0, -1):
                relative_term *= index / (discordant - index + 1)
                relative_sum += relative_term
            log_tail = log_probability_at_lower + math.log(relative_sum)
            p_value = min(1.0, 2.0 * math.exp(log_tail)) if log_tail > -746.0 else 0.0
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "two_sided_exact_p": float(p_value),
    }


def stratified_paired_bootstrap_accuracy(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 20260711,
    chunk_size: int = 128,
) -> Dict[str, float | int | str]:
    """Paired accuracy difference with class-stratified resampling."""

    labels = np.asarray(labels, dtype=np.int64)
    a = validate_probabilities(probabilities_a, name="bootstrap A").argmax(axis=1)
    b = validate_probabilities(probabilities_b, name="bootstrap B").argmax(axis=1)
    if len(a) != len(labels) or len(b) != len(labels):
        raise ValueError("Bootstrap arrays have different lengths")
    if samples <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap sample and chunk counts must be positive")
    delta = (a == labels).astype(np.float64) - (b == labels).astype(np.float64)
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("Bootstrap encountered an empty class stratum")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        totals = np.zeros(stop - start, dtype=np.float64)
        for indices in strata:
            draws = rng.integers(0, len(indices), size=(stop - start, len(indices)))
            totals += delta[indices[draws]].sum(axis=1)
        estimates[start:stop] = totals / len(labels)
    return {
        "estimand": "accuracy(A)-accuracy(B)",
        "resampling": "paired, stratified by true class with fixed stratum sizes",
        "difference": float(delta.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def validate_split_indices(
    labels: np.ndarray, train_indices: np.ndarray, val_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    train_raw, val_raw = np.asarray(train_indices), np.asarray(val_indices)
    if train_raw.dtype.kind not in "iu" or val_raw.dtype.kind not in "iu":
        raise RuntimeError("Split indices must have an integer dtype")
    train = train_raw.astype(np.int64, copy=False)
    val = val_raw.astype(np.int64, copy=False)
    if train.ndim != 1 or val.ndim != 1:
        raise RuntimeError("Split arrays must be one-dimensional")
    if not len(train) or not len(val):
        raise RuntimeError("Training and validation splits must both be non-empty")
    if len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val):
        raise RuntimeError("Split contains duplicate indices")
    if train.min() < 0 or val.min() < 0 or train.max() >= len(labels) or val.max() >= len(labels):
        raise RuntimeError("Split contains out-of-range indices")
    if np.intersect1d(train, val, assume_unique=True).size:
        raise RuntimeError("Training and validation splits overlap")
    combined = np.sort(np.concatenate((train, val)))
    if not np.array_equal(combined, np.arange(len(labels), dtype=np.int64)):
        raise RuntimeError("Split does not cover the complete development cache")
    return train, val


def canonical_model_i_split(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(MODEL_I_SPLIT_SEED)
    train_parts, val_parts = [], []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        validation_samples = int(round(len(indices) * MODEL_I_VAL_FRACTION))
        val_parts.append(indices[:validation_samples])
        train_parts.append(indices[validation_samples:])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    return train.astype(np.int64), val.astype(np.int64)


def load_validation_predictions(
    path: str | Path,
    expected_indices: np.ndarray,
    development_labels: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"indices", "labels", "logits"}
        if not required.issubset(data.files):
            raise RuntimeError(f"Validation predictions lack {sorted(required - set(data.files))}")
        indices_raw, labels_raw = np.asarray(data["indices"]), np.asarray(data["labels"])
        if indices_raw.dtype.kind not in "iu" or labels_raw.dtype.kind not in "iu":
            raise RuntimeError("Validation prediction indices/labels must be integer arrays")
        indices = indices_raw.astype(np.int64, copy=False)
        labels = labels_raw.astype(np.int64, copy=False)
        logits = np.asarray(data["logits"])
        stored_probabilities = (
            np.asarray(data["probabilities"], dtype=np.float64)
            if "probabilities" in data.files
            else None
        )
    expected_indices = np.asarray(expected_indices, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError("Validation prediction order does not exactly match the frozen split")
    expected_labels = np.asarray(development_labels, dtype=np.int64)[expected_indices]
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("Validation prediction labels disagree with the development cache")
    if logits.dtype.kind != "f" or logits.shape != (
        len(indices),
        len(class_names),
    ) or not np.isfinite(logits).all():
        raise RuntimeError("Validation logits have an invalid shape or non-finite values")
    probabilities = softmax_numpy(logits)
    if stored_probabilities is not None:
        validate_probabilities(stored_probabilities, name="stored validation probabilities")
        if not np.allclose(stored_probabilities, probabilities, rtol=2e-6, atol=2e-6):
            raise RuntimeError("Stored validation probabilities do not match softmax(logits)")
    return {
        "indices": indices,
        "labels": labels,
        "logits": logits,
        "probabilities": probabilities,
    }


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSON artifact: {path}") from error


def validate_cache_structure(
    cache_dir: str | Path, expected_classes: Sequence[str] | None = None
) -> Dict:
    cache_dir = Path(cache_dir)
    for name in CACHE_ARTIFACTS:
        path = cache_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Cache is incomplete or unsafe: {path}")
    metadata = _read_json(cache_dir / "metadata.json")
    if metadata.get("complete") is not True:
        raise RuntimeError(f"Cache metadata is not complete: {cache_dir}")
    classes = metadata.get("classes")
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
        raise RuntimeError("Cache class metadata is invalid")
    if expected_classes is not None and list(expected_classes) != classes:
        raise RuntimeError(f"Cache class order mismatch: {classes} != {list(expected_classes)}")
    labels = np.load(cache_dir / "labels.npy", allow_pickle=False)
    images = np.load(cache_dir / "images.npy", mmap_mode="r", allow_pickle=False)
    samples = int(metadata.get("samples", -1))
    image_size = int(metadata.get("image_size", -1))
    if labels.shape != (samples,) or images.shape != (samples, image_size, image_size):
        raise RuntimeError("Cache array shapes disagree with metadata")
    if labels.dtype.kind not in "iu" or images.dtype != np.float16:
        raise RuntimeError("Cache arrays must use integer labels and float16 images")
    labels = np.asarray(labels, dtype=np.int64)
    if samples <= 0 or (labels < 0).any() or (labels >= len(classes)).any():
        raise RuntimeError("Cache labels are invalid")
    digests: set[str] = set()
    digest_rows: List[str] = []
    relative_paths: List[str] = []
    relative_path_set: set[str] = set()
    rows = 0
    with (cache_dir / "manifest.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"index", "relative_path", "class", "label", "sha256_visible"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise RuntimeError("Cache manifest columns are incomplete")
        for row in reader:
            if rows >= samples:
                raise RuntimeError("Cache manifest has more rows than cache arrays")
            index = int(row["index"])
            if index != rows or int(row["label"]) != int(labels[index]):
                raise RuntimeError("Cache manifest index/label order is invalid")
            if row["class"] != classes[int(labels[index])]:
                raise RuntimeError("Cache manifest class name disagrees with labels")
            relative_path = row["relative_path"]
            parsed_relative = Path(relative_path)
            if (
                not relative_path
                or relative_path in relative_path_set
                or parsed_relative.is_absolute()
                or ".." in parsed_relative.parts
            ):
                raise RuntimeError("Cache manifest contains an unsafe relative path")
            relative_path_set.add(relative_path)
            digest = row["sha256_visible"]
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise RuntimeError("Cache manifest contains an invalid SHA-256 digest")
            digests.add(digest)
            digest_rows.append(digest)
            relative_paths.append(relative_path)
            rows += 1
    if rows != samples or len(digests) != samples:
        raise RuntimeError("Cache manifest is incomplete or contains duplicate visible images")
    return {
        "path": str(cache_dir.resolve()),
        "metadata": metadata,
        "classes": classes,
        "labels": labels,
        "digests": digests,
        "digest_rows": np.asarray(digest_rows, dtype="U64"),
        "relative_paths": np.asarray(relative_paths, dtype=str),
    }


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_model_config(
    kind: str, config: Mapping, state: Mapping[str, torch.Tensor]
) -> Tuple[Dict, List[Dict[str, str]]]:
    resolved = dict(config)
    migrations: List[Dict[str, str]] = []
    if kind == "d4-orqb":
        if bool(resolved.get("haar_subtype_max_envelope", False)):
            raise RuntimeError(
                "Locked evaluation does not support the derived "
                "max-preserving Haar subtype envelope"
            )
        if bool(resolved.get("shared_late_refinement", False)) or (
            "encoder.shared_refinement_gates" in state
        ):
            raise RuntimeError(
                "Locked evaluation does not yet support shared late refinement"
            )
        if bool(resolved.get("haar_subtype_residual", False)) or any(
            str(key).startswith("haar_subtype_residual.") for key in state
        ):
            raise RuntimeError(
                "Locked evaluation does not yet support the image-derived "
                "Haar subtype residual"
            )
        for key in ("heads", "reuploads", "core", "include_context", "image_size"):
            if key not in resolved:
                raise RuntimeError(f"D4-ORQB config is missing required field: {key}")
        if "encoder_variant" not in resolved:
            stem = state.get("encoder.stem.0.weight")
            final = state.get("encoder.final.0.weight")
            signature = (
                tuple(stem.shape) if stem is not None else None,
                tuple(final.shape) if final is not None else None,
            )
            if signature[0] is None or signature[1] is None:
                raise RuntimeError(
                    f"Cannot infer legacy encoder variant from missing tensors: {signature}"
                )
            if signature[0][0] == 16 and signature[1][0] == 128:
                resolved["encoder_variant"] = "tiny"
            elif signature[0][0] == 24 and signature[1][0] == 192:
                resolved["encoder_variant"] = "small"
            else:
                raise RuntimeError(f"Cannot infer legacy encoder variant from {signature}")
            migrations.append(
                {
                    "field": "encoder_variant",
                    "value": str(resolved["encoder_variant"]),
                    "evidence": "strict checkpoint encoder stem/final shapes",
                }
            )
        if "physics_variant" not in resolved:
            stem = state.get("encoder.stem.0.weight")
            if stem is None or stem.ndim != 4 or int(stem.shape[1]) not in (8, 10):
                raise RuntimeError("Cannot infer legacy physics variant from checkpoint stem")
            resolved["physics_variant"] = "base" if int(stem.shape[1]) == 8 else "radial"
            migrations.append(
                {
                    "field": "physics_variant",
                    "value": str(resolved["physics_variant"]),
                    "evidence": "strict checkpoint encoder input-channel count",
                }
            )
    elif kind == "lenspinn-repaired":
        for key in (
            "image_size",
            "patch_size",
            "faithful_softmax",
            "reconstruction",
            "retain_archived_unused_block",
        ):
            if key not in resolved:
                raise RuntimeError(f"LensPINN-repaired config is missing required field: {key}")
    return resolved, migrations


def build_model(kind: str, config: Mapping, classes: int):
    if kind == "d4-orqb":
        from .model import D4OrbitClassifier

        return D4OrbitClassifier(
            num_classes=classes,
            heads=int(config["heads"]),
            reuploads=int(config["reuploads"]),
            core=str(config["core"]),
            include_context=bool(config["include_context"]),
            encoder_variant=str(config["encoder_variant"]),
            physics_variant=str(config["physics_variant"]),
        )
    if kind == "lenspinn-repaired":
        if bool(config.get("faithful_softmax", False)):
            raise RuntimeError("LensPINN-repaired forbids Softmax-before-CrossEntropy")
        if config.get("reconstruction", "differentiable") != "differentiable":
            raise RuntimeError("LensPINN-repaired requires differentiable reconstruction")
        try:
            from .lenspinn import LensPINNSmall

            return LensPINNSmall(
                image_size=int(config["image_size"]),
                patch_size=int(config["patch_size"]),
                pretrained=False,
                logits_fix=True,
                reconstruction="differentiable",
                retain_archived_unused_block=bool(
                    config["retain_archived_unused_block"]
                ),
            )
        except RuntimeError as error:
            if "requires timm" in str(error):
                raise RuntimeError(
                    "LensPINN-repaired can only be frozen/evaluated when timm==0.9.16 is available"
                ) from error
            raise
    raise ValueError(f"Unknown classifier kind: {kind}")


def load_model_strict(kind: str, config: Mapping, checkpoint_path: Path, classes: int):
    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint has no model state: {checkpoint_path}")
    state = checkpoint["model"]
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"Checkpoint model state is invalid: {checkpoint_path}")
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise RuntimeError("Checkpoint state must map string names to tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Checkpoint tensor contains non-finite values: {name}")
    resolved_config, migrations = resolve_model_config(kind, config, state)
    model = build_model(kind, resolved_config, classes)
    model.load_state_dict(state, strict=True)
    return model, checkpoint, resolved_config, migrations


def _metric_close(actual, expected, name: str, tolerance: float = 1e-5) -> float:
    """Require an absolute metric-drift bound and return the largest drift."""

    maximum_difference = 0.0
    for key in (
        "samples",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auc_ovr",
        "nll",
        "brier",
        "ece_15",
    ):
        if key not in expected or key not in actual:
            raise RuntimeError(f"Missing {name} metric: {key}")
        actual_value, expected_value = float(actual[key]), float(expected[key])
        difference = abs(actual_value - expected_value)
        if not math.isfinite(difference) or difference > tolerance:
            raise RuntimeError(
                f"{name} metric mismatch for {key}: actual={actual[key]} expected={expected[key]}"
            )
        maximum_difference = max(maximum_difference, difference)
    if actual.get("confusion_matrix") != expected.get("confusion_matrix"):
        raise RuntimeError(f"{name} confusion matrix changed")
    if set(actual.get("per_class", {})) != set(expected.get("per_class", {})):
        raise RuntimeError(f"{name} per-class metric keys changed")
    for class_name in actual.get("per_class", {}):
        for key in ("precision", "recall", "f1", "auc_ovr", "support"):
            actual_value = float(actual["per_class"][class_name][key])
            expected_value = float(expected["per_class"][class_name][key])
            difference = abs(actual_value - expected_value)
            if not math.isfinite(difference) or difference > tolerance:
                raise RuntimeError(f"{name} per-class metric changed: {class_name}.{key}")
            maximum_difference = max(maximum_difference, difference)
    return maximum_difference


def _validate_completed_run(
    run_dir: Path,
    kind: str,
    development: Mapping,
    settle_seconds: float,
) -> Dict:
    run_dir = assert_canonical_directory(run_dir, "source run")
    unfinished = [path for path in run_dir.rglob("*") if "building-" in path.name]
    if unfinished:
        raise RuntimeError(f"Run contains unfinished atomic artifacts: {unfinished[:3]}")
    for name in RUN_ARTIFACTS:
        path = run_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Run is incomplete or unsafe: {path}")
    if (run_dir / "test_predictions.npz").exists():
        raise RuntimeError(f"Run already contains test predictions: {run_dir}")
    newest = max((run_dir / name).stat().st_mtime for name in RUN_ARTIFACTS)
    if time.time() - newest < settle_seconds:
        raise RuntimeError(
            f"Run has not been settled for {settle_seconds:g}s and may still be mutable: {run_dir}"
        )
    summary_mtime = (run_dir / "summary.json").stat().st_mtime_ns
    for name in ("history.json", "last.pt", "best.pt", "best_validation_predictions.npz"):
        if (run_dir / name).stat().st_mtime_ns > summary_mtime:
            raise RuntimeError(f"Run artifact {name} is newer than final summary")

    before = fingerprint_named_files(run_dir, RUN_ARTIFACTS)
    config = _read_json(run_dir / "config.json")
    report = _read_json(run_dir / "data_report.json")
    summary = _read_json(run_dir / "summary.json")
    history = _read_json(run_dir / "history.json")
    parameters = _read_json(run_dir / "parameter_report.json")
    if config.get("evaluate_test") is not False:
        raise RuntimeError("Frozen runs must have evaluate_test=false")
    if summary.get("official_test_evaluated") is not False:
        raise RuntimeError("Run summary does not attest that official test stayed locked")
    if report.get("official_test_locked_during_selection") is not True:
        raise RuntimeError("Run data report does not attest a locked official test")
    cache_opened = report.get("official_test_cache_opened")
    if cache_opened is False:
        if "test" in report or "digest_disjoint" in report:
            raise RuntimeError("Development-only run unexpectedly contains test metadata")
        test_isolation = "test-cache-not-opened-by-training-run"
    elif cache_opened is None and "test" in report and "digest_disjoint" in report:
        # Early runs prepared and hashed the cache but never performed model
        # inference. They remain usable with an explicit prospective-lock
        # disclosure; absence of predictions and summary/config attestations
        # above are still mandatory.
        test_isolation = "legacy-cache-metadata-seen-no-model-inference"
    else:
        raise RuntimeError("Run does not provide an acceptable official-test isolation record")
    if config.get("max_train_per_class") is not None or config.get("max_val_per_class") is not None:
        raise RuntimeError("Subset/pilot runs cannot be frozen for official evaluation")
    if int(config.get("split_seed", -1)) != MODEL_I_SPLIT_SEED or not math.isclose(
        float(config.get("val_fraction", -1.0)), MODEL_I_VAL_FRACTION, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Run does not use the canonical Model-I seed-42 80/20 split")
    if not isinstance(history, list) or not history:
        raise RuntimeError("Run history is empty")
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch <= 0 or best_epoch not in {int(row.get("epoch", -1)) for row in history}:
        raise RuntimeError("Summary best_epoch is absent from history")
    selected_epoch, selected_accuracy = -1, -math.inf
    for row in history:
        validation = row.get("validation")
        if not isinstance(validation, Mapping) or "accuracy" not in validation:
            raise RuntimeError("Run history lacks validation accuracy")
        accuracy = float(validation["accuracy"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise RuntimeError("Run history contains an invalid validation accuracy")
        if accuracy > selected_accuracy + 1e-12:
            selected_accuracy = accuracy
            selected_epoch = int(row.get("epoch", -1))
    if best_epoch != selected_epoch:
        raise RuntimeError(
            "Summary best_epoch disagrees with the training selection rule: "
            f"actual={best_epoch} expected={selected_epoch}"
        )

    class_names = report.get("class_names")
    if class_names is None and isinstance(report.get("development"), dict):
        class_names = report["development"].get("classes")
    if class_names != development["classes"]:
        raise RuntimeError("Run class order disagrees with development cache")
    if int(config.get("image_size", -1)) != int(development["metadata"]["image_size"]):
        raise RuntimeError("Run image size disagrees with development cache")
    if int(report.get("train_size", -1)) <= 0 or int(report.get("validation_size", -1)) <= 0:
        raise RuntimeError("Run data report lacks full split sizes")

    with np.load(run_dir / "split_indices.npz", allow_pickle=False) as split:
        if not {"train", "val"}.issubset(split.files):
            raise RuntimeError("Run split artifact lacks train/val arrays")
        train, val = validate_split_indices(
            development["labels"], split["train"], split["val"]
        )
    if int(report["train_size"]) != len(train) or int(report["validation_size"]) != len(val):
        raise RuntimeError("Run data report sizes disagree with split artifact")
    if not np.array_equal(train, development["canonical_train_indices"]) or not np.array_equal(
        val, development["canonical_val_indices"]
    ):
        raise RuntimeError("Run split arrays differ from the canonical Model-I split")
    predictions = load_validation_predictions(
        run_dir / "best_validation_predictions.npz",
        val,
        development["labels"],
        class_names,
    )
    calculated_metrics = classification_metrics(
        predictions["labels"], predictions["logits"], list(class_names)
    )
    _metric_close(calculated_metrics, summary.get("validation", {}), "summary validation")
    best_history_row = next(row for row in history if int(row.get("epoch", -1)) == best_epoch)
    _metric_close(
        calculated_metrics,
        best_history_row.get("validation", {}),
        "best-history validation",
    )
    model, checkpoint, resolved_config, migrations = load_model_strict(
        kind, config, run_dir / "best.pt", len(class_names)
    )
    if int(checkpoint.get("epoch", -1)) != best_epoch:
        raise RuntimeError("Checkpoint epoch disagrees with summary best_epoch")
    history_epochs = [int(row.get("epoch", -1)) for row in history]
    if history_epochs != list(range(1, len(history) + 1)):
        raise RuntimeError("Run history epochs are incomplete or out of order")
    last_checkpoint = _torch_load(run_dir / "last.pt")
    if (
        not isinstance(last_checkpoint, dict)
        or "model" not in last_checkpoint
        or int(last_checkpoint.get("epoch", -1)) != history_epochs[-1]
    ):
        raise RuntimeError("Last checkpoint is incomplete or disagrees with run history")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if int(parameters.get("total", -1)) != trainable:
        raise RuntimeError(
            f"Parameter report disagrees with strict model: {parameters.get('total')} != {trainable}"
        )
    del model, checkpoint, last_checkpoint
    after = fingerprint_named_files(run_dir, RUN_ARTIFACTS)
    assert_fingerprints_equal(after, before, f"run {run_dir}")
    return {
        "config": config,
        "resolved_config": resolved_config,
        "config_migrations": migrations,
        "summary": summary,
        "parameters": parameters,
        "train_indices": train,
        "val_indices": val,
        "predictions": predictions,
        "source_fingerprints": before,
        "test_isolation": test_isolation,
    }


def _parse_member(value: str) -> Tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("member must be NAME=KIND=/absolute/run/path")
    name, kind, path_text = parts
    if not SAFE_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(f"unsafe member name: {name}")
    if kind not in CLASSIFIER_KINDS:
        raise argparse.ArgumentTypeError(f"kind must be one of {CLASSIFIER_KINDS}")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("run path must be absolute")
    return name, kind, path


def _parse_ensemble(value: str) -> Tuple[str, List[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("ensemble must be NAME=MEMBER_A,MEMBER_B,...")
    name, member_text = value.split("=", 1)
    members = [item for item in member_text.split(",") if item]
    if not SAFE_NAME.fullmatch(name) or len(members) < 2 or len(set(members)) != len(members):
        raise argparse.ArgumentTypeError("ensemble name/members are invalid")
    return name, members


def _parse_comparison(value: str) -> Tuple[str, str, float | None]:
    parts = value.split(",")
    if len(parts) not in (2, 3) or parts[0] == parts[1]:
        raise argparse.ArgumentTypeError(
            "comparison must be DISTINCT_A,DISTINCT_B[,MIN_ACCEPTABLE_A_MINUS_B]"
        )
    margin = float(parts[2]) if len(parts) == 3 else None
    if margin is not None and not math.isfinite(margin):
        raise argparse.ArgumentTypeError("minimum acceptable accuracy difference must be finite")
    if margin is not None and not -1.0 <= margin <= 1.0:
        raise argparse.ArgumentTypeError(
            "minimum acceptable accuracy difference must be within [-1, 1]"
        )
    return parts[0], parts[1], margin


def _validate_fingerprint_map(value, names: Sequence[str], context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise RuntimeError(f"{context} has an invalid artifact set")
    for name in names:
        fingerprint = value[name]
        if not isinstance(fingerprint, Mapping) or set(fingerprint) != {"bytes", "sha256"}:
            raise RuntimeError(f"{context} has an invalid fingerprint for {name}")
        size, digest = fingerprint["bytes"], fingerprint["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError(f"{context} has an invalid byte count for {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"{context} has an invalid SHA-256 for {name}")


def _validate_replay_tolerances(
    probability_atol,
    probability_mean_atol,
    probability_p99_atol,
    metric_atol,
    *,
    error_type=ValueError,
) -> Dict[str, float]:
    specifications = (
        (
            "maximum probability",
            "probability_atol",
            probability_atol,
            MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
        ),
        (
            "mean probability",
            "probability_mean_atol",
            probability_mean_atol,
            MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
        ),
        (
            "p99 probability",
            "probability_p99_atol",
            probability_p99_atol,
            MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
        ),
        ("metric", "metric_atol", metric_atol, MAX_VALIDATION_REPLAY_METRIC_ATOL),
    )
    validated: Dict[str, float] = {}
    for label, key, raw_value, upper_bound in specifications:
        if isinstance(raw_value, bool):
            raise error_type(f"Validation replay {label} tolerance is invalid")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise error_type(f"Validation replay {label} tolerance is invalid") from error
        if not math.isfinite(value) or not 0.0 <= value <= upper_bound:
            raise error_type(
                f"Validation replay {label} tolerance is outside [0, {upper_bound:g}]"
            )
        validated[key] = value
    return validated


def validate_frozen_manifest(manifest: Mapping) -> None:
    """Validate every frozen analysis-plan reference before test access."""

    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported or invalid frozen manifest schema")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping) or not SAFE_NAME.fullmatch(str(protocol.get("id", ""))):
        raise RuntimeError("Frozen protocol identity is invalid")
    if (
        protocol.get("two_stage") is not True
        or protocol.get("test_unavailable_during_freeze") is not True
        or protocol.get("validation_replay_required_before_test_marker") is not True
        or protocol.get("ensemble_rule")
        != "unweighted arithmetic mean of member probabilities"
    ):
        raise RuntimeError("Frozen protocol safety invariants are invalid")
    bootstrap_samples = protocol.get("bootstrap_samples")
    analysis_seed = protocol.get("analysis_seed")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
        or isinstance(analysis_seed, bool)
        or not isinstance(analysis_seed, int)
        or analysis_seed < 0
    ):
        raise RuntimeError("Frozen analysis settings are invalid")
    _validate_replay_tolerances(
        protocol.get("validation_replay_probability_atol"),
        protocol.get("validation_replay_probability_mean_atol"),
        protocol.get("validation_replay_probability_p99_atol"),
        protocol.get("validation_replay_metric_atol"),
        error_type=RuntimeError,
    )
    inference = protocol.get("inference")
    if (
        not isinstance(inference, Mapping)
        or isinstance(inference.get("batch_size"), bool)
        or not isinstance(inference.get("batch_size"), int)
        or inference["batch_size"] <= 0
        or isinstance(inference.get("workers"), bool)
        or not isinstance(inference.get("workers"), int)
        or inference["workers"] < 0
        or isinstance(inference.get("loader_seed"), bool)
        or not isinstance(inference.get("loader_seed"), int)
        or inference.get("autocast") != "cuda-bfloat16"
    ):
        raise RuntimeError("Frozen inference settings are invalid")

    development = manifest.get("development_cache")
    if not isinstance(development, Mapping):
        raise RuntimeError("Frozen development-cache record is invalid")
    development_path = development.get("path")
    if (
        not isinstance(development_path, str)
        or not Path(development_path).is_absolute()
        or os.path.abspath(development_path) != development_path
        or development.get("classes") != MODEL_I_CLASSES
        or development.get("samples") != MODEL_I_DEVELOPMENT_SAMPLES
        or development.get("image_size") != MODEL_I_IMAGE_SIZE
    ):
        raise RuntimeError("Frozen development-cache identity is invalid")
    _validate_fingerprint_map(
        development.get("artifacts"), CACHE_ARTIFACTS, "frozen development cache"
    )

    expected_test = manifest.get("expected_official_test")
    if not isinstance(expected_test, Mapping):
        raise RuntimeError("Frozen official-test record is invalid")
    test_path = expected_test.get("canonical_job_path")
    if (
        not isinstance(test_path, str)
        or not Path(test_path).is_absolute()
        or os.path.abspath(test_path) != test_path
        or expected_test.get("classes") != MODEL_I_CLASSES
        or expected_test.get("samples") != MODEL_I_TEST_SAMPLES
        or expected_test.get("image_size") != MODEL_I_IMAGE_SIZE
    ):
        raise RuntimeError("Frozen official-test identity is invalid")
    artifact_sha256 = expected_test.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or set(artifact_sha256) != set(CACHE_ARTIFACTS):
        raise RuntimeError("Frozen official-test artifact set is invalid")
    for name, digest in artifact_sha256.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Frozen official-test SHA-256 is invalid for {name}")

    members = manifest.get("members")
    if not isinstance(members, Mapping) or not members:
        raise RuntimeError("Frozen manifest has no members")
    for name, member in members.items():
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise RuntimeError(f"Unsafe sealed member name: {name}")
        if not isinstance(member, Mapping) or member.get("kind") not in CLASSIFIER_KINDS:
            raise RuntimeError(f"Frozen member kind is invalid: {name}")
        source_run = member.get("source_run")
        if (
            not isinstance(source_run, str)
            or not Path(source_run).is_absolute()
            or os.path.abspath(source_run) != source_run
            or not isinstance(member.get("resolved_config"), Mapping)
            or not isinstance(member.get("config_migrations"), list)
        ):
            raise RuntimeError(f"Frozen member metadata is invalid: {name}")
        _validate_fingerprint_map(
            member.get("artifacts"), RUN_ARTIFACTS, f"frozen member {name}"
        )

    ensembles = manifest.get("ensembles")
    if not isinstance(ensembles, Mapping):
        raise RuntimeError("Frozen ensemble plan is invalid")
    member_names = set(members)
    if member_names.intersection(ensembles):
        raise RuntimeError("Frozen ensemble names collide with member names")
    for name, children in ensembles.items():
        if (
            not isinstance(name, str)
            or not SAFE_NAME.fullmatch(name)
            or not isinstance(children, list)
            or len(children) < 2
            or len(set(children)) != len(children)
            or not set(children).issubset(member_names)
        ):
            raise RuntimeError(f"Frozen ensemble is invalid: {name}")
    result_names = member_names | set(ensembles)

    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise RuntimeError("Frozen manifest has no predeclared comparisons")
    comparison_pairs = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise RuntimeError("Frozen comparison is invalid")
        a, b = comparison.get("a"), comparison.get("b")
        if a not in result_names or b not in result_names or a == b:
            raise RuntimeError(f"Frozen comparison references invalid results: {a},{b}")
        margin = comparison.get("minimum_acceptable_accuracy_difference")
        if margin is not None:
            try:
                margin = float(margin)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Frozen comparison margin is invalid") from error
            if not math.isfinite(margin) or not -1.0 <= margin <= 1.0:
                raise RuntimeError("Frozen comparison margin is outside [-1, 1]")
        comparison_pairs.append(frozenset((a, b)))
    if len(set(comparison_pairs)) != len(comparison_pairs):
        raise RuntimeError("Frozen comparisons contain duplicate or reversed pairs")

    validation_metrics = manifest.get("validation_plan_metrics")
    if not isinstance(validation_metrics, Mapping) or set(validation_metrics) != result_names:
        raise RuntimeError("Frozen validation-plan metric names are invalid")


def _code_fingerprints(kinds: Iterable[str]) -> Dict[str, Dict]:
    # Membership tests consume generators.  Materialize once so a LensPINN
    # member cannot disappear from the code seal merely because it precedes a
    # D4-ORQB member (or is the only member).
    kinds = set(kinds)
    package = Path(__file__).resolve().parent
    names = {"__init__.py", "evaluate_locked.py", "data.py", "metrics.py"}
    if "d4-orqb" in kinds:
        names.update(("model.py", "quantum.py"))
    if "lenspinn-repaired" in kinds:
        names.add("lenspinn.py")
    return fingerprint_named_files(package, sorted(names))


def runtime_fingerprint(kinds: Iterable[str]) -> Dict[str, str | int | None]:
    kinds = set(kinds)
    result: Dict[str, str | int | None] = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if "lenspinn-repaired" in kinds:
        try:
            import timm
            import torchvision
        except ImportError as error:
            raise RuntimeError(
                "LensPINN-repaired lock requires pinned timm and torchvision"
            ) from error
        if timm.__version__ != "0.9.16":
            raise RuntimeError(
                "LensPINN-repaired lock requires the training-pinned timm==0.9.16"
            )
        result["timm"] = timm.__version__
        result["torchvision"] = torchvision.__version__
    return result


def freeze(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Freeze must run in the same CUDA job environment as locked evaluation"
        )
    seal_dir = Path(args.seal_dir).expanduser()
    if not seal_dir.is_absolute():
        raise RuntimeError("--seal-dir must be absolute")
    if not SAFE_NAME.fullmatch(args.protocol_id):
        raise RuntimeError("--protocol-id contains unsafe characters")
    replay_tolerances = _validate_replay_tolerances(
        args.replay_atol,
        args.replay_mean_atol,
        args.replay_p99_atol,
        args.replay_metric_atol,
        error_type=RuntimeError,
    )
    for option, value in (
        ("--expected-development-manifest-sha256", args.expected_development_manifest_sha256),
        ("--expected-test-manifest-sha256", args.expected_test_manifest_sha256),
        ("--expected-test-images-sha256", args.expected_test_images_sha256),
        ("--expected-test-labels-sha256", args.expected_test_labels_sha256),
        ("--expected-test-metadata-sha256", args.expected_test_metadata_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"{option} must be a lowercase SHA-256")
    if int(args.expected_test_samples) != MODEL_I_TEST_SAMPLES:
        raise RuntimeError("Model-I official test must contain exactly 15,000 samples")
    expected_test_cache_path = os.path.abspath(
        os.path.expanduser(args.expected_test_cache_path)
    )
    if not Path(args.expected_test_cache_path).expanduser().is_absolute():
        raise RuntimeError("--expected-test-cache-path must be absolute")
    members = [_parse_member(value) for value in args.member]
    member_names = [name for name, _, _ in members]
    if len(set(member_names)) != len(member_names):
        raise RuntimeError("Member names must be unique")
    parsed_ensembles = [_parse_ensemble(value) for value in args.ensemble]
    ensemble_names = [name for name, _ in parsed_ensembles]
    if len(set(ensemble_names)) != len(ensemble_names):
        raise RuntimeError("Ensemble names must be unique")
    ensembles = dict(parsed_ensembles)
    if set(ensembles).intersection(member_names):
        raise RuntimeError("Ensemble names must not collide with member names")
    for name, children in ensembles.items():
        unknown = set(children) - set(member_names)
        if unknown:
            raise RuntimeError(f"Ensemble {name} contains unknown members: {sorted(unknown)}")
    result_names = member_names + list(ensembles)
    if not args.comparison:
        raise RuntimeError("At least one explicit predeclared --comparison is required")
    comparisons = [_parse_comparison(value) for value in args.comparison]
    for a, b, _ in comparisons:
        if a not in result_names or b not in result_names:
            raise RuntimeError(f"Comparison references an unknown result: {a},{b}")
    comparison_sets = [frozenset((a, b)) for a, b, _ in comparisons]
    if len(set(comparison_sets)) != len(comparison_sets):
        raise RuntimeError("Comparisons must be unique, including reversed pairs")

    development_cache = Path(args.development_cache).expanduser()
    if not development_cache.is_absolute():
        raise RuntimeError("--development-cache must be absolute")
    protected_paths = [
        ("seal", seal_dir),
        ("development", development_cache),
        ("expected-test", expected_test_cache_path),
        *((f"member-{name}", path) for name, _, path in members),
    ]
    for first_index, (first_name, first_path) in enumerate(protected_paths):
        for second_name, second_path in protected_paths[first_index + 1 :]:
            if lexical_paths_overlap(first_path, second_path):
                raise RuntimeError(
                    f"Unsafe overlapping {first_name}/{second_name} paths during freeze"
                )
    # The expected-test path has only been compared lexically up to here.  It
    # is now safe to inspect every other path without crossing the lock.
    seal_dir = assert_no_symlink_components(seal_dir, "seal")
    if seal_dir.exists():
        raise RuntimeError(f"Refusing to replace an existing seal directory: {seal_dir}")
    development_cache = assert_canonical_directory(
        development_cache, "development cache"
    )
    cache_before = fingerprint_named_files(development_cache, CACHE_ARTIFACTS)
    development = validate_cache_structure(development_cache)
    if development["classes"] != MODEL_I_CLASSES:
        raise RuntimeError("Development cache does not have canonical Model-I classes")
    if len(development["labels"]) != MODEL_I_DEVELOPMENT_SAMPLES:
        raise RuntimeError("Development cache does not have canonical Model-I sample count")
    if int(development["metadata"]["image_size"]) != MODEL_I_IMAGE_SIZE:
        raise RuntimeError("Locked Model-I protocol requires the selected 96-pixel cache")
    actual_counts = {
        name: int((development["labels"] == index).sum())
        for index, name in enumerate(development["classes"])
    }
    if actual_counts != MODEL_I_DEVELOPMENT_COUNTS:
        raise RuntimeError("Development cache class counts do not match canonical Model I")
    if cache_before["manifest.csv"]["sha256"] != args.expected_development_manifest_sha256:
        raise RuntimeError("Development cache manifest does not match the declared protocol hash")
    canonical_train, canonical_val = canonical_model_i_split(development["labels"])
    development["canonical_train_indices"] = canonical_train
    development["canonical_val_indices"] = canonical_val

    assert_no_symlink_components(seal_dir.parent, "seal parent")
    seal_dir.parent.mkdir(parents=True, exist_ok=True)
    assert_canonical_directory(seal_dir.parent, "seal parent")
    staging = seal_dir.with_name(
        f".{seal_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    try:
        frozen_members: Dict[str, Dict] = {}
        reference_train = reference_val = reference_labels = None
        validation_probabilities: Dict[str, np.ndarray] = {}
        checkpoint_hashes: set[str] = set()
        for name, kind, run_dir in members:
            inspected = _validate_completed_run(
                run_dir, kind, development, float(args.settle_seconds)
            )
            if reference_train is None:
                reference_train = inspected["train_indices"]
                reference_val = inspected["val_indices"]
                reference_labels = inspected["predictions"]["labels"]
            elif (
                not np.array_equal(reference_train, inspected["train_indices"])
                or not np.array_equal(reference_val, inspected["val_indices"])
                or not np.array_equal(reference_labels, inspected["predictions"]["labels"])
            ):
                raise RuntimeError("Frozen members do not use the exact same development split")

            destination = staging / "members" / name
            destination.mkdir(parents=True)
            for artifact in RUN_ARTIFACTS:
                shutil.copy2(run_dir / artifact, destination / artifact)
                fsync_file(destination / artifact)
            fsync_directory(destination)
            copied = fingerprint_named_files(destination, RUN_ARTIFACTS)
            checkpoint_hash = copied["best.pt"]["sha256"]
            if checkpoint_hash in checkpoint_hashes:
                raise RuntimeError("Distinct members contain the same best checkpoint")
            checkpoint_hashes.add(checkpoint_hash)
            assert_fingerprints_equal(
                copied, inspected["source_fingerprints"], f"sealed copy for {name}"
            )
            source_after_copy = fingerprint_named_files(run_dir, RUN_ARTIFACTS)
            assert_fingerprints_equal(
                source_after_copy, inspected["source_fingerprints"], f"source run {name}"
            )
            # Strict-load the sealed checkpoint too, so the manifest never seals
            # a copy that was only partially written.
            sealed_model, sealed_checkpoint, sealed_config, sealed_migrations = load_model_strict(
                kind,
                inspected["config"],
                destination / "best.pt",
                len(development["classes"]),
            )
            if (
                sealed_config != inspected["resolved_config"]
                or sealed_migrations != inspected["config_migrations"]
            ):
                raise RuntimeError("Sealed checkpoint architecture resolution changed")
            del sealed_model, sealed_checkpoint
            validation_probabilities[name] = inspected["predictions"]["probabilities"]
            frozen_members[name] = {
                "kind": kind,
                "source_run": str(run_dir.resolve()),
                "artifacts": copied,
                "best_epoch": int(inspected["summary"]["best_epoch"]),
                "parameters": inspected["parameters"],
                "test_isolation": inspected["test_isolation"],
                "resolved_config": inspected["resolved_config"],
                "config_migrations": inspected["config_migrations"],
                "validation": metrics_from_probabilities(
                    reference_labels,
                    validation_probabilities[name],
                    development["classes"],
                ),
            }

        fsync_directory(staging / "members")
        for name, children in ensembles.items():
            validation_probabilities[name] = uniform_probability_ensemble(
                [validation_probabilities[child] for child in children]
            )
        validation_plan_metrics = {
            name: metrics_from_probabilities(
                reference_labels, probabilities, development["classes"]
            )
            for name, probabilities in validation_probabilities.items()
        }
        cache_after = fingerprint_named_files(development_cache, CACHE_ARTIFACTS)
        assert_fingerprints_equal(cache_after, cache_before, "development cache during freeze")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "protocol": {
                "id": args.protocol_id,
                "dataset": "DeepLense Model I",
                "two_stage": True,
                "test_unavailable_during_freeze": True,
                "prospective_lock_scope": (
                    "This seals the present inference and analysis pass; the official Model-I "
                    "test is not globally pristine because older repository notebooks used it."
                ),
                "validation_replay_required_before_test_marker": True,
                "ensemble_rule": "unweighted arithmetic mean of member probabilities",
                "bootstrap_samples": int(args.bootstrap_samples),
                "analysis_seed": int(args.analysis_seed),
                "validation_replay_probability_atol": replay_tolerances[
                    "probability_atol"
                ],
                "validation_replay_probability_mean_atol": replay_tolerances[
                    "probability_mean_atol"
                ],
                "validation_replay_probability_p99_atol": replay_tolerances[
                    "probability_p99_atol"
                ],
                "validation_replay_metric_atol": replay_tolerances["metric_atol"],
                "inference": {
                    "batch_size": int(args.inference_batch_size),
                    "workers": int(args.inference_workers),
                    "loader_seed": int(args.inference_loader_seed),
                    "autocast": "cuda-bfloat16",
                },
            },
            "expected_official_test": {
                "artifact_sha256": {
                    "manifest.csv": args.expected_test_manifest_sha256,
                    "images.npy": args.expected_test_images_sha256,
                    "labels.npy": args.expected_test_labels_sha256,
                    "metadata.json": args.expected_test_metadata_sha256,
                },
                "samples": int(args.expected_test_samples),
                "image_size": int(development["metadata"]["image_size"]),
                "classes": development["classes"],
                "canonical_job_path": expected_test_cache_path,
            },
            "development_cache": {
                "path": str(development_cache.resolve()),
                "artifacts": cache_before,
                "classes": development["classes"],
                "samples": int(len(development["labels"])),
                "image_size": int(development["metadata"]["image_size"]),
            },
            "code": _code_fingerprints(kind for _, kind, _ in members),
            "runtime": runtime_fingerprint(kind for _, kind, _ in members),
            "members": frozen_members,
            "ensembles": ensembles,
            "comparisons": [
                {
                    "a": a,
                    "b": b,
                    "minimum_acceptable_accuracy_difference": margin,
                }
                for a, b, margin in comparisons
            ],
            "validation_plan_metrics": validation_plan_metrics,
        }
        atomic_json(staging / "manifest.json", manifest)
        manifest_digest = sha256_file(staging / "manifest.json")
        atomic_json(
            staging / "seal.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": utc_now(),
                "manifest_sha256": manifest_digest,
            },
        )
        os.replace(staging, seal_dir)
        fsync_directory(seal_dir.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "seal_dir": str(seal_dir),
                "manifest_sha256": sha256_file(seal_dir / "manifest.json"),
                "members": member_names,
                "ensembles": list(ensembles),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def load_and_verify_seal(seal_dir: str | Path) -> Tuple[Dict, Dict]:
    seal_dir = assert_canonical_directory(seal_dir, "frozen seal")
    expected_root_entries = {"manifest.json", "seal.json", "members"}
    if {path.name for path in seal_dir.iterdir()} != expected_root_entries:
        raise RuntimeError("Frozen seal contains missing or unexpected root artifacts")
    control_before = {
        name: file_fingerprint(seal_dir / name) for name in ("manifest.json", "seal.json")
    }
    seal = _read_json(seal_dir / "seal.json")
    manifest = _read_json(seal_dir / "manifest.json")
    control_after = {
        name: file_fingerprint(seal_dir / name) for name in ("manifest.json", "seal.json")
    }
    assert_fingerprints_equal(control_after, control_before, "frozen seal control files")
    if (
        not isinstance(seal, Mapping)
        or set(seal) != {"schema_version", "created_utc", "manifest_sha256"}
        or seal.get("schema_version") != SCHEMA_VERSION
        or not isinstance(seal.get("created_utc"), str)
        or not isinstance(seal.get("manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", seal["manifest_sha256"])
    ):
        raise RuntimeError("Unsupported frozen seal schema")
    validate_frozen_manifest(manifest)
    actual_manifest_digest = control_after["manifest.json"]["sha256"]
    if seal.get("manifest_sha256") != actual_manifest_digest:
        raise RuntimeError("Frozen manifest digest does not match seal")
    members_dir = assert_canonical_directory(seal_dir / "members", "frozen members")
    if {path.name for path in members_dir.iterdir()} != set(manifest["members"]):
        raise RuntimeError("Frozen member directories differ from the manifest")
    for name, member in manifest["members"].items():
        assert_canonical_directory(members_dir / name, f"frozen member {name}")
        actual = fingerprint_named_files(seal_dir / "members" / name, RUN_ARTIFACTS)
        assert_fingerprints_equal(actual, member["artifacts"], f"sealed member {name}")
    current_code = _code_fingerprints(
        member["kind"] for member in manifest["members"].values()
    )
    assert_fingerprints_equal(current_code, manifest.get("code", {}), "evaluation code")
    current_runtime = runtime_fingerprint(
        member["kind"] for member in manifest["members"].values()
    )
    if current_runtime != manifest.get("runtime"):
        raise RuntimeError(
            f"Evaluation runtime changed: actual={current_runtime} expected={manifest.get('runtime')}"
        )
    return manifest, seal


@torch.no_grad()
def evaluate_model(model, kind: str, loader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    labels_all, logits_all, indices_all = [], [], []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        if kind == "lenspinn-repaired":
            from .lenspinn import lenspinn_distortion

            distortion = lenspinn_distortion(images)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            if kind == "lenspinn-repaired":
                logits = model(images, distortion)
            else:
                logits = model(images)
        labels_all.append(labels.numpy())
        indices_all.append(indices.numpy())
        logits_all.append(logits.float().cpu().numpy())
    labels_np = np.concatenate(labels_all).astype(np.int64, copy=False)
    indices_np = np.concatenate(indices_all).astype(np.int64, copy=False)
    logits_np = np.concatenate(logits_all).astype(np.float64, copy=False)
    if not np.isfinite(logits_np).all():
        raise RuntimeError(f"{kind} evaluation produced non-finite logits")
    return {
        "indices": indices_np,
        "labels": labels_np,
        "logits": logits_np,
        "probabilities": softmax_numpy(logits_np),
    }


def _make_eval_loader(cache_dir: Path, indices: np.ndarray | None, args: argparse.Namespace):
    return make_loader(
        CachedNPYDataset(cache_dir, indices),
        int(args.batch_size),
        False,
        int(args.workers),
        int(args.loader_seed),
    )


def _load_sealed_model(
    seal_dir: Path, name: str, member: Mapping, class_count: int, device: torch.device
):
    member_dir = seal_dir / "members" / name
    config = _read_json(member_dir / "config.json")
    model, _, resolved_config, migrations = load_model_strict(
        member["kind"], config, member_dir / "best.pt", class_count
    )
    if resolved_config != member.get("resolved_config") or migrations != member.get(
        "config_migrations"
    ):
        raise RuntimeError(f"Sealed architecture resolution changed for member {name}")
    return model.to(device, memory_format=torch.channels_last)


def _materialize_ensembles(
    member_probabilities: Mapping[str, np.ndarray], ensembles: Mapping[str, Sequence[str]]
) -> Dict[str, np.ndarray]:
    result = dict(member_probabilities)
    for name, children in ensembles.items():
        result[name] = uniform_probability_ensemble([result[child] for child in children])
    return result


def lexical_paths_overlap(first: str | Path, second: str | Path) -> bool:
    first_text = os.path.abspath(os.path.expanduser(str(first)))
    second_text = os.path.abspath(os.path.expanduser(str(second)))
    common = os.path.commonpath((first_text, second_text))
    return common == first_text or common == second_text


def validate_resumable_output(output_dir: Path, result_names: Iterable[str]) -> None:
    """Reject ambiguous or redirecting artifacts from a partial test attempt."""

    allowed_root = {
        "TEST_ACCESS_MARKER.json",
        "validation_replay.json",
        "result.json",
        "predictions",
    }
    unexpected = {path.name for path in output_dir.iterdir()} - allowed_root
    if unexpected:
        raise RuntimeError(f"Locked-test output contains unexpected artifacts: {sorted(unexpected)}")
    for name in ("TEST_ACCESS_MARKER.json", "validation_replay.json", "result.json"):
        path = output_dir / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError(f"Locked-test output contains an unsafe artifact: {path}")
    predictions = output_dir / "predictions"
    if predictions.is_symlink():
        raise RuntimeError("Partial prediction directory is a symlink")
    if predictions.exists():
        assert_canonical_directory(predictions, "partial prediction directory")
        allowed_predictions = {f"{name}.npz" for name in result_names}
        unexpected_predictions = {
            path.name for path in predictions.iterdir()
        } - allowed_predictions
        if unexpected_predictions:
            raise RuntimeError(
                "Partial prediction directory contains unexpected artifacts: "
                f"{sorted(unexpected_predictions)}"
            )
        for path in predictions.iterdir():
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"Partial prediction artifact is unsafe: {path}")


def run_test(args: argparse.Namespace) -> None:
    seal_dir = Path(args.seal_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not seal_dir.is_absolute() or not output_dir.is_absolute():
        raise RuntimeError("--seal-dir and --output-dir must be absolute")
    test_cache_text = os.path.abspath(os.path.expanduser(args.test_cache))
    early_paths = [("seal", seal_dir), ("output", output_dir)]
    if args.development_cache:
        early_paths.append(("development", Path(args.development_cache).expanduser()))
    for name, path in early_paths:
        if lexical_paths_overlap(path, test_cache_text):
            raise RuntimeError(f"Unsafe overlapping {name}/test paths")
    seal_dir = assert_canonical_directory(seal_dir, "frozen seal")
    manifest, seal = load_and_verify_seal(seal_dir)
    expected_confirmation = f"{manifest['protocol']['id']}:{seal['manifest_sha256']}"
    if args.confirm != expected_confirmation:
        raise RuntimeError(
            "Locked-test confirmation mismatch; expected "
            f"--confirm '{expected_confirmation}'"
        )
    frozen_inference = manifest["protocol"]["inference"]
    actual_inference = {
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "loader_seed": int(args.loader_seed),
        "autocast": "cuda-bfloat16",
    }
    if actual_inference != frozen_inference:
        raise RuntimeError(
            f"Inference settings changed: actual={actual_inference} "
            f"expected={frozen_inference}"
        )
    class_names = manifest["development_cache"]["classes"]
    if not torch.cuda.is_available():
        raise RuntimeError("Locked CUDA evaluation requires a GPU job")
    device = torch.device("cuda")
    development_cache = Path(
        args.development_cache or manifest["development_cache"]["path"]
    ).expanduser()
    if not development_cache.is_absolute():
        raise RuntimeError("Development cache path must be absolute")
    expected_test_path = manifest["expected_official_test"]["canonical_job_path"]
    if test_cache_text != expected_test_path:
        raise RuntimeError("Test-cache argument differs from the prospectively frozen path")
    for first_name, first_path, second_name, second_path in (
        ("output", output_dir, "seal", seal_dir),
        ("output", output_dir, "development", development_cache),
        ("output", output_dir, "test", test_cache_text),
        ("seal", seal_dir, "test", test_cache_text),
        ("development", development_cache, "test", test_cache_text),
    ):
        if lexical_paths_overlap(first_path, second_path):
            raise RuntimeError(f"Unsafe overlapping {first_name}/{second_name} paths")
    # Only after lexical exclusion of the official-test tree may we lstat or
    # resolve the development/output paths.
    development_cache = assert_canonical_directory(
        development_cache, "development cache"
    )
    output_dir = assert_no_symlink_components(output_dir, "locked-test output")
    marker_path = output_dir / "TEST_ACCESS_MARKER.json"
    expected_marker_core = {
        "schema_version": SCHEMA_VERSION,
        "frozen_manifest_sha256": seal["manifest_sha256"],
        "test_cache_argument": test_cache_text,
        "meaning": "Official test access begins after this durable marker.",
    }
    existing_marker_fingerprint = None
    if output_dir.exists():
        output_dir = assert_canonical_directory(output_dir, "locked-test output")
        if (output_dir / "artifact_seal.json").exists():
            raise RuntimeError(f"A sealed test result already exists: {output_dir}")
        existing_entries = list(output_dir.iterdir())
        if existing_entries:
            if not args.resume_after_marker or not marker_path.is_file():
                raise RuntimeError(
                    "Refusing to mutate a non-empty locked-test output without an authorized marker resume"
                )
            existing_marker = _read_json(marker_path)
            if set(existing_marker) != set(expected_marker_core) | {"created_utc"}:
                raise RuntimeError("Existing test-access marker schema is invalid")
            if any(existing_marker.get(key) != value for key, value in expected_marker_core.items()):
                raise RuntimeError("Existing test-access marker belongs to a different attempt")
            if not isinstance(existing_marker.get("created_utc"), str):
                raise RuntimeError("Existing test-access marker timestamp is invalid")
            existing_marker_fingerprint = file_fingerprint(marker_path)
            validate_resumable_output(
                output_dir,
                list(manifest["members"]) + list(manifest.get("ensembles", {})),
            )
        elif args.resume_after_marker:
            raise RuntimeError("--resume-after-marker requires an existing marker")
    else:
        assert_no_symlink_components(output_dir.parent, "locked-test output parent")
        output_dir.mkdir(parents=True)
        output_dir = assert_canonical_directory(output_dir, "locked-test output")
        fsync_directory(output_dir.parent)
    dev_before = fingerprint_named_files(development_cache, CACHE_ARTIFACTS)
    assert_fingerprints_equal(
        dev_before,
        manifest["development_cache"]["artifacts"],
        "development cache before validation replay",
    )
    development = validate_cache_structure(development_cache, class_names)

    validation_replay: Dict[str, Dict] = {}
    replay_probabilities: Dict[str, np.ndarray] = {}
    reference_indices = reference_labels = None
    replay_tolerances = _validate_replay_tolerances(
        manifest["protocol"]["validation_replay_probability_atol"],
        manifest["protocol"]["validation_replay_probability_mean_atol"],
        manifest["protocol"]["validation_replay_probability_p99_atol"],
        manifest["protocol"]["validation_replay_metric_atol"],
        error_type=RuntimeError,
    )
    for name, member in manifest["members"].items():
        member_dir = seal_dir / "members" / name
        with np.load(member_dir / "split_indices.npz", allow_pickle=False) as split:
            _, val_indices = validate_split_indices(
                development["labels"], split["train"], split["val"]
            )
        expected = load_validation_predictions(
            member_dir / "best_validation_predictions.npz",
            val_indices,
            development["labels"],
            class_names,
        )
        model = _load_sealed_model(seal_dir, name, member, len(class_names), device)
        actual = evaluate_model(
            model,
            member["kind"],
            _make_eval_loader(development_cache, val_indices, args),
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not np.array_equal(actual["indices"], expected["indices"]):
            raise RuntimeError(f"Validation replay index mismatch for {name}")
        if not np.array_equal(actual["labels"], expected["labels"]):
            raise RuntimeError(f"Validation replay label mismatch for {name}")
        diagnostics = probability_replay_diagnostics(
            actual["probabilities"], expected["probabilities"]
        )
        if (
            diagnostics["max_probability_absolute_difference"]
            > replay_tolerances["probability_atol"]
            or diagnostics["mean_probability_absolute_difference"]
            > replay_tolerances["probability_mean_atol"]
            or diagnostics["p99_probability_absolute_difference"]
            > replay_tolerances["probability_p99_atol"]
            or not diagnostics["predicted_classes_exact"]
        ):
            raise RuntimeError(
                f"Validation replay failed for {name}: "
                f"max_probability_abs="
                f"{diagnostics['max_probability_absolute_difference']:g}, "
                f"mean_probability_abs="
                f"{diagnostics['mean_probability_absolute_difference']:g}, "
                f"p99_probability_abs="
                f"{diagnostics['p99_probability_absolute_difference']:g}, "
                f"predictions_equal={diagnostics['predicted_classes_exact']}"
            )
        replay_probabilities[name] = actual["probabilities"]
        validation_replay[name] = {
            **diagnostics,
            "metrics": metrics_from_probabilities(
                actual["labels"], actual["probabilities"], class_names
            ),
        }
        if reference_indices is None:
            reference_indices, reference_labels = actual["indices"], actual["labels"]
        elif not np.array_equal(reference_indices, actual["indices"]) or not np.array_equal(
            reference_labels, actual["labels"]
        ):
            raise RuntimeError("Validation replay members have different sample manifests")

    replay_all_probabilities = _materialize_ensembles(
        replay_probabilities, manifest.get("ensembles", {})
    )
    replay_plan_metrics: Dict[str, Dict] = {}
    replay_metric_drift: Dict[str, Dict[str, float]] = {}
    for name, probabilities in replay_all_probabilities.items():
        replay_metrics = metrics_from_probabilities(reference_labels, probabilities, class_names)
        replay_plan_metrics[name] = replay_metrics
        maximum_metric_difference = _metric_close(
            replay_metrics,
            manifest["validation_plan_metrics"][name],
            f"frozen validation plan for {name}",
            tolerance=replay_tolerances["metric_atol"],
        )
        replay_metric_drift[name] = {
            "max_metric_absolute_difference": maximum_metric_difference
        }
        if name in validation_replay:
            validation_replay[name][
                "max_metric_absolute_difference"
            ] = maximum_metric_difference

    dev_after = fingerprint_named_files(development_cache, CACHE_ARTIFACTS)
    assert_fingerprints_equal(dev_after, dev_before, "development cache during validation replay")
    atomic_json(
        output_dir / "validation_replay.json",
        {
            "schema_version": SCHEMA_VERSION,
            "frozen_manifest_sha256": seal["manifest_sha256"],
            "completed_utc": utc_now(),
            "device": str(device),
            "inference": frozen_inference,
            "gates": {
                "max_probability_absolute_difference": replay_tolerances[
                    "probability_atol"
                ],
                "mean_probability_absolute_difference": replay_tolerances[
                    "probability_mean_atol"
                ],
                "p99_probability_absolute_difference": replay_tolerances[
                    "probability_p99_atol"
                ],
                "max_metric_absolute_difference": replay_tolerances["metric_atol"],
                "predicted_classes_exact": True,
            },
            "members": validation_replay,
            "all_result_metric_drift": replay_metric_drift,
            "all_results": replay_plan_metrics,
        },
    )

    # This is the protocol boundary.  Do not access, stat, resolve, hash, or
    # validate the test-cache argument above this marker creation.
    marker_value = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "frozen_manifest_sha256": seal["manifest_sha256"],
        "test_cache_argument": test_cache_text,
        "meaning": "Official test access begins after this durable marker.",
    }
    if marker_path.exists():
        if existing_marker_fingerprint is None:
            raise RuntimeError("Unexpected marker appeared during validation replay")
        if file_fingerprint(marker_path) != existing_marker_fingerprint:
            raise RuntimeError("Existing test-access marker changed during validation replay")
    else:
        atomic_json(marker_path, marker_value, exclusive=True)
        existing_marker_fingerprint = file_fingerprint(marker_path)

    test_cache = assert_canonical_directory(test_cache_text, "official test cache")
    test_before = fingerprint_named_files(test_cache, CACHE_ARTIFACTS)
    test = validate_cache_structure(test_cache, class_names)
    expected_test = manifest["expected_official_test"]
    for artifact, expected_sha256 in expected_test["artifact_sha256"].items():
        if test_before[artifact]["sha256"] != expected_sha256:
            raise RuntimeError(
                f"Official test {artifact} does not match the prospectively frozen hash"
            )
    if int(test["metadata"]["samples"]) != int(expected_test["samples"]):
        raise RuntimeError("Official test sample count does not match the frozen protocol")
    if int(test["metadata"]["image_size"]) != int(expected_test["image_size"]):
        raise RuntimeError("Official test image size does not match the frozen protocol")
    if test["classes"] != expected_test["classes"]:
        raise RuntimeError("Official test classes do not match the frozen protocol")
    intersection = development["digests"].intersection(test["digests"])
    if intersection:
        raise RuntimeError(
            f"Development/test model-visible collision after marker: {len(intersection)} samples"
        )

    member_test_probabilities: Dict[str, np.ndarray] = {}
    member_test_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    test_indices_ref = test_labels_ref = None
    for name, member in manifest["members"].items():
        model = _load_sealed_model(seal_dir, name, member, len(class_names), device)
        actual = evaluate_model(
            model,
            member["kind"],
            _make_eval_loader(test_cache, None, args),
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if test_indices_ref is None:
            test_indices_ref, test_labels_ref = actual["indices"], actual["labels"]
            expected_test_indices = np.arange(len(test["labels"]), dtype=np.int64)
            if not np.array_equal(test_indices_ref, expected_test_indices):
                raise RuntimeError("Official test loader did not cover the cache exactly once in order")
        elif not np.array_equal(test_indices_ref, actual["indices"]) or not np.array_equal(
            test_labels_ref, actual["labels"]
        ):
            raise RuntimeError("Test members produced different sample manifests")
        if not np.array_equal(actual["labels"], test["labels"][actual["indices"]]):
            raise RuntimeError(f"Test labels disagree with cache for member {name}")
        member_test_probabilities[name] = actual["probabilities"]
        member_test_outputs[name] = actual

    all_probabilities = _materialize_ensembles(
        member_test_probabilities, manifest.get("ensembles", {})
    )
    metrics = {
        name: metrics_from_probabilities(test_labels_ref, probabilities, class_names)
        for name, probabilities in all_probabilities.items()
    }
    comparisons = {}
    for comparison_index, comparison in enumerate(manifest.get("comparisons", [])):
        a, b = comparison["a"], comparison["b"]
        key = f"comparison-{comparison_index:02d}:{a}__vs__{b}"
        bootstrap = stratified_paired_bootstrap_accuracy(
            test_labels_ref,
            all_probabilities[a],
            all_probabilities[b],
            samples=int(manifest["protocol"]["bootstrap_samples"]),
            seed=int(manifest["protocol"]["analysis_seed"]),
        )
        comparisons[key] = {
            "a": a,
            "b": b,
            "stratified_paired_bootstrap_accuracy": bootstrap,
            "mcnemar_exact": mcnemar_exact(
                test_labels_ref, all_probabilities[a], all_probabilities[b]
            ),
            "minimum_acceptable_accuracy_difference": comparison.get(
                "minimum_acceptable_accuracy_difference"
            ),
        }
        minimum_difference = comparison.get("minimum_acceptable_accuracy_difference")
        if minimum_difference is not None:
            comparisons[key]["minimum_difference_criterion_passed"] = bool(
                bootstrap["ci95_low"] > float(minimum_difference)
            )
    ordered_p_values = sorted(
        (
            value["mcnemar_exact"]["two_sided_exact_p"],
            key,
        )
        for key, value in comparisons.items()
    )
    running_adjusted = 0.0
    total_comparisons = len(ordered_p_values)
    for rank, (p_value, key) in enumerate(ordered_p_values):
        running_adjusted = max(
            running_adjusted,
            min(1.0, (total_comparisons - rank) * float(p_value)),
        )
        comparisons[key]["mcnemar_holm_adjusted_p"] = running_adjusted

    test_after = fingerprint_named_files(test_cache, CACHE_ARTIFACTS)
    assert_fingerprints_equal(test_after, test_before, "official test cache during evaluation")
    if file_fingerprint(marker_path) != existing_marker_fingerprint:
        raise RuntimeError("Test-access marker changed during official evaluation")
    prediction_directory = output_dir / "predictions"
    if prediction_directory.exists():
        assert_canonical_directory(prediction_directory, "prediction output")
    else:
        prediction_directory.mkdir()
        assert_canonical_directory(prediction_directory, "prediction output")
        fsync_directory(output_dir)
    prediction_files = []
    for name, probabilities in all_probabilities.items():
        relative = f"predictions/{name}.npz"
        logits = (
            member_test_outputs[name]["logits"]
            if name in member_test_outputs
            else np.log(np.clip(probabilities, 1e-300, 1.0))
        )
        atomic_npz(
            output_dir / relative,
            indices=test_indices_ref,
            relative_paths=test["relative_paths"][test_indices_ref],
            visible_sha256=test["digest_rows"][test_indices_ref],
            labels=test_labels_ref,
            logits=logits.astype(np.float64),
            probabilities=probabilities.astype(np.float64),
            predictions=probabilities.argmax(axis=1).astype(np.int64),
        )
        prediction_files.append(relative)
    result = {
        "schema_version": SCHEMA_VERSION,
        "completed_utc": utc_now(),
        "frozen_manifest_sha256": seal["manifest_sha256"],
        "test_access_marker": marker_path.name,
        "device": str(device),
        "inference": frozen_inference,
        "classes": class_names,
        "test_samples": int(len(test_labels_ref)),
        "development_test_visible_digest_intersection": 0,
        "test_cache_artifacts": test_before,
        "ensemble_rule": manifest["protocol"]["ensemble_rule"],
        "metrics": metrics,
        "comparisons": comparisons,
        "prediction_artifacts": prediction_files,
    }
    atomic_json(output_dir / "result.json", result)
    artifact_names = [
        "validation_replay.json",
        marker_path.name,
        "result.json",
        *prediction_files,
    ]
    artifact_fingerprints = {
        name: file_fingerprint(output_dir / name) for name in artifact_names
    }
    atomic_json(
        output_dir / "artifact_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "frozen_manifest_sha256": seal["manifest_sha256"],
            "artifacts": artifact_fingerprints,
        },
        exclusive=True,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Validate and seal development-only runs (this stage has no test argument)",
    )
    freeze_parser.add_argument("--development-cache", required=True)
    freeze_parser.add_argument("--seal-dir", required=True)
    freeze_parser.add_argument("--protocol-id", required=True)
    freeze_parser.add_argument("--expected-test-manifest-sha256", required=True)
    freeze_parser.add_argument("--expected-test-images-sha256", required=True)
    freeze_parser.add_argument("--expected-test-labels-sha256", required=True)
    freeze_parser.add_argument("--expected-test-metadata-sha256", required=True)
    freeze_parser.add_argument("--expected-development-manifest-sha256", required=True)
    freeze_parser.add_argument("--expected-test-samples", type=int, default=15_000)
    freeze_parser.add_argument("--expected-test-cache-path", required=True)
    freeze_parser.add_argument(
        "--member",
        action="append",
        required=True,
        metavar="NAME=KIND=/ABS/RUN",
    )
    freeze_parser.add_argument(
        "--ensemble",
        action="append",
        default=[],
        metavar="NAME=MEMBER_A,MEMBER_B",
    )
    freeze_parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="RESULT_A,RESULT_B[,MIN_ACCEPTABLE_A_MINUS_B]",
        help="Repeat to predeclare each primary/secondary paired comparison",
    )
    freeze_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    freeze_parser.add_argument("--analysis-seed", type=int, default=20260711)
    freeze_parser.add_argument(
        "--replay-atol",
        type=float,
        default=MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
        help="Maximum permitted absolute probability drift during validation replay",
    )
    freeze_parser.add_argument(
        "--replay-mean-atol",
        type=float,
        default=MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
        help="Maximum permitted mean absolute probability drift during validation replay",
    )
    freeze_parser.add_argument(
        "--replay-p99-atol",
        type=float,
        default=MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
        help="Maximum permitted p99 absolute probability drift during validation replay",
    )
    freeze_parser.add_argument(
        "--replay-metric-atol",
        type=float,
        default=MAX_VALIDATION_REPLAY_METRIC_ATOL,
        help="Maximum permitted absolute validation-metric drift",
    )
    freeze_parser.add_argument("--settle-seconds", type=float, default=30.0)
    freeze_parser.add_argument("--inference-batch-size", type=int, default=128)
    freeze_parser.add_argument("--inference-workers", type=int, default=8)
    freeze_parser.add_argument("--inference-loader-seed", type=int, default=42)
    freeze_parser.set_defaults(handler=freeze)

    test_parser = subparsers.add_parser("run-test", help="Replay validation, mark, then test")
    test_parser.add_argument("--seal-dir", required=True)
    test_parser.add_argument("--development-cache")
    test_parser.add_argument("--test-cache", required=True)
    test_parser.add_argument("--output-dir", required=True)
    test_parser.add_argument(
        "--confirm",
        required=True,
        help="Exact PROTOCOL_ID:FROZEN_MANIFEST_SHA256 confirmation token",
    )
    test_parser.add_argument("--batch-size", type=int, default=128)
    test_parser.add_argument("--workers", type=int, default=8)
    test_parser.add_argument("--loader-seed", type=int, default=42)
    test_parser.add_argument("--resume-after-marker", action="store_true")
    test_parser.set_defaults(handler=run_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "bootstrap_samples", 1) <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if getattr(args, "settle_seconds", 0.0) < 0:
        raise ValueError("--settle-seconds cannot be negative")
    if hasattr(args, "settle_seconds") and not math.isfinite(args.settle_seconds):
        raise ValueError("--settle-seconds must be finite")
    if getattr(args, "analysis_seed", 0) < 0:
        raise ValueError("--analysis-seed cannot be negative")
    if hasattr(args, "replay_atol"):
        _validate_replay_tolerances(
            args.replay_atol,
            args.replay_mean_atol,
            args.replay_p99_atol,
            args.replay_metric_atol,
        )
    if getattr(args, "inference_batch_size", 1) <= 0:
        raise ValueError("--inference-batch-size must be positive")
    if getattr(args, "inference_workers", 0) < 0:
        raise ValueError("--inference-workers cannot be negative")
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("--batch-size must be positive")
    if getattr(args, "workers", 0) < 0:
        raise ValueError("--workers cannot be negative")
    args.handler(args)


if __name__ == "__main__":
    main()
