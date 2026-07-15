"""Leakage-safe dataset loading, resize caching, splitting, and loaders."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config


IMAGE_KEYS = ("image", "img", "x", "data", "array", "arr", "lens", "sample")
EXPECTED_CLASSES = ("axion", "cdm", "no_sub")
MODEL_IV_AUDIT_SEED = 20_260_715
MODEL_IV_AUDIT_TRAIN_CAP = 4_000
MODEL_IV_AUDIT_VALIDATION_CAP = 2_000
MODEL_IV_AUDIT_MIN_PER_CLASS = 500
MODEL_IV_AUDIT_BOOTSTRAPS = 1_000
MODEL_IV_AUDIT_PERMUTATIONS = 999

PASS_SIGNAL_DETECTED = "PASS_SIGNAL_DETECTED"
PREPROCESSING_SIGNAL_LOSS = "PREPROCESSING_SIGNAL_LOSS"
INCONCLUSIVE_NO_SIGNAL_DETECTED = "INCONCLUSIVE_NO_SIGNAL_DETECTED"
INTEGRITY_FAILED = "INTEGRITY_FAILED"

_AUDIT_ANNULI = 8
_AUDIT_POOL = 8


def extract_image_array(value) -> np.ndarray:
    """Extract only the two-dimensional image and discard scalar metadata."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        if value.ndim == 0:
            return extract_image_array(value.item())
        for item in value.reshape(-1):
            candidate = extract_image_array(item)
            if np.asarray(candidate).ndim >= 2:
                return np.asarray(candidate)
        raise ValueError("Object array contains no image")
    if isinstance(value, dict):
        for key in IMAGE_KEYS:
            if key in value:
                candidate = extract_image_array(value[key])
                if np.asarray(candidate).ndim >= 2:
                    return np.asarray(candidate)
        for item in value.values():
            candidate = extract_image_array(item)
            if np.asarray(candidate).ndim >= 2:
                return np.asarray(candidate)
        raise ValueError("Dictionary contains no image")
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = extract_image_array(item)
            if np.asarray(candidate).ndim >= 2:
                return np.asarray(candidate)
        raise ValueError("Sequence contains no image")
    return np.asarray(value)


def load_model_visible_image(path: str | Path) -> Tuple[np.ndarray, str]:
    """Apply model-visible preprocessing and return a content digest."""

    raw = np.load(path, allow_pickle=True)
    image = np.asarray(extract_image_array(raw), dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image in {path}, got {image.shape}")
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(image.max())
    if maximum > 1.0:
        image = image / maximum
    image = np.ascontiguousarray(image.astype("<f4", copy=False))
    return image, hashlib.sha256(image.tobytes()).hexdigest()


def list_samples(root: str | Path) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if classes != list(EXPECTED_CLASSES):
        raise RuntimeError(
            "Class directories must be exactly "
            f"{list(EXPECTED_CLASSES)}; found {classes} under {root}"
        )
    samples: List[Tuple[str, int, str]] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        class_samples = sorted(class_dir.glob("*.npy"))
        if not class_samples:
            raise RuntimeError(f"No .npy samples under {class_dir}")
        for path in class_samples:
            samples.append((str(path), label, str(path.relative_to(root))))
    return samples, classes


def _load_path(
    record: Tuple[str, int, str]
) -> Tuple[np.ndarray, int, str, str]:
    path, label, relative = record
    image, digest = load_model_visible_image(path)
    return image, label, relative, digest


def prepare_cache(
    source_root: str | Path,
    cache_dir: str | Path,
    image_size: int,
    device: torch.device,
    io_workers: int = 8,
    chunk_size: int = 384,
    storage_dtype=np.float16,
) -> Dict:
    """Create or reuse an atomic resize cache at the requested precision."""

    source_root = Path(source_root).resolve()
    cache_dir = Path(cache_dir)
    cache_dtype = np.dtype(storage_dtype)
    if cache_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(f"Unsupported cache dtype: {cache_dtype}")
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        required = (
            cache_dir / "images.npy",
            cache_dir / "labels.npy",
            cache_dir / "manifest.csv",
        )
        if (
            metadata.get("complete")
            and metadata.get("image_size") == image_size
            and metadata.get("dtype") == cache_dtype.name
            and Path(metadata.get("source_root", "")) == source_root
            and metadata.get("classes") == list(EXPECTED_CLASSES)
            and all(path.exists() for path in required)
        ):
            print(
                f"CACHE_READY {cache_dir} samples={metadata['samples']}",
                flush=True,
            )
            return metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    samples, classes = list_samples(source_root)
    build_tag = f"building-{os.getpid()}"
    image_tmp = cache_dir / f"images-{build_tag}.npy"
    labels_tmp = cache_dir / f"labels-{build_tag}.npy"
    manifest_tmp = cache_dir / f"manifest-{build_tag}.csv"
    images_memmap = np.lib.format.open_memmap(
        image_tmp,
        mode="w+",
        dtype=cache_dtype,
        shape=(len(samples), image_size, image_size),
    )
    labels = np.empty(len(samples), dtype=np.int64)

    with manifest_tmp.open("w", newline="") as manifest_handle:
        writer = csv.writer(manifest_handle)
        writer.writerow(
            ("index", "relative_path", "class", "label", "sha256_visible")
        )
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            for start in range(0, len(samples), chunk_size):
                stop = min(start + chunk_size, len(samples))
                loaded = list(pool.map(_load_path, samples[start:stop]))
                batch = np.stack([item[0] for item in loaded], axis=0)
                tensor = torch.from_numpy(batch).unsqueeze(1).to(
                    device=device, dtype=torch.float32
                )
                resized = F.interpolate(
                    tensor,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                output_dtype = (
                    torch.float32
                    if cache_dtype == np.dtype(np.float32)
                    else torch.float16
                )
                images_memmap[start:stop] = (
                    resized[:, 0].to(dtype=output_dtype).cpu().numpy()
                )
                for offset, (_, label, relative, digest) in enumerate(loaded):
                    index = start + offset
                    labels[index] = label
                    writer.writerow(
                        (index, relative, classes[label], label, digest)
                    )
                print(f"CACHE_PROGRESS {stop}/{len(samples)}", flush=True)

    images_memmap.flush()
    np.save(labels_tmp, labels)
    os.replace(image_tmp, cache_dir / "images.npy")
    os.replace(labels_tmp, cache_dir / "labels.npy")
    os.replace(manifest_tmp, cache_dir / "manifest.csv")
    metadata = {
        "complete": True,
        "source_root": str(source_root),
        "image_size": image_size,
        "samples": len(samples),
        "classes": classes,
        "class_counts": {
            classes[index]: int((labels == index).sum())
            for index in range(len(classes))
        },
        "normalization": "nonfinite cleanup; divide by max only when max > 1",
        "interpolation": "bilinear align_corners=False antialias=True",
        "dtype": cache_dtype.name,
    }
    metadata_tmp = cache_dir / f"metadata-{build_tag}.json"
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(metadata_tmp, metadata_path)
    print(f"CACHE_COMPLETE {cache_dir}", flush=True)
    return metadata


def _visible_digests(cache_dir: str | Path) -> set[str]:
    """Return model-visible content hashes recorded by a completed cache."""

    manifest_path = Path(cache_dir) / "manifest.csv"
    with manifest_path.open(newline="") as manifest_handle:
        reader = csv.DictReader(manifest_handle)
        if reader.fieldnames is None or "sha256_visible" not in reader.fieldnames:
            raise RuntimeError(
                f"Cache manifest has no sha256_visible column: {manifest_path}"
            )
        return {row["sha256_visible"] for row in reader}


def _require_disjoint_visible_content(
    development_cache_dir: str | Path,
    validation_cache_dir: str | Path,
) -> None:
    """Reject model-visible samples shared by supplied train/validation roots."""

    overlap = _visible_digests(development_cache_dir).intersection(
        _visible_digests(validation_cache_dir)
    )
    if overlap:
        raise RuntimeError(
            "Supplied development and validation roots share "
            f"{len(overlap)} model-visible image digest(s); refusing a leaky run"
        )


@dataclass(slots=True)
class ModelIVAuditResult:
    """Outcome of the CPU-only, development-validation data gate."""

    status: str
    report_path: Path
    integrity_failures: List[str]
    probes: Dict[str, Dict[str, Any]]


def _schema_signature(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape},dtype={value.dtype})"
    return type(value).__name__


def _inspect_audit_record(
    record: Tuple[str, int, str],
) -> Dict[str, Any]:
    path, label, relative = record
    try:
        value = np.load(path, allow_pickle=True)
        schema = _schema_signature(value)
        image = np.asarray(extract_image_array(value), dtype=np.float32).squeeze()
        if image.shape != (64, 64):
            return {
                "label": label,
                "relative": relative,
                "schema": schema,
                "failure": f"wrong extracted shape {image.shape}",
            }
        if not np.isfinite(image).all():
            return {
                "label": label,
                "relative": relative,
                "schema": schema,
                "failure": "nonfinite raw pixels",
            }
        minimum = float(image.min())
        maximum = float(image.max())
        if maximum == minimum:
            return {
                "label": label,
                "relative": relative,
                "schema": schema,
                "failure": "constant raw image",
            }
        visible = image.copy()
        if maximum > 1.0:
            visible /= maximum
        visible = np.ascontiguousarray(visible.astype("<f4", copy=False))
        return {
            "label": label,
            "relative": relative,
            "schema": schema,
            "failure": "",
            "digest": hashlib.sha256(visible.tobytes()).hexdigest(),
            "minimum": minimum,
            "maximum": maximum,
            "negative_fraction": float(np.mean(image < 0.0)),
        }
    except Exception as error:  # The report retains the path and short reason.
        return {
            "label": label,
            "relative": relative,
            "schema": "unreadable",
            "failure": f"{type(error).__name__}: {error}",
        }


def _balanced_audit_indices(
    labels: np.ndarray, per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: List[np.ndarray] = []
    for label in range(len(EXPECTED_CLASSES)):
        choices = np.flatnonzero(labels == label)
        rng.shuffle(choices)
        parts.append(choices[: min(per_class, len(choices))])
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices


def _scan_audit_root(
    root: Path,
    split_name: str,
    io_workers: int,
) -> Tuple[
    List[Tuple[str, int, str]],
    np.ndarray,
    List[Dict[str, Any]],
    List[str],
]:
    samples, _ = list_samples(root)
    labels = np.asarray([record[1] for record in samples], dtype=np.int64)
    failures: List[str] = []
    for label, class_name in enumerate(EXPECTED_CLASSES):
        count = int(np.sum(labels == label))
        if count < MODEL_IV_AUDIT_MIN_PER_CLASS:
            failures.append(
                f"{split_name}/{class_name} has {count} samples; "
                f"at least {MODEL_IV_AUDIT_MIN_PER_CLASS} are required"
            )

    rows: List[Dict[str, Any]] = []
    failure_counts: Dict[str, int] = {}
    failure_examples: Dict[str, str] = {}
    workers = max(1, min(io_workers, _AUDIT_POOL))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(_inspect_audit_record, samples), 1):
            rows.append(row)
            reason = str(row["failure"])
            if reason:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
                failure_examples.setdefault(reason, str(row["relative"]))
            if index % 5_000 == 0 or index == len(samples):
                print(
                    f"MODEL_IV_AUDIT_SCAN {split_name} {index}/{len(samples)}",
                    flush=True,
                )
    for reason, count in sorted(failure_counts.items()):
        failures.append(
            f"{split_name}: {count} file(s) failed {reason}; "
            f"example={failure_examples[reason]}"
        )
    return samples, labels, rows, failures


def _digest_integrity(
    development_rows: List[Dict[str, Any]],
    validation_rows: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int]]:
    failures: List[str] = []

    def digest_map(rows: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        output: Dict[str, List[int]] = {}
        for row in rows:
            digest = row.get("digest")
            if digest:
                output.setdefault(str(digest), []).append(int(row["label"]))
        return output

    development = digest_map(development_rows)
    validation = digest_map(validation_rows)
    cross_label_development = sum(
        len(set(labels)) > 1 for labels in development.values()
    )
    cross_label_validation = sum(
        len(set(labels)) > 1 for labels in validation.values()
    )
    overlap = len(set(development).intersection(validation))
    if cross_label_development:
        failures.append(
            "development contains "
            f"{cross_label_development} model-visible digest(s) across labels"
        )
    if cross_label_validation:
        failures.append(
            "validation contains "
            f"{cross_label_validation} model-visible digest(s) across labels"
        )
    if overlap:
        failures.append(
            f"development and validation share {overlap} model-visible digest(s)"
        )
    same_label_duplicates = 0
    for mapping in (development, validation):
        for labels in mapping.values():
            if len(set(labels)) == 1 and len(labels) > 1:
                same_label_duplicates += len(labels) - 1
    return failures, {
        "cross_label_development": cross_label_development,
        "cross_label_validation": cross_label_validation,
        "development_validation_overlap": overlap,
        "same_label_duplicate_copies": same_label_duplicates,
    }


def _load_raw_audit_image(record: Tuple[str, int, str]) -> np.ndarray:
    value = np.load(record[0], allow_pickle=True)
    return np.asarray(extract_image_array(value), dtype=np.float32).squeeze()


def _load_audit_subset(
    samples: List[Tuple[str, int, str]],
    indices: np.ndarray,
    io_workers: int,
) -> np.ndarray:
    records = [samples[int(index)] for index in indices]
    workers = max(1, min(io_workers, _AUDIT_POOL))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = list(pool.map(_load_raw_audit_image, records))
    return np.stack(images).astype(np.float32, copy=False)


def _model_visible_chunk(images: np.ndarray, image_size: int) -> np.ndarray:
    visible = np.asarray(images, dtype=np.float32).copy()
    maxima = visible.max(axis=(1, 2), keepdims=True)
    divide = maxima[:, 0, 0] > 1.0
    visible[divide] /= maxima[divide]
    if visible.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            torch.from_numpy(visible).unsqueeze(1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        visible = tensor[:, 0].numpy()
    return visible


def _d4_feature_chunk(images: np.ndarray) -> np.ndarray:
    """Extract fixed D4-invariant radial and morphology summaries."""

    x = np.asarray(images, dtype=np.float32)
    count, height, width = x.shape
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    radius = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    edges = np.linspace(0.0, np.sqrt(2.0) + 1e-6, _AUDIT_ANNULI + 1)

    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    gx = 0.5 * (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2])
    gy = 0.5 * (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1])
    gradient = np.hypot(gx, gy)
    laplacian = (
        padded[:, 1:-1, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, :-2, 1:-1]
        - 4.0 * x
    )
    smooth = sum(
        padded[:, dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
    ) / 9.0
    highpass = x - smooth

    features: List[np.ndarray] = []
    weights_all = np.abs(x)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (radius >= lower) & (radius < upper)
        for channel in (x, gradient, laplacian, highpass):
            features.append(np.sqrt(np.mean(channel[:, mask] ** 2, axis=1)))
        weights = weights_all[:, mask]
        denominator = np.maximum(weights.sum(axis=1), 1e-8)
        ring_theta = theta[mask]
        for mode in range(1, 5):
            moment = (
                weights * np.exp(1j * mode * ring_theta)[None, :]
            ).sum(axis=1) / denominator
            features.append(np.abs(moment))

    # Use the full Fourier plane: an unweighted rFFT half-plane gives the
    # Nyquist/DC boundary different multiplicities after a 90-degree rotation.
    power = np.abs(np.fft.fft2(x, axes=(-2, -1))) ** 2
    frequency_y = np.fft.fftfreq(height)[:, None]
    frequency_x = np.fft.fftfreq(width)[None, :]
    frequency_radius = np.sqrt(frequency_x**2 + frequency_y**2)
    frequency_edges = np.linspace(
        0.0, float(frequency_radius.max()) + 1e-8, _AUDIT_ANNULI + 1
    )
    for lower, upper in zip(frequency_edges[:-1], frequency_edges[1:]):
        mask = (frequency_radius >= lower) & (frequency_radius < upper)
        features.append(np.log1p(power[:, mask].mean(axis=1)))

    orbit = []
    for rotation in range(4):
        rotated = np.rot90(x, rotation, axes=(-2, -1))
        orbit.extend((rotated, np.flip(rotated, axis=-1)))
    invariant_image = np.mean(orbit, axis=0)
    if height % 8 == 0 and width % 8 == 0:
        coarse = invariant_image.reshape(
            count, 8, height // 8, 8, width // 8
        ).mean(axis=(2, 4))
    else:
        coarse = F.adaptive_avg_pool2d(
            torch.from_numpy(invariant_image).unsqueeze(1), (8, 8)
        )[:, 0].numpy()
    features.extend(coarse.reshape(count, -1).T)
    features.extend(
        (
            x.mean(axis=(1, 2)),
            x.std(axis=(1, 2)),
            x.min(axis=(1, 2)),
            x.max(axis=(1, 2)),
            np.mean(np.abs(x - invariant_image), axis=(1, 2)),
        )
    )
    output = np.stack(features, axis=1).astype(np.float32)
    return np.nan_to_num(output, nan=0.0, posinf=1e20, neginf=-1e20)


def _audit_features(
    images: np.ndarray,
    model_visible: bool,
    image_size: int,
    chunk_size: int = 256,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size]
        if model_visible:
            chunk = _model_visible_chunk(chunk, image_size)
        chunks.append(_d4_feature_chunk(chunk))
    return np.concatenate(chunks, axis=0)


def _fit_fixed_ridge(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
) -> np.ndarray:
    train = np.asarray(train_features, dtype=np.float64)
    validation = np.asarray(validation_features, dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = np.clip((train - mean) / scale, -30.0, 30.0)
    validation = np.clip((validation - mean) / scale, -30.0, 30.0)
    train = np.concatenate((np.ones((len(train), 1)), train), axis=1)
    validation = np.concatenate(
        (np.ones((len(validation), 1)), validation), axis=1
    )
    targets = np.eye(len(EXPECTED_CLASSES), dtype=np.float64)[train_labels]
    penalty = np.eye(train.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        train.T @ train + penalty,
        train.T @ targets,
    )
    return validation @ weights


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop + 1)
        start = stop
    return ranks


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = np.asarray(labels, dtype=bool)
    positive_count = int(positive.sum())
    negative_count = len(positive) - positive_count
    if not positive_count or not negative_count:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    numerator = ranks[positive].sum() - positive_count * (positive_count + 1) / 2
    return float(numerator / (positive_count * negative_count))


def _score_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    prediction = np.asarray(scores).argmax(axis=1)
    confusion = np.zeros((len(EXPECTED_CLASSES), len(EXPECTED_CLASSES)), dtype=int)
    for truth, predicted in zip(labels, prediction):
        confusion[int(truth), int(predicted)] += 1
    recalls = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(len(EXPECTED_CLASSES), dtype=float),
        where=confusion.sum(axis=1) != 0,
    )
    f1_values = []
    for label in range(len(EXPECTED_CLASSES)):
        true_positive = confusion[label, label]
        false_positive = confusion[:, label].sum() - true_positive
        false_negative = confusion[label, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    per_class_auc = [
        _binary_auc(labels == label, scores[:, label])
        for label in range(len(EXPECTED_CLASSES))
    ]
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_auc_ovr": float(np.mean(per_class_auc)),
        "per_class_auc": per_class_auc,
        "confusion_matrix": confusion.tolist(),
    }


def _bootstrap_probe(
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    class_indices = [
        np.flatnonzero(labels == label) for label in range(len(EXPECTED_CLASSES))
    ]
    balanced_accuracy = np.empty(MODEL_IV_AUDIT_BOOTSTRAPS)
    macro_auc = np.empty(MODEL_IV_AUDIT_BOOTSTRAPS)
    per_class_auc = np.empty(
        (MODEL_IV_AUDIT_BOOTSTRAPS, len(EXPECTED_CLASSES))
    )
    for iteration in range(MODEL_IV_AUDIT_BOOTSTRAPS):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        metrics = _score_metrics(labels[sampled], scores[sampled])
        balanced_accuracy[iteration] = metrics["balanced_accuracy"]
        macro_auc[iteration] = metrics["macro_auc_ovr"]
        per_class_auc[iteration] = metrics["per_class_auc"]
    return {
        "balanced_accuracy_ci95": np.quantile(
            balanced_accuracy, (0.025, 0.975)
        ).tolist(),
        "macro_auc_ovr_ci95": np.quantile(
            macro_auc, (0.025, 0.975)
        ).tolist(),
        "per_class_auc_ci95": np.quantile(
            per_class_auc, (0.025, 0.975), axis=0
        ).T.tolist(),
        "bootstrap_repetitions": MODEL_IV_AUDIT_BOOTSTRAPS,
    }


def _permutation_max_t(
    labels: np.ndarray,
    scores_by_view: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = {
        name: _score_metrics(labels, scores)["macro_auc_ovr"]
        for name, scores in scores_by_view.items()
    }
    ranks = {
        name: np.stack(
            [_average_ranks(scores[:, label]) for label in range(len(EXPECTED_CLASSES))],
            axis=1,
        )
        for name, scores in scores_by_view.items()
    }
    positive_counts = np.asarray(
        [np.sum(labels == label) for label in range(len(EXPECTED_CLASSES))]
    )
    negative_counts = len(labels) - positive_counts
    max_null = np.empty(MODEL_IV_AUDIT_PERMUTATIONS)
    for iteration in range(MODEL_IV_AUDIT_PERMUTATIONS):
        permuted = rng.permutation(labels)
        view_statistics = []
        for view_ranks in ranks.values():
            aucs = []
            for label in range(len(EXPECTED_CLASSES)):
                positive = permuted == label
                numerator = view_ranks[positive, label].sum() - (
                    positive_counts[label] * (positive_counts[label] + 1) / 2
                )
                aucs.append(
                    numerator / (positive_counts[label] * negative_counts[label])
                )
            view_statistics.append(float(np.mean(aucs)))
        max_null[iteration] = max(view_statistics)
    return {
        name: float(
            (1 + np.sum(max_null >= statistic))
            / (MODEL_IV_AUDIT_PERMUTATIONS + 1)
        )
        for name, statistic in observed.items()
    }


def _probe_passes(probe: Dict[str, Any]) -> bool:
    return bool(
        probe["balanced_accuracy"] >= 0.40
        and probe["macro_auc_ovr"] >= 0.55
        and probe["macro_auc_ovr_ci95"][0] >= 0.52
        and all(interval[0] > 0.50 for interval in probe["per_class_auc_ci95"])
        and probe["permutation_max_t_p"] <= 0.01
    )


def _schema_summary(
    rows: List[Dict[str, Any]], labels: np.ndarray
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for label, class_name in enumerate(EXPECTED_CLASSES):
        counts: Dict[str, int] = {}
        for row in rows:
            if int(row["label"]) == label:
                schema = str(row["schema"])
                counts[schema] = counts.get(schema, 0) + 1
        summary[class_name] = counts
    return summary


def _write_audit_report(
    path: Path,
    status: str,
    counts: Dict[str, Dict[str, int]],
    schemas: Dict[str, Dict[str, Dict[str, int]]],
    integrity: Dict[str, int],
    failures: List[str],
    probes: Dict[str, Dict[str, Any]],
) -> None:
    lines = [
        "# Model IV dataset audit",
        "",
        f"- Status: `{status}`",
        "- Evaluation scope: supplied development-validation only",
        "- Official test evaluated: `false`",
        f"- Fixed seed: `{MODEL_IV_AUDIT_SEED}`",
        "",
        "This is an operational preflight gate. Failure to detect signal with "
        "this fixed probe does not prove that the Bayes-optimal signal is zero.",
        "",
        "## Integrity",
        "",
        "| Split | axion | cdm | no_sub |",
        "| --- | ---: | ---: | ---: |",
        "| development | {axion} | {cdm} | {no_sub} |".format(
            **{
                name: counts.get("development", {}).get(name, 0)
                for name in EXPECTED_CLASSES
            }
        ),
        "| validation | {axion} | {cdm} | {no_sub} |".format(
            **{
                name: counts.get("validation", {}).get(name, 0)
                for name in EXPECTED_CLASSES
            }
        ),
        "",
        f"- Cross-label development digests: `{integrity.get('cross_label_development', 0)}`",
        f"- Cross-label validation digests: `{integrity.get('cross_label_validation', 0)}`",
        f"- Development/validation digest overlap: `{integrity.get('development_validation_overlap', 0)}`",
        f"- Same-label duplicate copies (reported, not by itself fatal): `{integrity.get('same_label_duplicate_copies', 0)}`",
        "",
    ]
    if failures:
        lines.extend(("### Integrity failures", ""))
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    lines.extend(
        (
            "### Raw serialization schemas (warning-only provenance evidence)",
            "",
        )
    )
    for split_name, split_schemas in schemas.items():
        lines.append(f"- **{split_name}**")
        for class_name, class_schemas in split_schemas.items():
            rendered = ", ".join(
                f"`{schema}`: {count}" for schema, count in sorted(class_schemas.items())
            )
            lines.append(f"  - {class_name}: {rendered}")
    lines.extend(
        (
            "",
            "Schema differences are never used as classifier features and are "
            "not treated as proof of pixel corruption.",
            "",
            "## Frozen signal probe",
            "",
            "The probe uses only pixels: D4-invariant annular intensity, "
            "gradient, Laplacian and high-pass energies; angular multipole "
            "magnitudes; radial Fourier power; and an eight-view symmetrized "
            "coarse image. A fixed one-vs-rest ridge model is standardized on "
            "development data only and scored once on supplied validation.",
            "",
        )
    )
    if probes:
        lines.extend(
            (
                "| View | N train | N validation | Accuracy | Balanced accuracy (95% CI) | Macro OVR AUC (95% CI) | max-T p | Pass |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            )
        )
        for name in ("raw", "model_visible"):
            if name not in probes:
                continue
            probe = probes[name]
            ba_ci = probe["balanced_accuracy_ci95"]
            auc_ci = probe["macro_auc_ovr_ci95"]
            lines.append(
                f"| {name} | {probe['n_train']} | {probe['n_validation']} | "
                f"{probe['accuracy']:.5f} | {probe['balanced_accuracy']:.5f} "
                f"[{ba_ci[0]:.5f}, {ba_ci[1]:.5f}] | "
                f"{probe['macro_auc_ovr']:.5f} "
                f"[{auc_ci[0]:.5f}, {auc_ci[1]:.5f}] | "
                f"{probe['permutation_max_t_p']:.4f} | "
                f"{'yes' if probe['passes'] else 'no'} |"
            )
        lines.extend(("", "Per-class OVR AUC confidence intervals:", ""))
        for name, probe in probes.items():
            rendered = ", ".join(
                f"{class_name}={probe['per_class_auc'][index]:.5f} "
                f"[{probe['per_class_auc_ci95'][index][0]:.5f}, "
                f"{probe['per_class_auc_ci95'][index][1]:.5f}]"
                for index, class_name in enumerate(EXPECTED_CLASSES)
            )
            lines.append(f"- **{name}:** {rendered}")
        lines.extend(
            (
                "",
                "A view passes only when balanced accuracy is at least 0.40, "
                "macro AUC is at least 0.55, its bootstrap lower bound is at "
                "least 0.52, every class-AUC lower bound exceeds 0.50, and the "
                "two-view max-T permutation p-value is at most 0.01.",
                "",
                "The archive has no pair/source IDs, so confidence intervals "
                "use a sample-level stratified bootstrap and can be optimistic "
                "under source reuse. A repaired release must use grouped "
                "source/pair inference.",
                "",
            )
        )
    if status == PREPROCESSING_SIGNAL_LOSS:
        lines.append(
            "Raw pixels pass while the exact model-visible view does not; "
            "training is blocked until preprocessing preserves the signal."
        )
    elif status == INCONCLUSIVE_NO_SIGNAL_DETECTED:
        lines.append(
            "Neither fixed view demonstrates held-out signal. The archive is "
            "quarantined before GPU training; this is not a proof of no signal."
        )
    elif status == INTEGRITY_FAILED:
        lines.append("Definite integrity failures block any signal interpretation.")
    elif status == PASS_SIGNAL_DETECTED:
        lines.append("The model-visible development-validation signal gate passes.")
    lines.append("")
    path.write_text("\n".join(lines))


def run_model_iv_audit(config: Config, output_root: str | Path) -> ModelIVAuditResult:
    """Run the fixed Model-IV integrity and signal audit without CUDA."""

    if config.dataset_id != "model_iv" or config.validation_path is None:
        raise ValueError("The Model-IV audit requires Model IV and supplied validation")
    report_path = Path(output_root) / "dataset_audit.md"
    failures: List[str] = []
    probes: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    schemas: Dict[str, Dict[str, Dict[str, int]]] = {}
    integrity: Dict[str, int] = {}
    try:
        development_samples, development_labels, development_rows, dev_failures = (
            _scan_audit_root(
                config.development_path, "development", config.io_workers
            )
        )
        validation_samples, validation_labels, validation_rows, val_failures = (
            _scan_audit_root(
                config.validation_path, "validation", config.io_workers
            )
        )
        failures.extend(dev_failures)
        failures.extend(val_failures)
        digest_failures, integrity = _digest_integrity(
            development_rows, validation_rows
        )
        failures.extend(digest_failures)
        counts = {
            "development": {
                class_name: int(np.sum(development_labels == label))
                for label, class_name in enumerate(EXPECTED_CLASSES)
            },
            "validation": {
                class_name: int(np.sum(validation_labels == label))
                for label, class_name in enumerate(EXPECTED_CLASSES)
            },
        }
        schemas = {
            "development": _schema_summary(development_rows, development_labels),
            "validation": _schema_summary(validation_rows, validation_labels),
        }
    except Exception as error:
        failures.append(f"dataset discovery failed: {type(error).__name__}: {error}")
        _write_audit_report(
            report_path,
            INTEGRITY_FAILED,
            counts,
            schemas,
            integrity,
            failures,
            probes,
        )
        return ModelIVAuditResult(
            INTEGRITY_FAILED, report_path, failures, probes
        )

    if failures:
        _write_audit_report(
            report_path,
            INTEGRITY_FAILED,
            counts,
            schemas,
            integrity,
            failures,
            probes,
        )
        return ModelIVAuditResult(
            INTEGRITY_FAILED, report_path, failures, probes
        )

    development_indices = _balanced_audit_indices(
        development_labels,
        MODEL_IV_AUDIT_TRAIN_CAP,
        MODEL_IV_AUDIT_SEED + 1,
    )
    validation_indices = _balanced_audit_indices(
        validation_labels,
        MODEL_IV_AUDIT_VALIDATION_CAP,
        MODEL_IV_AUDIT_SEED + 2,
    )
    development_images = _load_audit_subset(
        development_samples, development_indices, config.io_workers
    )
    validation_images = _load_audit_subset(
        validation_samples, validation_indices, config.io_workers
    )
    train_labels = development_labels[development_indices]
    heldout_labels = validation_labels[validation_indices]

    scores_by_view: Dict[str, np.ndarray] = {}
    for name, model_visible in (("raw", False), ("model_visible", True)):
        train_features = _audit_features(
            development_images, model_visible, config.image_size
        )
        validation_features = _audit_features(
            validation_images, model_visible, config.image_size
        )
        scores_by_view[name] = _fit_fixed_ridge(
            train_features, train_labels, validation_features
        )
        print(
            f"MODEL_IV_AUDIT_PROBE_FEATURES view={name} "
            f"dimension={train_features.shape[1]}",
            flush=True,
        )

    corrected_p = _permutation_max_t(
        heldout_labels, scores_by_view, MODEL_IV_AUDIT_SEED + 3
    )
    for offset, (name, scores) in enumerate(scores_by_view.items()):
        probe = _score_metrics(heldout_labels, scores)
        probe.update(
            _bootstrap_probe(
                heldout_labels,
                scores,
                MODEL_IV_AUDIT_SEED + 100 + offset,
            )
        )
        probe.update(
            {
                "n_train": int(len(train_labels)),
                "n_validation": int(len(heldout_labels)),
                "permutation_max_t_p": corrected_p[name],
            }
        )
        probe["passes"] = _probe_passes(probe)
        probes[name] = probe

    if probes["model_visible"]["passes"]:
        status = PASS_SIGNAL_DETECTED
    elif probes["raw"]["passes"]:
        status = PREPROCESSING_SIGNAL_LOSS
    else:
        status = INCONCLUSIVE_NO_SIGNAL_DETECTED
    _write_audit_report(
        report_path,
        status,
        counts,
        schemas,
        integrity,
        failures,
        probes,
    )
    return ModelIVAuditResult(status, report_path, failures, probes)


def fixed_stratified_split(
    labels: np.ndarray,
    split_path: str | Path,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create or verify the fixed class-stratified development split."""

    split_path = Path(split_path)
    if split_path.exists():
        saved = np.load(split_path)
        train, validation = saved["train"], saved["val"]
        if "seed" in saved and int(saved["seed"]) != seed:
            raise RuntimeError(f"Cached split seed mismatch in {split_path}")
        if "val_fraction" in saved and not np.isclose(
            float(saved["val_fraction"]),
            val_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Cached validation fraction mismatch in {split_path}"
            )
    else:
        rng = np.random.default_rng(seed)
        train_parts, validation_parts = [], []
        for label in sorted(np.unique(labels).tolist()):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            validation_count = int(round(len(indices) * val_fraction))
            validation_parts.append(indices[:validation_count])
            train_parts.append(indices[validation_count:])
        train = np.concatenate(train_parts)
        validation = np.concatenate(validation_parts)
        rng.shuffle(train)
        rng.shuffle(validation)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = split_path.with_name(
            f"{split_path.stem}-building-{os.getpid()}.npz"
        )
        np.savez(
            temporary,
            train=train,
            val=validation,
            seed=seed,
            val_fraction=val_fraction,
        )
        os.replace(temporary, split_path)

    train = np.asarray(train, dtype=np.int64)
    validation = np.asarray(validation, dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1:
        raise RuntimeError("Split indices must be one-dimensional")
    if len(np.unique(train)) != len(train) or len(np.unique(validation)) != len(
        validation
    ):
        raise RuntimeError("Split contains duplicate indices")
    if np.intersect1d(train, validation, assume_unique=True).size:
        raise RuntimeError("Training and validation indices overlap")
    if len(train) + len(validation) != len(labels):
        raise RuntimeError("Split does not cover the development dataset")
    combined = np.sort(np.concatenate((train, validation)))
    if not np.array_equal(combined, np.arange(len(labels))):
        raise RuntimeError("Split coverage does not match development indices")
    return train, validation


class CachedNPYDataset(Dataset):
    def __init__(
        self, cache_dir: str | Path, indices: Optional[np.ndarray] = None
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.images_path = self.cache_dir / "images.npy"
        self.labels = np.load(self.cache_dir / "labels.npy")
        self.indices = (
            np.arange(len(self.labels), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self._images = None

    @property
    def images(self):
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        return self._images

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        image = np.array(self.images[index], copy=True)
        return (
            torch.from_numpy(image).unsqueeze(0),
            int(self.labels[index]),
            index,
        )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_images"] = None
        return state


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    options = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )
    if workers:
        options.update(persistent_workers=True, prefetch_factor=3)
    return DataLoader(**options)


@dataclass(slots=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    class_names: List[str]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    metadata: Dict


def build_loaders(
    config: Config,
    seed: int,
    device: torch.device,
) -> LoaderBundle:
    """Build loaders with an internal or supplied development-validation split."""

    development_cache_dir = config.cache_path / config.cache_key
    cache_storage_dtype = (
        np.float32 if config.dataset_id == "model_iv" else np.float16
    )
    development_metadata = prepare_cache(
        config.development_path,
        development_cache_dir,
        config.image_size,
        device,
        io_workers=config.io_workers,
        storage_dtype=cache_storage_dtype,
    )
    development_labels = np.load(development_cache_dir / "labels.npy")
    validation_path = config.validation_path
    metadata = dict(development_metadata)

    if validation_path is None:
        train_indices, validation_indices = fixed_stratified_split(
            development_labels,
            config.output_path / "split_indices.npz",
            val_fraction=config.val_fraction,
            seed=config.split_seed,
        )
        train_dataset = CachedNPYDataset(
            development_cache_dir, train_indices
        )
        validation_dataset = CachedNPYDataset(
            development_cache_dir, validation_indices
        )
        metadata["validation_mode"] = "fixed_stratified_development_split"
    else:
        validation_cache_dir = (
            config.cache_path / f"{config.cache_key}_validation"
        )
        validation_metadata = prepare_cache(
            validation_path,
            validation_cache_dir,
            config.image_size,
            device,
            io_workers=config.io_workers,
            storage_dtype=cache_storage_dtype,
        )
        if development_metadata["classes"] != validation_metadata["classes"]:
            raise RuntimeError(
                "Development and validation class mappings do not match: "
                f"{development_metadata['classes']} != "
                f"{validation_metadata['classes']}"
            )
        _require_disjoint_visible_content(
            development_cache_dir, validation_cache_dir
        )
        validation_labels = np.load(validation_cache_dir / "labels.npy")
        train_indices = np.arange(len(development_labels), dtype=np.int64)
        validation_indices = np.arange(
            len(validation_labels), dtype=np.int64
        )
        train_dataset = CachedNPYDataset(
            development_cache_dir, train_indices
        )
        validation_dataset = CachedNPYDataset(
            validation_cache_dir, validation_indices
        )
        metadata["validation_mode"] = "supplied_development_validation"
        metadata["visible_content_overlap"] = 0
        metadata["validation"] = validation_metadata

    train_loader = make_loader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        workers=config.workers,
        seed=seed,
    )
    metadata["training_sampler"] = {"kind": "random_batches"}

    return LoaderBundle(
        train=train_loader,
        validation=make_loader(
            validation_dataset,
            config.batch_size,
            shuffle=False,
            workers=config.workers,
            seed=seed + 10_000,
        ),
        class_names=list(development_metadata["classes"]),
        train_indices=train_indices,
        validation_indices=validation_indices,
        metadata=metadata,
    )
