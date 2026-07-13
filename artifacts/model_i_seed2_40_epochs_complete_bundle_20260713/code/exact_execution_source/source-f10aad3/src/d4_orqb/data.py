"""Leakage-safe Model-I loading, persistent resize cache, and fixed splits."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_KEYS = ("image", "img", "x", "data", "array", "arr", "lens", "sample")


def extract_image_array(value) -> np.ndarray:
    """Extract only the 2-D image; scalar axion mass metadata is discarded."""

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
    """Apply the repository's model-visible preprocessing and return its digest."""

    raw = np.load(path, allow_pickle=True)
    image = np.asarray(extract_image_array(raw), dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image in {path}, got {image.shape}")
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(image.max())
    if maximum > 1.0:
        image = image / maximum
    image = np.ascontiguousarray(image.astype("<f4", copy=False))
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    return image, digest


def list_samples(root: str | Path) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    samples: List[Tuple[str, int, str]] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        for path in sorted(class_dir.glob("*.npy")):
            samples.append((str(path), label, str(path.relative_to(root))))
    if not samples:
        raise RuntimeError(f"No .npy samples under {root}")
    return samples, classes


def _load_path(record: Tuple[str, int, str]) -> Tuple[np.ndarray, int, str, str]:
    path, label, relative = record
    image, digest = load_model_visible_image(path)
    return image, label, relative, digest


def prepare_cache(
    source_root: str | Path,
    cache_dir: str | Path,
    image_size: int,
    device: torch.device,
    io_workers: int = 12,
    chunk_size: int = 384,
) -> Dict:
    """Create an atomic float16 NPY cache, labels, and digest manifest."""

    source_root = Path(source_root).resolve()
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        required = (cache_dir / "images.npy", cache_dir / "labels.npy", cache_dir / "manifest.csv")
        if (
            metadata.get("complete")
            and metadata.get("image_size") == image_size
            and Path(metadata.get("source_root", "")) == source_root
            and all(path.exists() for path in required)
        ):
            print(f"CACHE_READY {cache_dir} samples={metadata['samples']}", flush=True)
            return metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    samples, classes = list_samples(source_root)
    build_tag = f"building-{os.getpid()}"
    image_tmp = cache_dir / f"images-{build_tag}.npy"
    labels_tmp = cache_dir / f"labels-{build_tag}.npy"
    manifest_tmp = cache_dir / f"manifest-{build_tag}.csv"
    images_mm = np.lib.format.open_memmap(
        image_tmp, mode="w+", dtype=np.float16, shape=(len(samples), image_size, image_size)
    )
    labels = np.empty(len(samples), dtype=np.int64)

    with manifest_tmp.open("w", newline="") as manifest_handle:
        writer = csv.writer(manifest_handle)
        writer.writerow(("index", "relative_path", "class", "label", "sha256_visible"))
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            for start in range(0, len(samples), chunk_size):
                stop = min(start + chunk_size, len(samples))
                loaded = list(pool.map(_load_path, samples[start:stop]))
                batch = np.stack([item[0] for item in loaded], axis=0)
                tensor = torch.from_numpy(batch).unsqueeze(1).to(device=device, dtype=torch.float32)
                resized = F.interpolate(
                    tensor,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                images_mm[start:stop] = resized[:, 0].to(dtype=torch.float16).cpu().numpy()
                for offset, (_, label, relative, digest) in enumerate(loaded):
                    index = start + offset
                    labels[index] = label
                    writer.writerow((index, relative, classes[label], label, digest))
                if start == 0 or stop == len(samples) or stop % (chunk_size * 20) == 0:
                    print(f"CACHE_PROGRESS {stop}/{len(samples)}", flush=True)
    images_mm.flush()
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
        "class_counts": {classes[i]: int((labels == i).sum()) for i in range(len(classes))},
        "normalization": "float32 nonfinite cleanup; divide by max only when max > 1",
        "interpolation": "bilinear align_corners=False antialias=True",
        "dtype": "float16",
    }
    metadata_tmp = cache_dir / f"metadata-{build_tag}.json"
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(metadata_tmp, metadata_path)
    print(f"CACHE_COMPLETE {cache_dir} {json.dumps(metadata, sort_keys=True)}", flush=True)
    return metadata


def manifest_digests(cache_dir: str | Path) -> set[str]:
    with (Path(cache_dir) / "manifest.csv").open(newline="") as handle:
        return {row["sha256_visible"] for row in csv.DictReader(handle)}


def verify_cache_disjoint(dev_cache: str | Path, test_cache: str | Path) -> Dict[str, int]:
    development = manifest_digests(dev_cache)
    test = manifest_digests(test_cache)
    result = {
        "development_unique": len(development),
        "test_unique": len(test),
        "intersection": len(development.intersection(test)),
    }
    if result["intersection"]:
        raise RuntimeError(f"Model-visible development/test collision: {result}")
    print(f"CACHE_DISJOINT {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def fixed_stratified_split(
    labels: np.ndarray,
    split_path: str | Path,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    split_path = Path(split_path)
    if split_path.exists():
        saved = np.load(split_path)
        train, val = saved["train"], saved["val"]
        if "seed" in saved and int(saved["seed"]) != seed:
            raise RuntimeError(f"Cached split seed mismatch in {split_path}")
        if "val_fraction" in saved and not np.isclose(
            float(saved["val_fraction"]), val_fraction, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"Cached validation fraction mismatch in {split_path}")
    else:
        rng = np.random.default_rng(seed)
        train_parts, val_parts = [], []
        for label in sorted(np.unique(labels).tolist()):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            n_val = int(round(len(indices) * val_fraction))
            val_parts.append(indices[:n_val])
            train_parts.append(indices[n_val:])
        train = np.concatenate(train_parts)
        val = np.concatenate(val_parts)
        rng.shuffle(train)
        rng.shuffle(val)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = split_path.with_name(f"{split_path.stem}-building-{os.getpid()}.npz")
        np.savez(tmp, train=train, val=val, seed=seed, val_fraction=val_fraction)
        os.replace(tmp, split_path)
    train = np.asarray(train, dtype=np.int64)
    val = np.asarray(val, dtype=np.int64)
    if train.ndim != 1 or val.ndim != 1:
        raise RuntimeError("Train/validation indices must be one-dimensional")
    if len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val):
        raise RuntimeError("Train/validation split contains duplicate indices")
    if len(train) and (train.min() < 0 or train.max() >= len(labels)):
        raise RuntimeError("Training split contains out-of-range indices")
    if len(val) and (val.min() < 0 or val.max() >= len(labels)):
        raise RuntimeError("Validation split contains out-of-range indices")
    if np.intersect1d(train, val, assume_unique=True).size:
        raise RuntimeError("Train/validation indices overlap")
    if len(train) + len(val) != len(labels):
        raise RuntimeError("Split does not cover the development set")
    if not np.array_equal(np.sort(np.concatenate((train, val))), np.arange(len(labels))):
        raise RuntimeError("Split coverage does not match development indices")
    return train, val


def deterministic_subset(
    indices: np.ndarray, labels: np.ndarray, max_per_class: Optional[int], seed: int
) -> np.ndarray:
    if not max_per_class:
        return indices
    rng = np.random.default_rng(seed)
    parts = []
    for label in sorted(np.unique(labels).tolist()):
        candidates = indices[labels[indices] == label].copy()
        rng.shuffle(candidates)
        parts.append(candidates[:max_per_class])
    selected = np.concatenate(parts)
    rng.shuffle(selected)
    return selected


HASH_SUBSET_DOMAIN = b"D4ORQB-M1-RD-v1\0"
OOF_FOLD_DOMAIN = b"D4ORQB-M1-OOF-v1\0"


def hash_ranked_subset(
    indices: np.ndarray,
    labels: np.ndarray,
    max_per_class: Optional[int],
    development_manifest_sha256: str,
) -> np.ndarray:
    """Return a cross-version-stable, nested class-balanced subset.

    Membership is independent of model initialisation and NumPy RNG details.
    Increasing ``max_per_class`` produces a superset for the same canonical
    split and development manifest.
    """

    if not max_per_class:
        return np.asarray(indices, dtype=np.int64)
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    if len(development_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in development_manifest_sha256
    ):
        raise ValueError("development manifest SHA-256 must be lowercase hexadecimal")

    prefix = (
        HASH_SUBSET_DOMAIN
        + development_manifest_sha256.encode("ascii")
        + b"\0"
    )
    parts = []
    for label in sorted(np.unique(labels).tolist()):
        candidates = np.asarray(indices[labels[indices] == label], dtype=np.int64)
        ranked = sorted(
            candidates.tolist(),
            key=lambda index: (
                hashlib.sha256(
                    prefix
                    + int(label).to_bytes(1, "little", signed=False)
                    + int(index).to_bytes(8, "little", signed=False)
                ).digest(),
                int(index),
            ),
        )
        parts.append(np.asarray(ranked[:max_per_class], dtype=np.int64))
    return np.sort(np.concatenate(parts))


def index_membership_sha256(indices: np.ndarray) -> str:
    canonical = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def stratified_hash_folds(
    indices: np.ndarray,
    labels: np.ndarray,
    fold_count: int,
    development_manifest_sha256: str,
) -> Tuple[np.ndarray, ...]:
    """Partition fixed membership into stable, nearly equal stratified folds."""

    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    if fold_count < 2:
        raise ValueError("OOF fold_count must be at least two")
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("OOF source indices must be unique and one-dimensional")
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(labels):
        raise ValueError("OOF source indices are empty or out of range")
    if len(development_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in development_manifest_sha256
    ):
        raise ValueError("development manifest SHA-256 must be lowercase hexadecimal")

    prefix = OOF_FOLD_DOMAIN + development_manifest_sha256.encode("ascii") + b"\0"
    fold_parts = [[] for _ in range(fold_count)]
    ordered_labels = sorted(np.unique(labels[indices]).tolist())
    for label_position, label in enumerate(ordered_labels):
        candidates = indices[labels[indices] == label]
        ranked = sorted(
            candidates.tolist(),
            key=lambda index: (
                hashlib.sha256(
                    prefix
                    + int(label).to_bytes(1, "little", signed=False)
                    + int(index).to_bytes(8, "little", signed=False)
                ).digest(),
                int(index),
            ),
        )
        for part_index, part in enumerate(np.array_split(ranked, fold_count)):
            fold_index = (part_index + label_position) % fold_count
            fold_parts[fold_index].append(np.asarray(part, dtype=np.int64))

    folds = tuple(
        np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
        for parts in fold_parts
    )
    combined = np.concatenate(folds)
    if len(np.unique(combined)) != len(indices) or not np.array_equal(
        np.sort(combined), np.sort(indices)
    ):
        raise RuntimeError("OOF folds do not exactly partition source membership")
    for left in range(fold_count):
        for right in range(left + 1, fold_count):
            if np.intersect1d(folds[left], folds[right], assume_unique=True).size:
                raise RuntimeError("OOF folds overlap")
    return folds


class CachedNPYDataset(Dataset):
    def __init__(self, cache_dir: str | Path, indices: Optional[np.ndarray] = None) -> None:
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
        return torch.from_numpy(image).unsqueeze(0), int(self.labels[index]), index

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
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=3)
    return DataLoader(**kwargs)
