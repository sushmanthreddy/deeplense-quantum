"""Leakage-safe Model-I loading, resize caching, splitting, and loaders."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config


IMAGE_KEYS = ("image", "img", "x", "data", "array", "arr", "lens", "sample")


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
    samples: List[Tuple[str, int, str]] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        for path in sorted(class_dir.glob("*.npy")):
            samples.append((str(path), label, str(path.relative_to(root))))
    if not samples:
        raise RuntimeError(f"No .npy samples under {root}")
    expected_classes = ["axion", "cdm", "no_sub"]
    if classes != expected_classes:
        raise RuntimeError(
            "Model-I class directories must be exactly "
            f"{expected_classes}; found {classes}"
        )
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
) -> Dict:
    """Create or reuse an atomic float16 resize cache."""

    source_root = Path(source_root).resolve()
    cache_dir = Path(cache_dir)
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
            and Path(metadata.get("source_root", "")) == source_root
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
        dtype=np.float16,
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
                images_memmap[start:stop] = (
                    resized[:, 0].to(dtype=torch.float16).cpu().numpy()
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
        "dtype": "float16",
    }
    metadata_tmp = cache_dir / f"metadata-{build_tag}.json"
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(metadata_tmp, metadata_path)
    print(f"CACHE_COMPLETE {cache_dir}", flush=True)
    return metadata


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
    """Build loaders for one stage using the shared cache and fixed split."""

    cache_dir = config.cache_path / f"model_i_{config.image_size}"
    metadata = prepare_cache(
        config.development_path,
        cache_dir,
        config.image_size,
        device,
        io_workers=config.io_workers,
    )
    labels = np.load(cache_dir / "labels.npy")
    train_indices, validation_indices = fixed_stratified_split(
        labels,
        config.output_path / "split_indices.npz",
        val_fraction=config.val_fraction,
        seed=config.split_seed,
    )
    train_dataset = CachedNPYDataset(cache_dir, train_indices)
    validation_dataset = CachedNPYDataset(cache_dir, validation_indices)
    return LoaderBundle(
        train=make_loader(
            train_dataset,
            config.batch_size,
            shuffle=True,
            workers=config.workers,
            seed=seed,
        ),
        validation=make_loader(
            validation_dataset,
            config.batch_size,
            shuffle=False,
            workers=config.workers,
            seed=seed + 10_000,
        ),
        class_names=list(metadata["classes"]),
        train_indices=train_indices,
        validation_indices=validation_indices,
        metadata=metadata,
    )
