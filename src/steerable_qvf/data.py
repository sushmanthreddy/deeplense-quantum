"""Dataset, transforms, and dataloaders for the .npy lensing data.

Faithful refactor of the loader from ``frozen_quantum_model_1.ipynb`` /
``steerable_qvf_quantum_lensing.py``: robust .npy decoding, in-RAM caching,
stratified train/val split, and label-preserving augmentation.
"""

from __future__ import annotations

import collections
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config

IMAGE_KEYS = ("image", "img", "x", "data", "array", "arr", "lens", "sample")


def extract_image_array(value):
    """Return the image array from raw .npy content, including object arrays."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:
                return extract_image_array(value.item())
            for item in value.reshape(-1):
                try:
                    candidate = extract_image_array(item)
                    if np.asarray(candidate).ndim >= 2:
                        return candidate
                except Exception:
                    continue
            return np.asarray(value.tolist())
        return value

    if isinstance(value, dict):
        for key in IMAGE_KEYS:
            if key in value:
                return extract_image_array(value[key])
        for item in value.values():
            try:
                candidate = extract_image_array(item)
                if np.asarray(candidate).ndim >= 2:
                    return candidate
            except Exception:
                continue
        raise ValueError("Could not find an image-like array in .npy dict")

    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                candidate = extract_image_array(item)
                if np.asarray(candidate).ndim >= 2:
                    return candidate
            except Exception:
                continue
        return np.asarray(value)

    return np.asarray(value)


def load_npy_image(filepath):
    raw = np.load(filepath, allow_pickle=True)
    arr = np.asarray(extract_image_array(raw))
    if arr.dtype == object:
        arr = np.asarray(extract_image_array(arr), dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] not in (1, 3):
            arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Expected a 2D or 3D image array from {filepath}, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError(f"Empty image array in {filepath}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr_max = float(np.max(arr))
    if arr_max > 1.0:
        arr = arr / arr_max
    return arr.astype(np.float32, copy=False)


class NPYImageFolder(Dataset):
    """Dataset loader for .npy image files organized in class folders."""

    def __init__(self, root_dir, transform=None, samples=None, classes=None,
                 class_to_idx=None, cache=False):
        self.root_dir = str(root_dir)
        self.transform = transform
        self.cache = None
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.classes = list(classes) if classes is not None else sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )
        self.class_to_idx = dict(class_to_idx) if class_to_idx is not None else {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        if samples is None:
            self.samples = []
            for class_name in self.classes:
                class_dir = os.path.join(self.root_dir, class_name)
                for filename in sorted(os.listdir(class_dir)):
                    if filename.endswith(".npy"):
                        filepath = os.path.join(class_dir, filename)
                        self.samples.append((filepath, self.class_to_idx[class_name]))
        else:
            self.samples = list(samples)

        if not self.samples:
            raise RuntimeError(f"No .npy files found under {self.root_dir}")

        self.labels = torch.tensor([label for _, label in self.samples], dtype=torch.long)

        print(f"Found {len(self.samples)} samples in {self.root_dir}")
        print(f"Classes: {self.classes}")

        if cache:
            desc = f"cache {Path(self.root_dir).name}"
            print(f"Caching {len(self.samples)} decoded samples from {self.root_dir} in RAM...")
            self.cache = [
                torch.from_numpy(np.ascontiguousarray(load_npy_image(filepath))).float()
                for filepath, _ in tqdm(self.samples, desc=desc, leave=False)
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        if self.cache is None:
            arr = load_npy_image(filepath)
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
        else:
            tensor = self.cache[idx]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


class NPYTransform:
    """Resize and channel-match NPY tensors."""

    def __init__(self, img_size, in_channels=1):
        self.img_size = img_size
        self.in_channels = in_channels

    def __call__(self, x):
        if x.shape[0] == 1 and self.in_channels == 3:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 3 and self.in_channels == 1:
            x = x.mean(dim=0, keepdim=True)
        elif x.shape[0] != self.in_channels:
            x = x[:self.in_channels]

        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return x


class NPYAugment:
    """Train-only augmentation: base resize/channel-match + random flips/rotation/translation.

    Strong-lensing class labels are invariant to flips and rotations, so all of these are
    label-preserving and act purely as a regularizer against the fast train/val overfitting.
    """

    def __init__(self, img_size, in_channels=1, max_rotation=15.0, max_translate=0.1):
        self.base = NPYTransform(img_size, in_channels=in_channels)
        self.img_size = img_size
        self.max_rotation = max_rotation
        self.max_translate = max_translate

    def __call__(self, x):
        x = self.base(x)
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-2])
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        max_d = self.max_translate * self.img_size
        translate = [int(random.uniform(-max_d, max_d)),
                     int(random.uniform(-max_d, max_d))]
        x = TF.affine(
            x, angle=angle, translate=translate, scale=1.0, shear=[0.0, 0.0],
            interpolation=TF.InterpolationMode.BILINEAR, fill=0.0,
        )
        return x


def stratified_train_val_samples(samples, val_split=0.20, seed=42):
    """Split a single class-folder dataset into train/val while preserving each class."""
    labels = np.array([label for _, label in samples])
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for class_id in sorted(np.unique(labels).tolist()):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        if len(class_indices) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(len(class_indices) * val_split)))
            n_val = min(n_val, len(class_indices) - 1)
        val_indices.extend(class_indices[:n_val].tolist())
        train_indices.extend(class_indices[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    if not train_samples or not val_samples:
        raise ValueError("Train/val split is empty. Check val_split and per-class sample counts.")
    return train_samples, val_samples


def make_dataloader(dataset, batch_size, shuffle, cfg: Config):
    pin_memory = torch.cuda.is_available()
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
        # drop the last partial batch only while training -> keeps BatchNorm happy.
        "drop_last": shuffle,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_loaders(cfg: Config):
    """Build (train_loader, val_loader, test_loader, class_names) from a Config."""
    t0 = time.time()
    transform = NPYTransform(cfg.load_img_size, in_channels=cfg.in_channels)
    if cfg.augment:
        train_transform = NPYAugment(
            cfg.load_img_size, in_channels=cfg.in_channels,
            max_rotation=cfg.aug_max_rotation, max_translate=cfg.aug_max_translate,
        )
    else:
        train_transform = transform
    cache_data = cfg.cache_data_in_memory

    full_dataset = NPYImageFolder(cfg.data_root, transform=None, cache=False)
    train_samples, val_samples = stratified_train_val_samples(
        full_dataset.samples, val_split=cfg.val_split, seed=cfg.seed
    )
    train_set = NPYImageFolder(
        cfg.data_root, train_transform, samples=train_samples,
        classes=full_dataset.classes, class_to_idx=full_dataset.class_to_idx, cache=cache_data,
    )
    val_set = NPYImageFolder(
        cfg.data_root, transform, samples=val_samples,
        classes=full_dataset.classes, class_to_idx=full_dataset.class_to_idx, cache=cache_data,
    )
    test_set = NPYImageFolder(cfg.test_dir, transform, cache=cache_data)

    if test_set.classes != full_dataset.classes:
        raise ValueError(
            f"Class folders differ between data_root and test_dir: "
            f"{full_dataset.classes} vs {test_set.classes}"
        )

    train_loader = make_dataloader(train_set, cfg.batch_size, shuffle=True, cfg=cfg)
    val_loader = make_dataloader(val_set, cfg.batch_size, shuffle=False, cfg=cfg)
    test_loader = make_dataloader(test_set, cfg.batch_size, shuffle=False, cfg=cfg)

    counts = collections.Counter(train_set.labels.tolist())
    print(f"Loader build took {time.time() - t0:.1f}s")
    print(f"Sizes: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print("Class counts (train):", {full_dataset.classes[k]: v for k, v in sorted(counts.items())})
    return train_loader, val_loader, test_loader, full_dataset.classes
