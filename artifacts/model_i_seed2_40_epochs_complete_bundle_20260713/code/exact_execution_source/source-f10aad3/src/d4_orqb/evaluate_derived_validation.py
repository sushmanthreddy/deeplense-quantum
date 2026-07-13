"""Derive and evaluate the SHA-locked max-envelope Model-I candidate.

This command has one purpose: compose the frozen annular-Haar primary with the
four Haar-subtype residual tensors from a fixed epoch-20 donor, then evaluate
that derived model once on the canonical development validation split.  It has
no fitting code and deliberately exposes no official-test argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

from .data import (
    CachedNPYDataset,
    hash_ranked_subset,
    index_membership_sha256,
    make_loader,
)
from .evaluate_locked import (
    CACHE_ARTIFACTS,
    MAX_VALIDATION_REPLAY_METRIC_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
    assert_canonical_directory,
    assert_fingerprints_equal,
    assert_no_symlink_components,
    atomic_json,
    atomic_npz,
    canonical_model_i_split,
    file_fingerprint,
    fingerprint_named_files,
    fsync_directory,
    fsync_file,
    load_validation_predictions,
    probability_replay_diagnostics,
    softmax_numpy,
    validate_cache_structure,
)
from .metrics import classification_metrics
from .model import D4OrbitClassifier, max_preserving_subtype_envelope


SCHEMA_VERSION = 1
DERIVATION_ID = "sha-locked-max-subtype-envelope-v1"
TRANSFORM_ID = "max-subtype-envelope-v1"
MODEL_I_CLASSES = ["axion", "cdm", "no_sub"]
MODEL_I_CLASS_COUNTS = {"axion": 28_897, "cdm": 29_772, "no_sub": 28_856}
MODEL_I_SAMPLES = 87_525
MODEL_I_IMAGE_SIZE = 96
TRAIN_SAMPLES = 35_001
TRAIN_PER_CLASS = 11_667
VALIDATION_SAMPLES = 17_504

PRIMARY_CHECKPOINT_SHA256 = (
    "9850a7d0c53b6332739696a9952b94a85a9ecea9c567fe5e0e85bdeef296b144"
)
DONOR_CHECKPOINT_SHA256 = (
    "7e37c72ef9571642b8f314903c780730a4fc68bdb2547f3d746844961297643b"
)
SELECTION_FILE_SHA256 = (
    "02878d3cb807af9f449f3e482f446bfae6558e5233d372687b7497cfe7652b9e"
)
SELECTION_SPEC_SHA256 = (
    "8c8aca6a0cf66ce4d547ec746851c7c698fbc9fe85b6262d70c94085d098d470"
)
SPLIT_FILE_SHA256 = (
    "6fba1c6f36aa7b37775d7525ae0690c75a4814dabb3d2e2c9dd93d667b877eb3"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "c04a3c62afebe3f660ffaad4333b6632471a91c6f5f239f84e68b4b94c330025"
)
DEVELOPMENT_IMAGES_SHA256 = (
    "c3c639584e0a9e2d6ba369e3fb41ba0451b2170a362ec419fa1182e55d5ce070"
)
DEVELOPMENT_LABELS_SHA256 = (
    "3a12100d1df155738b57255e6625ba1287c2d74c5bab53bce2cd0597afe89b17"
)
DEVELOPMENT_METADATA_SHA256 = (
    "9f36faff5fc3300b97512b1c29439b2e993888318a31835938749e10fbf12379"
)
TRAIN_MEMBERSHIP_SHA256 = (
    "571d23ced25095cf0cfb57216654f9b7be289b0589a95489a5a815a866aaee71"
)
PRIMARY_PREDICTIONS_SHA256 = (
    "96e375f1004f27e5032341604764126e9cb7f30ee319387da49df088d9b02569"
)
WEIGHT_TENSOR_SHA256 = (
    "edf10175dab3244725d7b9cef7d9d9d56163775f6937af2221f39bb69c678bcd"
)
SELECTION_TENSOR_SHA256 = {
    "haar_subtype_residual.selected_indices": (
        "ec04f502b1bfab1eb0b5f62f61b8487c3f5cd232d90ac3e40cd8171e7cde571b"
    ),
    "haar_subtype_residual.center": (
        "e3eb1d45d62bb3b7f1613ab336fe2db0692e2882a2eae2db37439e540c247cda"
    ),
    "haar_subtype_residual.scale": (
        "081428f5525a00a3499ab6e916365e220a92bb23ba88490d8e4efae24765711f"
    ),
}
SELECTED_INDICES = [31, 50, 42, 25, 36, 19, 24, 30, 37, 48, 13, 7, 52, 18, 45]
PRIMARY_EPOCH = 17
DONOR_EPOCH = 20
PRIMARY_SEED = 0
PRIMARY_STATE_TENSORS = 115
DERIVED_STATE_TENSORS = 119
PRIMARY_PARAMETERS = 122_595
DERIVED_PARAMETERS = 122_610
RESIDUAL_PARAMETERS = 15

RESIDUAL_STATE_KEYS = (
    "haar_subtype_residual.weight",
    "haar_subtype_residual.selected_indices",
    "haar_subtype_residual.center",
    "haar_subtype_residual.scale",
)
FIXED_FEATURE_STATE_KEYS = (
    "morphology_mean",
    "morphology_scale",
    "haar_mean",
    "haar_scale",
    "morphology_context_indices",
)
PRIMARY_ARTIFACTS = (
    "best.pt",
    "config.json",
    "data_report.json",
    "parameter_report.json",
    "summary.json",
    "split_indices.npz",
    "best_validation_predictions.npz",
)
DONOR_ARTIFACTS = (
    "last.pt",
    "config.json",
    "data_report.json",
    "parameter_report.json",
    "summary.json",
    "initialization_report.json",
    "split_indices.npz",
    "haar_subtype_selection.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_training_runtime(seed: int) -> None:
    """Mirror the nondeterministic CUDA backend used by the frozen source runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def _read_json(path: Path) -> Dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def require_sha256(path: Path, expected: str, context: str) -> Dict[str, int | str]:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"Invalid expected SHA-256 for {context}")
    fingerprint = file_fingerprint(path)
    if fingerprint["sha256"] != expected:
        raise RuntimeError(
            f"{context} SHA-256 mismatch: actual={fingerprint['sha256']} expected={expected}"
        )
    return fingerprint


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def canonical_spec_sha256(value: Mapping, digest_key: str) -> str:
    material = dict(value)
    material.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_fields(actual: Mapping, expected: Mapping, context: str) -> None:
    drift = {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if drift:
        raise RuntimeError(f"{context} field drift: {drift}")


def _validate_state_mapping(state: Mapping[str, torch.Tensor], context: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"{context} model state is empty or invalid")
    for key, tensor in state.items():
        if not isinstance(key, str) or not torch.is_tensor(tensor):
            raise RuntimeError(f"{context} state must map string keys to tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise RuntimeError(f"{context} contains a non-finite tensor: {key}")


def load_sha_locked_checkpoint(
    path: Path,
    expected_sha256: str,
    expected_epoch: int,
    context: str,
) -> Tuple[Dict, Dict[str, torch.Tensor], Dict[str, int | str]]:
    before = require_sha256(path, expected_sha256, context)
    checkpoint = _torch_load(path)
    after = file_fingerprint(path)
    if after != before:
        raise RuntimeError(f"{context} changed while it was being loaded")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model", "epoch", "record"}:
        raise RuntimeError(
            f"{context} must contain exactly model/epoch/record top-level keys"
        )
    if int(checkpoint["epoch"]) != expected_epoch:
        raise RuntimeError(
            f"{context} epoch drift: {checkpoint['epoch']} != {expected_epoch}"
        )
    record = checkpoint["record"]
    if not isinstance(record, Mapping) or int(record.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(f"{context} record epoch disagrees with its checkpoint")
    state = checkpoint["model"]
    _validate_state_mapping(state, context)
    return checkpoint, dict(state), before


def validate_run_metadata(
    primary_dir: Path,
    donor_dir: Path,
) -> Dict:
    primary_config = _read_json(primary_dir / "config.json")
    donor_config = _read_json(donor_dir / "config.json")
    common = {
        "core": "quantum",
        "encoder_variant": "deep-se-haar-morph",
        "physics_variant": "base",
        "physics_summary": "moments-morphology-haar",
        "heads": 4,
        "reuploads": 2,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "include_context": False,
        "dropout": 0.1,
        "image_size": MODEL_I_IMAGE_SIZE,
        "split_seed": 42,
        "val_fraction": 0.2,
        "train_subset_protocol": "hash-v1",
        "max_train_per_class": TRAIN_PER_CLASS,
        "max_val_per_class": None,
        "seed": 0,
        "evaluate_test": False,
    }
    _require_fields(primary_config, common, "primary config")
    _require_fields(donor_config, common, "donor config")
    if any(
        bool(primary_config.get(key, False))
        for key in (
            "tied_mean_dispersion",
            "haar_subtype_residual",
            "haar_subtype_max_envelope",
            "shared_late_refinement",
        )
    ):
        raise RuntimeError("Primary config is not the unextended annular-Haar model")
    if donor_config.get("haar_subtype_residual") is not True or any(
        bool(donor_config.get(key, False))
        for key in (
            "tied_mean_dispersion",
            "haar_subtype_max_envelope",
            "shared_late_refinement",
        )
    ):
        raise RuntimeError("Donor config is not the original additive residual run")
    if int(donor_config.get("epochs", -1)) != DONOR_EPOCH:
        raise RuntimeError("Donor training horizon is not the fixed epoch-20 horizon")

    primary_data = _read_json(primary_dir / "data_report.json")
    donor_data = _read_json(donor_dir / "data_report.json")
    if primary_data != donor_data:
        raise RuntimeError("Primary and donor data reports are not identical")
    expected_data = {
        "class_names": MODEL_I_CLASSES,
        "development_manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
        "train_membership_sha256": TRAIN_MEMBERSHIP_SHA256,
        "train_subset_protocol": "hash-v1",
        "train_size": TRAIN_SAMPLES,
        "validation_size": VALIDATION_SAMPLES,
        "train_counts": {name: TRAIN_PER_CLASS for name in MODEL_I_CLASSES},
        "validation_counts": {"axion": 5_779, "cdm": 5_954, "no_sub": 5_771},
        "official_test_locked_during_selection": True,
        "official_test_cache_opened": False,
    }
    _require_fields(primary_data, expected_data, "source data report")
    development = primary_data.get("development")
    if not isinstance(development, Mapping):
        raise RuntimeError("Source data report lacks development metadata")
    _require_fields(
        development,
        {
            "classes": MODEL_I_CLASSES,
            "class_counts": MODEL_I_CLASS_COUNTS,
            "samples": MODEL_I_SAMPLES,
            "image_size": MODEL_I_IMAGE_SIZE,
            "dtype": "float16",
            "complete": True,
        },
        "source development report",
    )
    if "test" in primary_data or "digest_disjoint" in primary_data:
        raise RuntimeError("Development-only source unexpectedly contains test metadata")

    primary_parameters = _read_json(primary_dir / "parameter_report.json")
    donor_parameters = _read_json(donor_dir / "parameter_report.json")
    _require_fields(
        primary_parameters,
        {
            "total": PRIMARY_PARAMETERS,
            "core": 88,
            "quantum": 88,
            "encoder_variant": "deep-se-haar-morph",
            "physics_summary": "moments-morphology-haar",
        },
        "primary parameter report",
    )
    if any(
        int(primary_parameters.get(key, 0)) != 0
        for key in (
            "haar_subtype_residual_trainable",
            "dispersion_gate_trainable",
            "shared_late_refinement_gate_trainable",
        )
    ):
        raise RuntimeError("Primary parameter report contains an extension")
    _require_fields(
        donor_parameters,
        {
            "total": DERIVED_PARAMETERS,
            "core": 88,
            "quantum": 88,
            "haar_subtype_residual": True,
            "haar_subtype_residual_trainable": RESIDUAL_PARAMETERS,
            "dispersion_gate_trainable": 0,
            "encoder_variant": "deep-se-haar-morph",
            "physics_summary": "moments-morphology-haar",
        },
        "donor parameter report",
    )
    if int(donor_parameters.get("shared_late_refinement_gate_trainable", 0)) != 0:
        raise RuntimeError("Donor parameter report contains shared refinement")

    primary_summary = _read_json(primary_dir / "summary.json")
    donor_summary = _read_json(donor_dir / "summary.json")
    if primary_summary.get("official_test_evaluated") is not False or donor_summary.get(
        "official_test_evaluated"
    ) is not False:
        raise RuntimeError("A source summary does not preserve the official-test lock")

    initialization = _read_json(donor_dir / "initialization_report.json")
    _require_fields(
        initialization,
        {
            "method": "haar-to-zero-subtype-residual-exact-remap",
            "checkpoint_sha256": PRIMARY_CHECKPOINT_SHA256,
            "source_epoch": PRIMARY_EPOCH,
        },
        "donor initialization",
    )
    source = initialization.get("haar_subtype_source")
    if not isinstance(source, Mapping):
        raise RuntimeError("Donor initialization lacks its base-Haar source record")
    _require_fields(
        source,
        {
            "checkpoint_sha256": PRIMARY_CHECKPOINT_SHA256,
            "source_parameters": PRIMARY_PARAMETERS,
            "source_core": "quantum",
            "same_training_membership": True,
            "same_development_manifest": True,
            "source_official_test_opened": False,
        },
        "donor source record",
    )
    residual = initialization.get("haar_subtype_residual")
    if not isinstance(residual, Mapping):
        raise RuntimeError("Donor initialization lacks its residual audit")
    replay = residual.get("exact_replay")
    if not isinstance(replay, Mapping) or any(
        replay.get(key) is not True
        for key in (
            "exact_logits",
            "predictions_equal",
            "residual_weights_all_zero",
            "nonresidual_state_bitwise_equal",
        )
    ):
        raise RuntimeError("Donor zero-initialization did not exactly replay the primary")
    if residual.get("selection_spec_sha256") != SELECTION_SPEC_SHA256:
        raise RuntimeError("Donor initialization selection identity drifted")

    return {
        "primary_config": primary_config,
        "donor_config": donor_config,
        "data_report": primary_data,
        "primary_parameters": primary_parameters,
        "donor_parameters": donor_parameters,
        "primary_summary": primary_summary,
        "donor_summary": donor_summary,
        "donor_initialization": initialization,
    }


def load_and_validate_splits(
    primary_dir: Path,
    donor_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    primary_path = primary_dir / "split_indices.npz"
    donor_path = donor_dir / "split_indices.npz"
    primary_fingerprint = require_sha256(
        primary_path, SPLIT_FILE_SHA256, "primary split"
    )
    donor_fingerprint = require_sha256(donor_path, SPLIT_FILE_SHA256, "donor split")

    def load(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as split:
            if set(split.files) != {"train", "val"}:
                raise RuntimeError(f"Split must contain exactly train/val arrays: {path}")
            train = np.asarray(split["train"])
            val = np.asarray(split["val"])
        if train.dtype != np.int64 or val.dtype != np.int64:
            raise RuntimeError("Source split arrays must use exact int64 dtype")
        if train.shape != (TRAIN_SAMPLES,) or val.shape != (VALIDATION_SAMPLES,):
            raise RuntimeError("Source split arrays have unexpected sizes")
        if len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val):
            raise RuntimeError("Source split contains duplicate indices")
        return train, val

    primary_train, primary_val = load(primary_path)
    donor_train, donor_val = load(donor_path)
    if not np.array_equal(primary_train, donor_train) or not np.array_equal(
        primary_val, donor_val
    ):
        raise RuntimeError("Primary and donor split contents differ")
    if index_membership_sha256(primary_train) != TRAIN_MEMBERSHIP_SHA256:
        raise RuntimeError("Source training membership digest drifted")
    return primary_train, primary_val, {
        "sha256": SPLIT_FILE_SHA256,
        "primary_fingerprint": primary_fingerprint,
        "donor_fingerprint": donor_fingerprint,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "train_membership_sha256": TRAIN_MEMBERSHIP_SHA256,
        "primary_donor_bitwise_equal": True,
    }


def validate_selection_artifact(
    path: Path,
    donor_state: Mapping[str, torch.Tensor],
) -> Dict:
    fingerprint = require_sha256(path, SELECTION_FILE_SHA256, "selection artifact")
    selection = _read_json(path)
    claimed = selection.get("selection_spec_sha256")
    computed = canonical_spec_sha256(selection, "selection_spec_sha256")
    if claimed != SELECTION_SPEC_SHA256 or computed != SELECTION_SPEC_SHA256:
        raise RuntimeError(
            f"Selection specification digest drift: claimed={claimed} computed={computed}"
        )
    _require_fields(
        selection,
        {
            "algorithm_version": "invariant-haar-subtype-v1",
            "canonical_feature_count": 56,
            "selected_feature_count": RESIDUAL_PARAMETERS,
            "selected_indices": SELECTED_INDICES,
            "fit_images": TRAIN_SAMPLES,
            "fit_views": TRAIN_SAMPLES * 8,
            "selection_samples": 2 * TRAIN_PER_CLASS,
            "class_counts": {"axion": TRAIN_PER_CLASS, "cdm": TRAIN_PER_CLASS},
            "no_sub_selection_samples": 0,
            "validation_samples_used": 0,
            "official_test_samples_used": 0,
            "fixed_morphology_normalization_preserved": True,
            "fixed_haar_normalization_preserved": True,
            "source_checkpoint_sha256": PRIMARY_CHECKPOINT_SHA256,
            "train_membership_sha256": TRAIN_MEMBERSHIP_SHA256,
            "development_manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "class_names": MODEL_I_CLASSES,
        },
        "selection artifact",
    )
    if len(selection.get("selected_features", ())) != RESIDUAL_PARAMETERS:
        raise RuntimeError("Selection artifact does not define exactly 15 features")

    mappings = {
        "selected_indices": "haar_subtype_residual.selected_indices",
        "selected_center": "haar_subtype_residual.center",
        "selected_scale": "haar_subtype_residual.scale",
    }
    for json_key, state_key in mappings.items():
        tensor = donor_state.get(state_key)
        if tensor is None:
            raise RuntimeError(f"Donor is missing selection buffer: {state_key}")
        expected = torch.tensor(selection[json_key], dtype=tensor.dtype)
        if not torch.equal(tensor.detach().cpu(), expected):
            raise RuntimeError(f"Donor {state_key} differs bitwise from selection JSON")
        if tensor_bytes_sha256(tensor) != SELECTION_TENSOR_SHA256[state_key]:
            raise RuntimeError(f"Donor {state_key} raw tensor identity drifted")

    indices = donor_state["haar_subtype_residual.selected_indices"]
    center = donor_state["haar_subtype_residual.center"]
    scale = donor_state["haar_subtype_residual.scale"]
    if indices.dtype != torch.int64 or tuple(indices.shape) != (RESIDUAL_PARAMETERS,):
        raise RuntimeError("Donor selected indices have invalid dtype or shape")
    if len(torch.unique(indices)) != RESIDUAL_PARAMETERS or not bool(
        ((indices >= 0) & (indices < 56)).all()
    ):
        raise RuntimeError("Donor selected indices are not unique values in [0,56)")
    for name, tensor in (("center", center), ("scale", scale)):
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (RESIDUAL_PARAMETERS,):
            raise RuntimeError(f"Donor selection {name} has invalid dtype or shape")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Donor selection {name} is non-finite")
    if not bool((scale > 0).all()):
        raise RuntimeError("Donor selection scales must be positive")

    return {
        "file_fingerprint": fingerprint,
        "selection_spec_sha256": computed,
        "selected_indices": SELECTED_INDICES,
        "buffer_tensor_sha256": dict(SELECTION_TENSOR_SHA256),
        "validation_samples_used": 0,
        "official_test_samples_used": 0,
    }


def compose_derived_state(
    primary_state: Mapping[str, torch.Tensor],
    donor_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Copy only primary tensors and the four allowlisted donor tensors."""

    _validate_state_mapping(primary_state, "primary")
    _validate_state_mapping(donor_state, "donor")
    _validate_state_mapping(target_state, "target")
    primary_keys = set(primary_state)
    donor_keys = set(donor_state)
    target_keys = set(target_state)
    residual_keys = set(RESIDUAL_STATE_KEYS)
    if len(primary_keys) != PRIMARY_STATE_TENSORS:
        raise RuntimeError(
            f"Primary state tensor count drifted: {len(primary_keys)} != {PRIMARY_STATE_TENSORS}"
        )
    if len(donor_keys) != DERIVED_STATE_TENSORS:
        raise RuntimeError(
            f"Donor state tensor count drifted: {len(donor_keys)} != {DERIVED_STATE_TENSORS}"
        )
    if donor_keys != primary_keys | residual_keys:
        raise RuntimeError(
            "Donor keyset is not exactly primary plus the four residual tensors: "
            f"missing={sorted((primary_keys | residual_keys) - donor_keys)} "
            f"unexpected={sorted(donor_keys - (primary_keys | residual_keys))}"
        )
    if target_keys != donor_keys:
        raise RuntimeError(
            "Derived architecture keyset differs from the SHA-locked donor schema: "
            f"missing={sorted(donor_keys - target_keys)} "
            f"unexpected={sorted(target_keys - donor_keys)}"
        )
    for key in primary_keys:
        if primary_state[key].shape != donor_state[key].shape or primary_state[
            key
        ].dtype != donor_state[key].dtype:
            raise RuntimeError(f"Primary/donor shared tensor schema drifted: {key}")
    for key in FIXED_FEATURE_STATE_KEYS:
        if key not in primary_state or not torch.equal(
            primary_state[key].detach().cpu(), donor_state[key].detach().cpu()
        ):
            raise RuntimeError(
                f"Donor fixed Haar/morphology source buffer differs from primary: {key}"
            )

    composed: Dict[str, torch.Tensor] = {}
    for key, target in target_state.items():
        source = donor_state[key] if key in residual_keys else primary_state[key]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise RuntimeError(
                f"Source/target tensor schema drift for {key}: "
                f"source={tuple(source.shape)}/{source.dtype} "
                f"target={tuple(target.shape)}/{target.dtype}"
            )
        composed[key] = source.detach().cpu().clone()

    if any(
        not torch.equal(composed[key], primary_state[key].detach().cpu())
        for key in primary_keys
    ):
        raise RuntimeError("A composed shared tensor does not exactly equal the primary")
    if any(
        not torch.equal(composed[key], donor_state[key].detach().cpu())
        for key in residual_keys
    ):
        raise RuntimeError("A composed residual tensor does not exactly equal the donor")
    return composed


def build_derived_model(
    primary_state: Mapping[str, torch.Tensor],
    donor_state: Mapping[str, torch.Tensor],
) -> Tuple[D4OrbitClassifier, Dict[str, torch.Tensor], Dict, Dict]:
    weight = donor_state.get("haar_subtype_residual.weight")
    if weight is None or weight.dtype != torch.float32 or tuple(weight.shape) != (
        RESIDUAL_PARAMETERS,
    ):
        raise RuntimeError("Donor residual weight must be float32[15]")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("Donor residual weight is non-finite")
    if tensor_bytes_sha256(weight) != WEIGHT_TENSOR_SHA256:
        raise RuntimeError("Donor epoch-20 residual weight identity drifted")

    model = D4OrbitClassifier(
        num_classes=3,
        heads=4,
        reuploads=2,
        core="quantum",
        include_context=False,
        dropout=0.1,
        encoder_variant="deep-se-haar-morph",
        physics_variant="base",
        physics_summary="moments-morphology-haar",
        quantum_encoding="angle",
        observable_readout="pair",
        tied_mean_dispersion=False,
        haar_subtype_residual=True,
        haar_subtype_max_envelope=True,
        shared_late_refinement=False,
    )
    composed = compose_derived_state(primary_state, donor_state, model.state_dict())
    model.load_state_dict(composed, strict=True)
    loaded = model.state_dict()
    primary_equal = all(torch.equal(loaded[key].cpu(), value.cpu()) for key, value in primary_state.items())
    residual_equal = all(
        torch.equal(loaded[key].cpu(), donor_state[key].cpu())
        for key in RESIDUAL_STATE_KEYS
    )
    if not primary_equal or not residual_equal:
        raise RuntimeError("Selective derived-state loading failed its bitwise audit")

    report = model.parameter_report()
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (
        all_parameters != DERIVED_PARAMETERS
        or int(report.get("total", -1)) != DERIVED_PARAMETERS
        or int(report.get("haar_subtype_residual_trainable", -1)) != RESIDUAL_PARAMETERS
        or report.get("haar_subtype_max_envelope") is not True
    ):
        raise RuntimeError(f"Derived parameter budget drifted: {report}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("Derived evaluation model is not fully frozen")
    report.update(
        {
            "inference_total": all_parameters,
            "frozen_for_derivation_and_evaluation": all_parameters,
            "requires_grad_during_derivation_and_evaluation": 0,
            "optimized_during_derivation": 0,
            "primary_source_parameters": PRIMARY_PARAMETERS,
            "donor_residual_parameters": RESIDUAL_PARAMETERS,
            "max_envelope_parameters": 0,
            "derivation_id": DERIVATION_ID,
        }
    )
    return model, composed, report, {
        "primary_state_tensors_copied": len(primary_state),
        "donor_state_tensors_copied": len(RESIDUAL_STATE_KEYS),
        "donor_tensor_allowlist": list(RESIDUAL_STATE_KEYS),
        "primary_state_bitwise_equal_after_load": primary_equal,
        "donor_residual_bitwise_equal_after_load": residual_equal,
        "donor_nonresidual_tensors_copied": 0,
    }


def synthetic_envelope_audit() -> Dict:
    base_float32 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [-3.0, -2.0, -2.5],
            [4.0, -5.0, 3.0],
        ],
        dtype=torch.float32,
    )
    delta_float32 = torch.tensor([9.0, -7.0, -8.0, 0.5, 6.0, -20.0])
    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        base = base_float32.to(dtype)
        zeros = torch.zeros(len(base), dtype=dtype)
        replay = max_preserving_subtype_envelope(base, zeros)
        if not torch.equal(replay, base):
            raise RuntimeError(f"Zero-delta envelope does not replay bitwise in {dtype}")
        adjusted = max_preserving_subtype_envelope(
            base, delta_float32.to(dtype)
        )
        if not torch.equal(adjusted[:, :2].amax(1), base[:, :2].amax(1)):
            raise RuntimeError(f"Subtype parent envelope changed in {dtype}")
        base_no_sub = base.argmax(1) == 2
        adjusted_no_sub = adjusted.argmax(1) == 2
        if not torch.equal(base_no_sub, adjusted_no_sub):
            raise RuntimeError(f"no_sub argmax indicator changed in {dtype}")
        results[str(dtype).removeprefix("torch.")] = {
            "zero_delta_bitwise_replay": True,
            "parent_envelope_exact": True,
            "no_sub_argmax_indicator_exact": True,
            "samples": len(base),
        }
    return results


def validate_development_and_split(
    development_cache: Path,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[Dict, Dict]:
    manifest_fingerprint = require_sha256(
        development_cache / "manifest.csv",
        DEVELOPMENT_MANIFEST_SHA256,
        "development manifest",
    )
    images_fingerprint = require_sha256(
        development_cache / "images.npy",
        DEVELOPMENT_IMAGES_SHA256,
        "development image array",
    )
    labels_fingerprint = require_sha256(
        development_cache / "labels.npy",
        DEVELOPMENT_LABELS_SHA256,
        "development label array",
    )
    metadata_fingerprint = require_sha256(
        development_cache / "metadata.json",
        DEVELOPMENT_METADATA_SHA256,
        "development metadata",
    )
    development = validate_cache_structure(
        development_cache, expected_classes=MODEL_I_CLASSES
    )
    metadata = development["metadata"]
    _require_fields(
        metadata,
        {
            "complete": True,
            "classes": MODEL_I_CLASSES,
            "class_counts": MODEL_I_CLASS_COUNTS,
            "samples": MODEL_I_SAMPLES,
            "image_size": MODEL_I_IMAGE_SIZE,
            "dtype": "float16",
        },
        "development cache",
    )
    labels = development["labels"]
    canonical_train, canonical_val = canonical_model_i_split(labels)
    expected_train = hash_ranked_subset(
        canonical_train,
        labels,
        TRAIN_PER_CLASS,
        DEVELOPMENT_MANIFEST_SHA256,
    )
    if not np.array_equal(train_indices, expected_train):
        raise RuntimeError("Stored training split is not the canonical hash-v1 half subset")
    if not np.array_equal(val_indices, canonical_val):
        raise RuntimeError("Stored validation ordering is not the canonical seed-42 split")
    if np.intersect1d(train_indices, val_indices, assume_unique=True).size:
        raise RuntimeError("Stored half-training and validation indices overlap")
    if train_indices.min() < 0 or train_indices.max() >= len(labels):
        raise RuntimeError("Stored half-training indices are out of range")
    if val_indices.min() < 0 or val_indices.max() >= len(labels):
        raise RuntimeError("Stored validation indices are out of range")
    counts = {
        MODEL_I_CLASSES[label]: int((labels[train_indices] == label).sum())
        for label in range(len(MODEL_I_CLASSES))
    }
    if counts != {name: TRAIN_PER_CLASS for name in MODEL_I_CLASSES}:
        raise RuntimeError("Stored half-training split is not class balanced")
    return development, {
        "manifest_fingerprint": manifest_fingerprint,
        "images_fingerprint": images_fingerprint,
        "labels_fingerprint": labels_fingerprint,
        "metadata_fingerprint": metadata_fingerprint,
        "canonical_validation_order_exact": True,
        "hash_v1_half_training_membership_exact": True,
        "training_counts": counts,
        "validation_counts": {
            MODEL_I_CLASSES[label]: int((labels[val_indices] == label).sum())
            for label in range(len(MODEL_I_CLASSES))
        },
    }


def validate_primary_prediction_replay(
    primary_run: Path,
    val_indices: np.ndarray,
    development_labels: np.ndarray,
    actual_base_probabilities: np.ndarray,
    actual_base_metrics: Mapping,
) -> Dict:
    path = primary_run / "best_validation_predictions.npz"
    fingerprint = require_sha256(
        path, PRIMARY_PREDICTIONS_SHA256, "primary validation predictions"
    )
    expected = load_validation_predictions(
        path, val_indices, development_labels, MODEL_I_CLASSES
    )
    diagnostics = probability_replay_diagnostics(
        actual_base_probabilities, expected["probabilities"]
    )
    limits = {
        "max_probability_absolute_difference": MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
        "mean_probability_absolute_difference": (
            MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL
        ),
        "p99_probability_absolute_difference": (
            MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL
        ),
    }
    violations = {
        key: {"actual": float(diagnostics[key]), "maximum": maximum}
        for key, maximum in limits.items()
        if float(diagnostics[key]) > maximum
    }
    if violations or diagnostics["predicted_classes_exact"] is not True:
        raise RuntimeError(
            "Derived base path does not replay the SHA-locked primary predictions: "
            f"violations={violations} diagnostics={diagnostics}"
        )
    expected_metrics = classification_metrics(
        expected["labels"], expected["logits"], MODEL_I_CLASSES
    )
    scalar_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auc_ovr",
        "nll",
        "brier",
        "ece_15",
    )
    metric_drift = {
        key: abs(float(actual_base_metrics[key]) - float(expected_metrics[key]))
        for key in scalar_keys
    }
    if (
        max(metric_drift.values()) > MAX_VALIDATION_REPLAY_METRIC_ATOL
        or actual_base_metrics.get("confusion_matrix")
        != expected_metrics.get("confusion_matrix")
    ):
        raise RuntimeError(
            "Derived base path metrics do not replay the SHA-locked primary: "
            f"drift={metric_drift}"
        )
    return {
        "prediction_fingerprint": fingerprint,
        "diagnostics": diagnostics,
        "limits": limits,
        "maximum_metric_absolute_difference": max(metric_drift.values()),
        "metric_tolerance": MAX_VALIDATION_REPLAY_METRIC_ATOL,
        "confusion_matrix_exact": True,
    }


@torch.inference_mode()
def evaluate_validation_once(
    model: D4OrbitClassifier,
    development_cache: Path,
    val_indices: np.ndarray,
    development_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    workers: int,
    loader_seed: int,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    loader = make_loader(
        CachedNPYDataset(development_cache, val_indices),
        batch_size,
        False,
        workers,
        loader_seed,
    )
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()
    labels_all = []
    indices_all = []
    base_logits_all = []
    derived_logits_all = []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            derived_logits, auxiliary = model(images, return_aux=True)
        base_logits = auxiliary.get("haar_subtype_base_logits")
        if base_logits is None:
            raise RuntimeError("Derived model did not expose its pre-residual logits")
        if not bool(torch.isfinite(base_logits).all()) or not bool(
            torch.isfinite(derived_logits).all()
        ):
            raise RuntimeError("Validation produced non-finite logits")
        if not torch.equal(
            base_logits[:, :2].amax(dim=1),
            derived_logits[:, :2].amax(dim=1),
        ):
            raise RuntimeError("Validation subtype parent envelope changed")
        if not torch.equal(
            base_logits.argmax(dim=1) == 2,
            derived_logits.argmax(dim=1) == 2,
        ):
            raise RuntimeError("Validation no_sub argmax membership changed")
        labels_all.append(labels.numpy())
        indices_all.append(indices.numpy())
        base_logits_all.append(base_logits.float().cpu().numpy())
        derived_logits_all.append(derived_logits.float().cpu().numpy())

    labels = np.concatenate(labels_all).astype(np.int64, copy=False)
    indices = np.concatenate(indices_all).astype(np.int64, copy=False)
    base_logits = np.concatenate(base_logits_all).astype(np.float32, copy=False)
    logits = np.concatenate(derived_logits_all).astype(np.float32, copy=False)
    if not np.array_equal(indices, val_indices):
        raise RuntimeError("Validation loader order differs from the canonical split")
    if not np.array_equal(labels, development_labels[val_indices]):
        raise RuntimeError("Validation labels differ from the development cache")
    base_predictions = base_logits.argmax(axis=1).astype(np.int64)
    predictions = logits.argmax(axis=1).astype(np.int64)
    base_parent = base_logits[:, :2].max(axis=1)
    derived_parent = logits[:, :2].max(axis=1)
    if not np.array_equal(base_parent, derived_parent):
        raise RuntimeError("Saved validation subtype parent envelope changed")
    base_no_sub = base_predictions == 2
    derived_no_sub = predictions == 2
    if not np.array_equal(base_no_sub, derived_no_sub):
        raise RuntimeError("Saved validation no_sub membership changed")

    base_probabilities = softmax_numpy(base_logits)
    probabilities = softmax_numpy(logits)
    base_metrics = classification_metrics(labels, base_logits, MODEL_I_CLASSES)
    derived_metrics = classification_metrics(labels, logits, MODEL_I_CLASSES)
    audit = {
        "samples": int(len(labels)),
        "single_loader_pass": True,
        "parent_envelope_exact_all_samples": True,
        "no_sub_argmax_indicator_exact_all_samples": True,
        "base_predicted_no_sub": int(base_no_sub.sum()),
        "derived_predicted_no_sub": int(derived_no_sub.sum()),
        "predicted_no_sub_membership_changes": 0,
        "subtype_prediction_changes": int(
            ((base_predictions != predictions) & ~base_no_sub).sum()
        ),
        "base_no_sub_true_positives": int(((labels == 2) & base_no_sub).sum()),
        "derived_no_sub_true_positives": int(((labels == 2) & derived_no_sub).sum()),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "derivation_id": DERIVATION_ID,
        "classes": MODEL_I_CLASSES,
        "validation_samples": int(len(labels)),
        "primary_base": base_metrics,
        "derived": derived_metrics,
        "hierarchy_audit": audit,
        "official_test_evaluated": False,
    }, {
        "indices": indices,
        "labels": labels,
        "base_logits": base_logits,
        "logits": logits,
        "base_probabilities": base_probabilities.astype(np.float64),
        "probabilities": probabilities.astype(np.float64),
        "base_predictions": base_predictions,
        "predictions": predictions,
        "base_no_sub_indicator": base_no_sub,
        "no_sub_indicator": derived_no_sub,
    }


def _atomic_torch_save(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.building-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_output_destination(path: Path) -> Path:
    path = assert_no_symlink_components(path, "derived output")
    if os.path.lexists(path):
        raise RuntimeError(
            f"Refusing overwrite or resume: derived output already exists: {path}"
        )
    assert_canonical_directory(path.parent, "derived output parent")
    return path


def assert_path_sets_disjoint(
    output: Path, protected: Mapping[str, Path]
) -> None:
    """Reject output/source aliases and every ancestor/descendant overlap."""

    items = list(protected.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise RuntimeError(
                    f"Protected paths overlap: {left_name}={left} "
                    f"and {right_name}={right}"
                )
    for name, path in items:
        if output == path or path in output.parents or output in path.parents:
            raise RuntimeError(
                f"Derived output must be disjoint from {name}: "
                f"output={output} protected={path}"
            )


def _runtime_report(device: torch.device) -> Dict:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "autocast": "cuda-bfloat16" if device.type == "cuda" else "disabled",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if device.type == "cuda":
        report.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return report


def derive_and_evaluate(args: argparse.Namespace) -> None:
    configure_training_runtime(PRIMARY_SEED)
    primary_dir = assert_canonical_directory(args.primary_run, "primary run")
    donor_dir = assert_canonical_directory(args.donor_run, "donor run")
    development_cache = assert_canonical_directory(
        args.development_cache, "development cache"
    )
    output_dir = validate_output_destination(Path(args.output_dir))
    if primary_dir == donor_dir:
        raise RuntimeError("Primary and donor run directories must differ")
    assert_path_sets_disjoint(
        output_dir,
        {
            "primary run": primary_dir,
            "donor run": donor_dir,
            "development cache": development_cache,
        },
    )

    source_before = {
        "primary": fingerprint_named_files(primary_dir, PRIMARY_ARTIFACTS),
        "donor": fingerprint_named_files(donor_dir, DONOR_ARTIFACTS),
    }
    metadata = validate_run_metadata(primary_dir, donor_dir)
    primary_checkpoint, primary_state, primary_checkpoint_fingerprint = (
        load_sha_locked_checkpoint(
            primary_dir / "best.pt",
            PRIMARY_CHECKPOINT_SHA256,
            PRIMARY_EPOCH,
            "primary best checkpoint",
        )
    )
    donor_checkpoint, donor_state, donor_checkpoint_fingerprint = (
        load_sha_locked_checkpoint(
            donor_dir / "last.pt",
            DONOR_CHECKPOINT_SHA256,
            DONOR_EPOCH,
            "donor epoch-20 checkpoint",
        )
    )
    train_indices, val_indices, split_audit = load_and_validate_splits(
        primary_dir, donor_dir
    )
    selection_audit = validate_selection_artifact(
        donor_dir / "haar_subtype_selection.json", donor_state
    )
    model, composed_state, parameter_report, selective_load_audit = (
        build_derived_model(primary_state, donor_state)
    )
    synthetic_audit = synthetic_envelope_audit()

    cache_before = fingerprint_named_files(development_cache, CACHE_ARTIFACTS)
    development, development_audit = validate_development_and_split(
        development_cache, train_indices, val_indices
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Derived evaluator supports only cpu or cuda")
    metrics, predictions = evaluate_validation_once(
        model,
        development_cache,
        val_indices,
        development["labels"],
        device,
        int(args.batch_size),
        int(args.workers),
        int(args.loader_seed),
    )
    primary_replay = validate_primary_prediction_replay(
        primary_dir,
        val_indices,
        development["labels"],
        predictions["base_probabilities"],
        metrics["primary_base"],
    )
    metrics["primary_saved_prediction_replay"] = primary_replay

    cache_after = fingerprint_named_files(development_cache, CACHE_ARTIFACTS)
    assert_fingerprints_equal(
        cache_after, cache_before, "development cache during derived evaluation"
    )
    source_after = {
        "primary": fingerprint_named_files(primary_dir, PRIMARY_ARTIFACTS),
        "donor": fingerprint_named_files(donor_dir, DONOR_ARTIFACTS),
    }
    assert_fingerprints_equal(
        source_after["primary"], source_before["primary"], "primary source run"
    )
    assert_fingerprints_equal(
        source_after["donor"], source_before["donor"], "donor source run"
    )

    output_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
    fsync_directory(output_dir.parent)
    config = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "derived-development-validation-candidate",
        "derivation_id": DERIVATION_ID,
        "development_validation_only": True,
        "analysis_status": "exploratory-derived-candidate",
        "training_performed": False,
        "evaluate_test": False,
        "official_test_supported": False,
        "num_classes": 3,
        "class_names": MODEL_I_CLASSES,
        "core": "quantum",
        "heads": 4,
        "reuploads": 2,
        "include_context": False,
        "dropout": 0.1,
        "encoder_variant": "deep-se-haar-morph",
        "physics_variant": "base",
        "physics_summary": "moments-morphology-haar",
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "tied_mean_dispersion": False,
        "haar_subtype_residual": True,
        "haar_subtype_max_envelope": True,
        "shared_late_refinement": False,
        "transform_id": TRANSFORM_ID,
        "parameters": DERIVED_PARAMETERS,
        "primary_checkpoint_sha256": PRIMARY_CHECKPOINT_SHA256,
        "donor_checkpoint_sha256": DONOR_CHECKPOINT_SHA256,
        "donor_epoch": DONOR_EPOCH,
    }
    checkpoint_record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "derived-no-training-checkpoint",
        "derivation_id": DERIVATION_ID,
        "analysis_status": {
            "label": "exploratory-derived-candidate",
            "independent_prospective_result": False,
            "reason": (
                "The hierarchy-preserving transform was designed after inspecting "
                "earlier development-validation diagnostics."
            ),
        },
        "training_performed": False,
        "primary_checkpoint_sha256": PRIMARY_CHECKPOINT_SHA256,
        "primary_epoch": PRIMARY_EPOCH,
        "donor_checkpoint_sha256": DONOR_CHECKPOINT_SHA256,
        "donor_epoch": DONOR_EPOCH,
        "donor_tensor_allowlist": list(RESIDUAL_STATE_KEYS),
        "transform_id": TRANSFORM_ID,
        "official_test_opened": False,
    }
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "parameter_report.json", parameter_report)
    _atomic_torch_save(
        output_dir / "derived.pt",
        {"model": composed_state, "derivation": checkpoint_record},
    )
    atomic_npz(output_dir / "validation_predictions.npz", **predictions)
    atomic_json(output_dir / "metrics.json", metrics)

    code_files = {
        "evaluate_derived_validation.py": file_fingerprint(Path(__file__)),
        "evaluate_locked.py": file_fingerprint(
            Path(__file__).with_name("evaluate_locked.py")
        ),
        "model.py": file_fingerprint(Path(__file__).with_name("model.py")),
        "data.py": file_fingerprint(Path(__file__).with_name("data.py")),
        "metrics.py": file_fingerprint(Path(__file__).with_name("metrics.py")),
        "quantum.py": file_fingerprint(Path(__file__).with_name("quantum.py")),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "derivation_id": DERIVATION_ID,
        "analysis_status": {
            "label": "exploratory-derived-candidate",
            "independent_prospective_result": False,
            "reason": (
                "The hierarchy-preserving transform was designed after inspecting "
                "earlier development-validation diagnostics."
            ),
        },
        "transform": {
            "id": TRANSFORM_ID,
            "formula": (
                "u=a+d; v=c-d; m=max(a,c); shift=m-max(u,v); "
                "assign the winning subtype exactly m and clamp the loser <=m; n'=n"
            ),
            "trainable_parameters_added": 0,
            "guarantee": (
                "max(axion',cdm') equals max(axion,cdm) exactly in the inference "
                "dtype; therefore the no_sub argmax indicator is unchanged"
            ),
        },
        "training": {
            "performed": False,
            "optimizer_created": False,
            "fitting_samples": 0,
            "validation_samples_used_for_fitting": 0,
            "validation_samples_used_for_transform_design": VALIDATION_SAMPLES,
            "development_validation_samples_evaluated_this_run": VALIDATION_SAMPLES,
            "official_test_samples_used": 0,
        },
        "primary": {
            "run_dir": str(primary_dir),
            "checkpoint": "best.pt",
            "checkpoint_fingerprint": primary_checkpoint_fingerprint,
            "checkpoint_sha256": PRIMARY_CHECKPOINT_SHA256,
            "epoch": int(primary_checkpoint["epoch"]),
            "parameters": PRIMARY_PARAMETERS,
            "source_artifacts": source_before["primary"],
        },
        "donor": {
            "run_dir": str(donor_dir),
            "checkpoint": "last.pt",
            "checkpoint_fingerprint": donor_checkpoint_fingerprint,
            "checkpoint_sha256": DONOR_CHECKPOINT_SHA256,
            "epoch": int(donor_checkpoint["epoch"]),
            "weight_tensor_sha256": WEIGHT_TENSOR_SHA256,
            "source_artifacts": source_before["donor"],
        },
        "selection": selection_audit,
        "split": split_audit,
        "development": {
            "cache_dir": str(development_cache),
            "cache_artifacts": cache_before,
            "audit": development_audit,
        },
        "selective_loading": selective_load_audit,
        "synthetic_envelope_audit": synthetic_audit,
        "validation_evaluation": {
            "performed_once": True,
            "samples": VALIDATION_SAMPLES,
            "hierarchy_audit": metrics["hierarchy_audit"],
            "primary_saved_prediction_replay": primary_replay,
        },
        "official_test": {
            "argument_exposed": False,
            "opened": False,
            "evaluated": False,
        },
        "runtime": _runtime_report(device),
        "code": code_files,
        "derived_checkpoint_fingerprint": file_fingerprint(
            output_dir / "derived.pt"
        ),
        "source_metadata_checks": {
            "primary_parameter_total": metadata["primary_parameters"]["total"],
            "donor_parameter_total": metadata["donor_parameters"]["total"],
            "source_data_reports_identical": True,
            "source_official_test_locked": True,
        },
    }
    atomic_json(output_dir / "provenance.json", provenance)

    artifact_names = (
        "config.json",
        "parameter_report.json",
        "derived.pt",
        "validation_predictions.npz",
        "metrics.json",
        "provenance.json",
    )
    artifact_fingerprints = fingerprint_named_files(output_dir, artifact_names)
    atomic_json(
        output_dir / "artifact_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "derivation_id": DERIVATION_ID,
            "artifacts": artifact_fingerprints,
        },
        exclusive=True,
    )
    fsync_file(output_dir / "artifact_seal.json")
    fsync_directory(output_dir)
    print(json.dumps(metrics, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", required=True)
    parser.add_argument("--donor-run", required=True)
    parser.add_argument("--development-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--loader-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "primary_run",
        "donor_run",
        "development_cache",
        "output_dir",
    ):
        value = Path(getattr(args, name)).expanduser()
        if not value.is_absolute():
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path")
        setattr(args, name, value)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.loader_seed < 0:
        raise ValueError("--loader-seed cannot be negative")
    derive_and_evaluate(args)


if __name__ == "__main__":
    main()
