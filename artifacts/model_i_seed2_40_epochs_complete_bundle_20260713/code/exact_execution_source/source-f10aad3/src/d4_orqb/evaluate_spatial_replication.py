"""Audited fixed-seed evaluation for the paired spatial-stat replication.

The protocol evaluates the predeclared seeds 0, 1, and 2 for the 132-parameter
quantum core and its exactly parameter-matched classical core.  The endpoint is
always ``last.pt`` at epoch 40; validation is performed once at that endpoint,
so ``best.pt`` must contain a bitwise-identical model state.

The primary evaluation set is the part of the canonical Model-I development
training split that was not used by the fixed half-training subset.  This set
is training-disjoint for all six models, but it has previously been exposed to
the research process by another experiment and is therefore not described as
an unseen or investigator-blind holdout.  Canonical validation is used only as
an exact endpoint-replay and consistency assessment.

There is deliberately no official-test argument or inference path.  Scientific
gate failure is a valid result; protocol or provenance drift raises before an
output directory is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from .data import CachedNPYDataset, index_membership_sha256, make_loader
from .evaluate_locked import (
    assert_canonical_directory,
    assert_fingerprints_equal,
    assert_no_symlink_components,
    atomic_json,
    atomic_npz,
    canonical_model_i_split,
    file_fingerprint,
    fsync_directory,
    mcnemar_exact,
    metrics_from_probabilities,
    softmax_numpy,
    validate_cache_structure,
)
from .model import D4OrbitClassifier


SCHEMA_VERSION = 1
PROTOCOL_ID = "paired-spatial-stat-fixed-three-seed-development-v1"
CLASS_NAMES = ("axion", "cdm", "no_sub")
SEEDS = (0, 1, 2)
CORES = ("quantum", "classical")
RUN_KEYS = tuple((seed, core) for seed in SEEDS for core in CORES)
IMAGE_SIZE = 96
EPOCHS = 40
BATCH_SIZE = 256
WORKERS = 4
LOADER_SEED = 20260812
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20260812
NONINFERIORITY_MARGIN = 0.005
SUPERIORITY_PRACTICAL_FLOOR = 0.002

DEVELOPMENT_COUNTS = (28_897, 29_772, 28_856)
DEVELOPMENT_SAMPLES = 87_525
FULL_TRAIN_SAMPLES = 70_021
HALF_TRAIN_SAMPLES = 35_001
COMPLEMENT_SAMPLES = 35_020
COMPLEMENT_COUNTS = (11_451, 12_151, 11_418)
VALIDATION_SAMPLES = 17_504
VALIDATION_COUNTS = (5_779, 5_954, 5_771)

DEVELOPMENT_CACHE_SHA256 = {
    "images.npy": "c3c639584e0a9e2d6ba369e3fb41ba0451b2170a362ec419fa1182e55d5ce070",
    "labels.npy": "3a12100d1df155738b57255e6625ba1287c2d74c5bab53bce2cd0597afe89b17",
    "metadata.json": "9f36faff5fc3300b97512b1c29439b2e993888318a31835938749e10fbf12379",
    "manifest.csv": "c04a3c62afebe3f660ffaad4333b6632471a91c6f5f239f84e68b4b94c330025",
}
FULL_TRAIN_MEMBERSHIP_SHA256 = (
    "b14cfdc30d9f9803843dd2471aeb0948241f1115cbb73e2db522c6071c860de1"
)
HALF_TRAIN_MEMBERSHIP_SHA256 = (
    "571d23ced25095cf0cfb57216654f9b7be289b0589a95489a5a815a866aaee71"
)
COMPLEMENT_MEMBERSHIP_SHA256 = (
    "16be51c4068f124d4408a224d5df8aba6a6baa909c113cbd99e6ad818c6aa831"
)
VALIDATION_MEMBERSHIP_SHA256 = (
    "454935a294c5bb0f7c66c5b5c61072e469575b1ad68fda9fa3efb057db97ec52"
)

RUN_ARTIFACTS = (
    "config.json",
    "data_report.json",
    "history.json",
    "parameter_report.json",
    "split_indices.npz",
    "last.pt",
    "best.pt",
    "last_validation_predictions.npz",
    "best_validation_predictions.npz",
    "summary.json",
    "initialization_report.json",
    "training_rng_report.json",
    "stochastic_trace.json",
)
CACHE_ARTIFACTS = tuple(DEVELOPMENT_CACHE_SHA256)
SCALAR_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auc_ovr",
    "nll",
    "brier",
    "ece_15",
)


def run_name(seed: int, core: str) -> str:
    return f"seed-{int(seed)}/{core}"


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSON artifact: {path}") from error
    return value


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def guard_forbidden_path(path: str | Path, context: str) -> Path:
    """Reject the official-test name, lexical aliases, and symlink aliases."""

    path = Path(path).expanduser()
    if any(part.casefold() == "model_i_test" for part in path.parts):
        raise RuntimeError(f"{context} must not reference the official Model-I test")
    return assert_no_symlink_components(path, context)


def refuse_existing_output(path: str | Path) -> Path:
    path = guard_forbidden_path(path, "output")
    if os.path.lexists(path):
        raise RuntimeError(f"Refusing to replace an existing output path: {path}")
    assert_canonical_directory(path.parent, "output parent")
    return path


def fingerprint_files(root: Path, names: Sequence[str]) -> Dict[str, Dict]:
    root = assert_canonical_directory(root, "artifact root")
    return {name: file_fingerprint(root / name) for name in names}


def fingerprint_cache(cache_dir: Path) -> Dict[str, Dict]:
    actual = fingerprint_files(cache_dir, CACHE_ARTIFACTS)
    drift = {
        name: {
            "actual": actual[name]["sha256"],
            "expected": expected,
        }
        for name, expected in DEVELOPMENT_CACHE_SHA256.items()
        if actual[name]["sha256"] != expected
    }
    if drift:
        raise RuntimeError(f"Development cache identity drifted: {drift}")
    return actual


def state_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash state semantics independently of checkpoint container bytes."""

    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Checkpoint model state is empty or invalid")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise RuntimeError("Checkpoint state must map string keys to tensors")
        cpu = value.detach().cpu().contiguous()
        if (cpu.is_floating_point() or cpu.is_complex()) and not bool(
            torch.isfinite(cpu).all()
        ):
            raise RuntimeError(f"Checkpoint contains a non-finite tensor: {key}")
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(cpu.dtype).encode("ascii")
        digest.update(len(key_bytes).to_bytes(4, "little"))
        digest.update(key_bytes)
        digest.update(len(dtype_bytes).to_bytes(2, "little"))
        digest.update(dtype_bytes)
        digest.update(len(cpu.shape).to_bytes(2, "little"))
        for dimension in cpu.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=True))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def states_bitwise_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        tuple(left[key].shape) == tuple(right[key].shape)
        and left[key].dtype == right[key].dtype
        and torch.equal(left[key].detach().cpu(), right[key].detach().cpu())
        for key in left
    )


def _require_exact(mapping: Mapping, expected: Mapping, context: str) -> None:
    drift = {
        key: {"actual": mapping.get(key), "expected": value}
        for key, value in expected.items()
        if mapping.get(key) != value
    }
    if drift:
        raise RuntimeError(f"{context} protocol drift: {drift}")


def _require_floats(mapping: Mapping, expected: Mapping[str, float], context: str) -> None:
    drift = {}
    for key, value in expected.items():
        actual = mapping.get(key)
        if actual is None or not math.isclose(
            float(actual), float(value), rel_tol=0.0, abs_tol=1e-12
        ):
            drift[key] = {"actual": actual, "expected": value}
    if drift:
        raise RuntimeError(f"{context} floating-point protocol drift: {drift}")


def validate_run_config(config: Mapping, run: Path, seed: int, core: str) -> Dict:
    if not isinstance(config, Mapping):
        raise RuntimeError("Run config must be a JSON object")
    exact = {
        "image_size": IMAGE_SIZE,
        "encoder_variant": "micro-stat",
        "physics_variant": "base",
        "physics_summary": "moments",
        "include_context": False,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "heads": 4,
        "reuploads": 3,
        "core": core,
        "epochs": EPOCHS,
        "patience": 41,
        "validation_interval": EPOCHS,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "io_workers": 8,
        "seed": seed,
        "training_rng_seed": 20_000 + seed,
        "split_seed": 42,
        "max_train_per_class": 11_667,
        "max_val_per_class": None,
        "train_subset_protocol": "hash-v1",
        "evaluate_test": False,
        "deterministic": True,
        "fixed_final_validation_only": True,
        "save_last_validation_predictions": True,
        "save_stochastic_trace": True,
        "init_backbone_checkpoint": None,
        "init_compatible_backbone_checkpoint": None,
        "reinitialize_core_after_init": False,
        "tied_mean_dispersion": False,
        "haar_subtype_residual": False,
        "haar_subtype_max_envelope": False,
        "freeze_haar_subtype_residual_at_zero": False,
        "freeze_base_for_haar_subtype_residual": False,
        "shared_late_refinement": False,
        "r2_entanglers": False,
        "freeze_r2_entanglers_at_zero": False,
        "equatorial_readout": False,
        "freeze_equatorial_readout_at_zero": False,
        "meridional_readout": False,
        "freeze_meridional_readout_at_zero": False,
        "subtype_specialist": False,
        "oof_teacher_fold_index": None,
        "distillation_teacher_checkpoint": None,
        "oof_distillation_artifact": None,
        "oof_distillation_report": None,
        "hierarchical_loss_weight": 0.0,
        "branch_loss_weight": 0.0,
        "max_translation_pixels": 0,
        "translation_probability": 1.0,
        "psf_blur_probability": 0.0,
        "read_noise_std": 0.0,
        "subtype_mixup_probability": 0.0,
    }
    _require_exact(config, exact, f"{run_name(seed, core)} config")
    _require_floats(
        config,
        {
            "encoder_learning_rate": 5e-4,
            "learning_rate": 3e-3,
            "core_learning_rate": 5e-3,
            "weight_decay": 1e-4,
            "label_smoothing": 0.02,
            "dropout": 0.10,
            "photon_noise_probability": 0.5,
            "photon_count_min": 2048.0,
            "photon_count_max": 8192.0,
            "val_fraction": 0.20,
            "distillation_weight": 0.0,
        },
        f"{run_name(seed, core)} config",
    )
    if not config.get("init_full_checkpoint"):
        raise RuntimeError("Paired spatial run lacks its full initializer")
    if not config.get("paired_spatial_init_report"):
        raise RuntimeError("Paired spatial run lacks its initializer report")
    configured_output = guard_forbidden_path(config.get("output_dir", ""), "run output")
    if configured_output != run:
        raise RuntimeError(
            f"Configured output does not equal run directory: {configured_output} != {run}"
        )
    for context, key in (
        ("development root", "development_root"),
        ("unused test root", "test_root"),
        ("initializer", "init_full_checkpoint"),
        ("initializer report", "paired_spatial_init_report"),
    ):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw:
            raise RuntimeError(f"Run config lacks {key}")
        guard_forbidden_path(raw, context)
    return dict(config)


def validate_parameter_report(report: Mapping, seed: int, core: str) -> None:
    expected = {
        "total": 122_573,
        "physics": 0,
        "encoder": 119_682,
        "orbit_projection": 1_672,
        "core": 132,
        "head_and_context": 1_087,
        "quantum": 132 if core == "quantum" else 0,
        "parallel_classical": 0 if core == "quantum" else 132,
        "mixture_trainable": 0,
        "encoder_variant": "micro-stat",
        "encoder_output_dim": 192,
        "physics_variant": "base",
        "physics_summary": "moments",
        "physics_summary_dim": 16,
        "core_architecture": core,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "tied_mean_dispersion": False,
        "haar_subtype_residual": False,
        "shared_late_refinement": False,
        "r2_entanglers": False,
        "equatorial_readout": False,
        "meridional_readout": False,
    }
    _require_exact(report, expected, f"{run_name(seed, core)} parameter report")
    if int(report.get("total", -1)) > 122_610:
        raise RuntimeError("Spatial replication exceeds the inference parameter ceiling")


def validate_paired_initializer_binding(
    initialization: Mapping,
    config: Mapping,
    seed: int,
    core: str,
    initializer_fingerprint: Mapping,
    report_fingerprint: Mapping,
) -> Dict:
    """Revalidate the semantic seed/core binding recorded during training.

    Quantum and classical initializers intentionally share an identical state
    schema, including ``core.params``.  Strict state loading therefore cannot
    detect a cross-wired arm; the report, checkpoint metadata, and canonical
    component-state digests are all required here.
    """

    from .spatial_paired_init import PROTOCOL as INITIALIZER_PROTOCOL
    from .spatial_paired_init import state_sha256 as paired_state_sha256

    initializer_path = Path(config["init_full_checkpoint"])
    report_path = Path(config["paired_spatial_init_report"])
    report = _read_json(report_path)
    if not isinstance(report, Mapping):
        raise RuntimeError("Paired initializer report must be a JSON object")
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id") != INITIALIZER_PROTOCOL
        or report.get("seeds") != list(SEEDS)
        or report.get("official_test_opened") is not False
        or report.get("official_test_reference_accepted") is not False
    ):
        raise RuntimeError("Paired initializer report protocol identity drifted")
    payload_sha256 = report.get("report_payload_sha256")
    unhashed = dict(report)
    unhashed.pop("report_payload_sha256", None)
    canonical = (
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if payload_sha256 != hashlib.sha256(canonical).hexdigest():
        raise RuntimeError("Paired initializer report payload digest drifted")
    try:
        seed_report = report["per_seed"][str(seed)]
        arm = seed_report["arms"][core]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Paired initializer report lacks the requested arm") from error
    expected_path = (report_path.parent / arm["checkpoint_path"]).resolve()
    if initializer_path.resolve() != expected_path:
        raise RuntimeError(
            f"Paired initializer is cross-wired: {initializer_path} != {expected_path}"
        )
    checkpoint = _torch_load(initializer_path)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise RuntimeError("Paired initializer checkpoint schema is invalid")
    state = checkpoint["model"]
    full_sha256 = paired_state_sha256(state)
    core_state = {
        key: value for key, value in state.items() if key.startswith("core.")
    }
    noncore_state = {
        key: value for key, value in state.items() if not key.startswith("core.")
    }
    core_sha256 = paired_state_sha256(core_state)
    noncore_sha256 = paired_state_sha256(noncore_state)
    expected_arm = {
        "core_name": core,
        "checkpoint_sha256": initializer_fingerprint["sha256"],
        "full_state_sha256": full_sha256,
        "core_state_sha256": core_sha256,
        "native_core_state_sha256": core_sha256,
        "noncore_state_sha256": noncore_sha256,
    }
    _require_exact(arm, expected_arm, f"seed-{seed}/{core} initializer arm")
    _require_exact(
        checkpoint,
        {
            "schema_version": 1,
            "protocol_id": INITIALIZER_PROTOCOL,
            "epoch": 0,
            "seed": seed,
            "core_name": core,
            "full_state_sha256": full_sha256,
            "core_state_sha256": core_sha256,
            "native_core_state_sha256": core_sha256,
            "noncore_state_sha256": noncore_sha256,
            "common_noncore_state_sha256": noncore_sha256,
        },
        f"seed-{seed}/{core} initializer checkpoint",
    )
    if seed_report.get("common_noncore_state_sha256") != noncore_sha256:
        raise RuntimeError("Initializer pair common non-core digest drifted")
    parameters = checkpoint.get("parameters")
    if not isinstance(parameters, Mapping):
        raise RuntimeError("Initializer checkpoint lacks its parameter report")
    validate_parameter_report(parameters, seed, core)
    binding_payload = {
        "protocol_id": INITIALIZER_PROTOCOL,
        "seed": seed,
        "backbone_sha256": seed_report["backbone_fingerprint"]["sha256"],
        "common_noncore_state_sha256": noncore_sha256,
        "quantum": {
            key: seed_report["arms"]["quantum"][key]
            for key in (
                "core_name",
                "checkpoint_sha256",
                "full_state_sha256",
                "core_state_sha256",
                "noncore_state_sha256",
            )
        },
        "classical": {
            key: seed_report["arms"]["classical"][key]
            for key in (
                "core_name",
                "checkpoint_sha256",
                "full_state_sha256",
                "core_state_sha256",
                "noncore_state_sha256",
            )
        },
    }
    expected_pair_binding = hashlib.sha256(
        (
            json.dumps(
                binding_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if seed_report.get("pair_binding_sha256") != expected_pair_binding:
        raise RuntimeError("Initializer pair-binding payload digest drifted")
    expected_binding = {
        "protocol_id": INITIALIZER_PROTOCOL,
        "report": str(report_path),
        "report_sha256": report_fingerprint["sha256"],
        "report_payload_sha256": payload_sha256,
        "seed": seed,
        "core": core,
        "checkpoint_sha256": initializer_fingerprint["sha256"],
        "full_state_sha256": full_sha256,
        "core_state_sha256": core_sha256,
        "common_noncore_state_sha256": noncore_sha256,
        "pair_binding_sha256": expected_pair_binding,
        "cross_wire_rejected": True,
    }
    binding = initialization.get("paired_spatial_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("Training initialization report lacks paired spatial binding")
    _require_exact(binding, expected_binding, f"{run_name(seed, core)} paired binding")
    return {
        **expected_binding,
        "initializer_report_seeds": list(report["seeds"]),
        "semantic_cross_wire_check_replayed": True,
    }


def validate_data_report(report: Mapping, seed: int, core: str) -> None:
    expected = {
        "class_names": list(CLASS_NAMES),
        "train_size": HALF_TRAIN_SAMPLES,
        "validation_size": VALIDATION_SAMPLES,
        "train_counts": dict(zip(CLASS_NAMES, (11_667, 11_667, 11_667))),
        "validation_counts": dict(zip(CLASS_NAMES, VALIDATION_COUNTS)),
        "train_subset_protocol": "hash-v1",
        "train_membership_sha256": HALF_TRAIN_MEMBERSHIP_SHA256,
        "development_manifest_sha256": DEVELOPMENT_CACHE_SHA256["manifest.csv"],
        "official_test_cache_opened": False,
        "official_test_locked_during_selection": True,
    }
    _require_exact(report, expected, f"{run_name(seed, core)} data report")
    if "test" in report:
        raise RuntimeError(f"{run_name(seed, core)} data report contains test material")
    development = report.get("development")
    if not isinstance(development, Mapping):
        raise RuntimeError("Data report lacks development metadata")
    _require_exact(
        development,
        {
            "complete": True,
            "samples": DEVELOPMENT_SAMPLES,
            "image_size": IMAGE_SIZE,
            "classes": list(CLASS_NAMES),
            "class_counts": dict(zip(CLASS_NAMES, DEVELOPMENT_COUNTS)),
            "dtype": "float16",
        },
        f"{run_name(seed, core)} development metadata",
    )


def validate_history(history, seed: int, core: str) -> Dict:
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise RuntimeError(f"{run_name(seed, core)} history is not exactly 40 epochs")
    observed_epochs = [record.get("epoch") for record in history if isinstance(record, Mapping)]
    if observed_epochs != list(range(1, EPOCHS + 1)):
        raise RuntimeError(f"{run_name(seed, core)} history epoch sequence drifted")
    validation_epochs = [
        int(record["epoch"])
        for record in history
        if record.get("validation") is not None
    ]
    if validation_epochs != [EPOCHS]:
        raise RuntimeError(
            f"{run_name(seed, core)} must have exactly one validation at epoch 40"
        )
    return dict(history[-1])


def validate_split_artifact(
    path: Path, labels: np.ndarray, seed: int, core: str
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as split:
        if set(split.files) != {"train", "val"}:
            raise RuntimeError(f"{run_name(seed, core)} split artifact schema drifted")
        train_raw = np.asarray(split["train"])
        val_raw = np.asarray(split["val"])
    if train_raw.dtype.kind not in "iu" or val_raw.dtype.kind not in "iu":
        raise RuntimeError("Split arrays must be integer arrays")
    train = train_raw.astype(np.int64, copy=False)
    val = val_raw.astype(np.int64, copy=False)
    if train.shape != (HALF_TRAIN_SAMPLES,) or val.shape != (VALIDATION_SAMPLES,):
        raise RuntimeError("Spatial split sizes drifted")
    if len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val):
        raise RuntimeError("Spatial split contains duplicate indices")
    if (
        train.min() < 0
        or val.min() < 0
        or train.max() >= len(labels)
        or val.max() >= len(labels)
        or np.intersect1d(train, val, assume_unique=True).size
    ):
        raise RuntimeError("Spatial train/validation split is invalid")
    if index_membership_sha256(train) != HALF_TRAIN_MEMBERSHIP_SHA256:
        raise RuntimeError("Spatial half-training membership drifted")
    if index_membership_sha256(val) != VALIDATION_MEMBERSHIP_SHA256:
        raise RuntimeError("Spatial validation membership drifted")
    counts = tuple(int((labels[train] == label).sum()) for label in range(3))
    val_counts = tuple(int((labels[val] == label).sum()) for label in range(3))
    if counts != (11_667, 11_667, 11_667) or val_counts != VALIDATION_COUNTS:
        raise RuntimeError("Spatial split class counts drifted")
    return train, val


def load_endpoint_predictions(
    path: Path,
    expected_indices: np.ndarray,
    labels: np.ndarray,
    *,
    require_epoch: bool,
) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        required = {"indices", "labels", "logits", "probabilities"}
        if require_epoch:
            required.add("epoch")
        if not required.issubset(values.files):
            raise RuntimeError(
                f"Endpoint predictions lack fields: {sorted(required - set(values.files))}"
            )
        indices = np.asarray(values["indices"])
        observed_labels = np.asarray(values["labels"])
        logits = np.asarray(values["logits"])
        probabilities = np.asarray(values["probabilities"])
        epoch = int(np.asarray(values["epoch"]).item()) if require_epoch else None
    if indices.dtype.kind not in "iu" or observed_labels.dtype.kind not in "iu":
        raise RuntimeError("Endpoint indices and labels must be integer arrays")
    indices = indices.astype(np.int64, copy=False)
    observed_labels = observed_labels.astype(np.int64, copy=False)
    if not np.array_equal(indices, np.asarray(expected_indices, dtype=np.int64)):
        raise RuntimeError("Endpoint prediction membership/order drifted")
    if not np.array_equal(observed_labels, labels[expected_indices]):
        raise RuntimeError("Endpoint prediction labels disagree with the cache")
    if (
        logits.dtype != np.float32
        or probabilities.dtype != np.float32
        or logits.shape != (len(indices), len(CLASS_NAMES))
        or probabilities.shape != logits.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("Endpoint prediction arrays have invalid dtype/shape/values")
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected_probabilities = np.exp(shifted)
    expected_probabilities /= expected_probabilities.sum(axis=1, keepdims=True)
    if not np.array_equal(probabilities, expected_probabilities):
        raise RuntimeError("Stored endpoint probabilities are not exact softmax(logits)")
    if require_epoch and epoch != EPOCHS:
        raise RuntimeError("Last validation predictions are not from epoch 40")
    return {
        "indices": indices,
        "labels": observed_labels,
        "logits": logits,
        "probabilities": probabilities,
        "epoch": np.asarray(EPOCHS, dtype=np.int64),
    }


def build_strict_model(
    config: Mapping, state: Mapping[str, torch.Tensor], seed: int, core: str
) -> D4OrbitClassifier:
    model = D4OrbitClassifier(
        num_classes=len(CLASS_NAMES),
        heads=int(config["heads"]),
        reuploads=int(config["reuploads"]),
        core=str(config["core"]),
        include_context=bool(config["include_context"]),
        dropout=float(config["dropout"]),
        encoder_variant=str(config["encoder_variant"]),
        physics_variant=str(config["physics_variant"]),
        physics_summary=str(config["physics_summary"]),
        quantum_encoding=str(config["quantum_encoding"]),
        observable_readout=str(config["observable_readout"]),
        tied_mean_dispersion=bool(config["tied_mean_dispersion"]),
        haar_subtype_residual=bool(config["haar_subtype_residual"]),
        shared_late_refinement=bool(config["shared_late_refinement"]),
        haar_subtype_max_envelope=bool(config["haar_subtype_max_envelope"]),
        r2_entanglers=bool(config["r2_entanglers"]),
        equatorial_readout=bool(config["equatorial_readout"]),
        meridional_readout=bool(config["meridional_readout"]),
    )
    state_tensor_sha256(state)
    model.load_state_dict(state, strict=True)
    report = model.parameter_report()
    validate_parameter_report(report, seed, core)
    if sum(parameter.numel() for parameter in model.parameters()) != 122_573:
        raise RuntimeError("Strictly reconstructed spatial model size drifted")
    model.requires_grad_(False)
    model.eval()
    return model


def validate_stochastic_trace(
    report: Mapping, trace, seed: int, core: str
) -> Tuple[Dict, list]:
    _require_exact(
        report,
        {
            "schema_version": 1,
            "training_rng_seed": 20_000 + seed,
            "loader_and_subset_seed": seed,
            "reset_point": "immediately before epoch 1 after optimizer/scheduler setup",
            "stochastic_trace_enabled": True,
        },
        f"{run_name(seed, core)} training RNG report",
    )
    digests = report.get("initial_state_sha256")
    if not isinstance(digests, Mapping) or not digests:
        raise RuntimeError("Training RNG report lacks state digests")
    if not isinstance(trace, list) or len(trace) != EPOCHS:
        raise RuntimeError("Stochastic trace must contain exactly 40 epochs")
    expected_batches = math.ceil(HALF_TRAIN_SAMPLES / BATCH_SIZE)
    for epoch, record in enumerate(trace, start=1):
        if not isinstance(record, Mapping):
            raise RuntimeError("Stochastic trace record is invalid")
        if (
            record.get("epoch") != epoch
            or record.get("sample_count") != HALF_TRAIN_SAMPLES
            or record.get("batch_count") != expected_batches
        ):
            raise RuntimeError("Stochastic trace epoch/sample/batch contract drifted")
        ordered = record.get("ordered_sample_indices_sha256")
        batches = record.get("batches")
        if (
            not isinstance(ordered, str)
            or len(ordered) != 64
            or not isinstance(batches, list)
            or len(batches) != expected_batches
        ):
            raise RuntimeError("Stochastic trace digest or batch list drifted")
        if sum(int(batch.get("batch_size", -1)) for batch in batches) != HALF_TRAIN_SAMPLES:
            raise RuntimeError("Stochastic trace batch sizes do not cover training membership")
        for batch_index, batch in enumerate(batches):
            if batch.get("batch_index") != batch_index:
                raise RuntimeError("Stochastic trace batch ordering drifted")
            for key in ("sample_indices_sha256", "pre_augmentation_rng_sha256"):
                value = batch.get(key)
                if key == "pre_augmentation_rng_sha256":
                    if not isinstance(value, Mapping) or not value:
                        raise RuntimeError("Stochastic trace RNG digest mapping is invalid")
                elif not isinstance(value, str) or len(value) != 64:
                    raise RuntimeError("Stochastic trace sample digest is invalid")
    return dict(report), list(trace)


def _load_checkpoint(path: Path, seed: int, core: str, endpoint: str) -> Dict:
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model",
        "epoch",
        "record",
    }:
        raise RuntimeError(
            f"{run_name(seed, core)} {endpoint} checkpoint schema drifted"
        )
    if int(checkpoint.get("epoch", -1)) != EPOCHS:
        raise RuntimeError(
            f"{run_name(seed, core)} {endpoint} is not the epoch-40 endpoint"
        )
    record = checkpoint.get("record")
    if (
        not isinstance(record, Mapping)
        or record.get("epoch") != EPOCHS
        or record.get("validation") is None
    ):
        raise RuntimeError(
            f"{run_name(seed, core)} {endpoint} checkpoint record is invalid"
        )
    state_tensor_sha256(checkpoint["model"])
    return checkpoint


def validate_run(
    run: Path,
    labels: np.ndarray,
    seed: int,
    core: str,
    expected_fingerprints: Mapping[str, Mapping],
) -> Dict:
    """Validate one fixed endpoint and reconstruct its frozen CPU model."""

    run = assert_canonical_directory(
        guard_forbidden_path(run, f"{run_name(seed, core)} run"),
        f"{run_name(seed, core)} run",
    )
    config = validate_run_config(_read_json(run / "config.json"), run, seed, core)
    data_report = _read_json(run / "data_report.json")
    parameter_report = _read_json(run / "parameter_report.json")
    summary = _read_json(run / "summary.json")
    initialization = _read_json(run / "initialization_report.json")
    validate_data_report(data_report, seed, core)
    validate_parameter_report(parameter_report, seed, core)
    if not isinstance(summary, Mapping):
        raise RuntimeError("Summary must be a JSON object")
    if (
        summary.get("best_epoch") != EPOCHS
        or summary.get("official_test_evaluated") is not False
        or "test" in summary
    ):
        raise RuntimeError(f"{run_name(seed, core)} summary violates the endpoint lock")
    if summary.get("parameters") != parameter_report:
        raise RuntimeError(f"{run_name(seed, core)} summary parameter report drifted")

    history = _read_json(run / "history.json")
    final_history_record = validate_history(history, seed, core)
    train_indices, val_indices = validate_split_artifact(
        run / "split_indices.npz", labels, seed, core
    )
    last = _load_checkpoint(run / "last.pt", seed, core, "last")
    best = _load_checkpoint(run / "best.pt", seed, core, "best")
    if not states_bitwise_equal(last["model"], best["model"]):
        raise RuntimeError(
            f"{run_name(seed, core)} last/best model states are not bitwise equal"
        )
    if last["record"].get("validation") != final_history_record.get("validation"):
        raise RuntimeError(
            f"{run_name(seed, core)} endpoint record/history validation differs"
        )
    last_predictions = load_endpoint_predictions(
        run / "last_validation_predictions.npz",
        val_indices,
        labels,
        require_epoch=True,
    )
    best_predictions = load_endpoint_predictions(
        run / "best_validation_predictions.npz",
        val_indices,
        labels,
        require_epoch=False,
    )
    for key in ("indices", "labels", "logits", "probabilities"):
        if not np.array_equal(last_predictions[key], best_predictions[key]):
            raise RuntimeError(
                f"{run_name(seed, core)} last/best endpoint predictions differ for {key}"
            )

    if not isinstance(initialization, Mapping):
        raise RuntimeError("Initialization report must be a JSON object")
    initializer = assert_no_symlink_components(
        Path(config["init_full_checkpoint"]), "initializer checkpoint"
    )
    initializer_report = assert_no_symlink_components(
        Path(config["paired_spatial_init_report"]), "paired initializer report"
    )
    initializer_fingerprint = file_fingerprint(initializer)
    initializer_report_fingerprint = file_fingerprint(initializer_report)
    if (
        initialization.get("checkpoint") != str(initializer)
        or initialization.get("checkpoint_sha256")
        != initializer_fingerprint["sha256"]
        or initialization.get("loaded_tensors") != 93
        or initialization.get("missing_target_tensors") != 0
        or initialization.get("adapted_tensors") != []
        or initialization.get("method") != "full-model-exact-or-zero-extension"
    ):
        raise RuntimeError(
            f"{run_name(seed, core)} full-initializer provenance drifted"
        )
    paired_binding = validate_paired_initializer_binding(
        initialization,
        config,
        seed,
        core,
        initializer_fingerprint,
        initializer_report_fingerprint,
    )

    rng_report, stochastic_trace = validate_stochastic_trace(
        _read_json(run / "training_rng_report.json"),
        _read_json(run / "stochastic_trace.json"),
        seed,
        core,
    )
    model = build_strict_model(config, last["model"], seed, core)

    after = fingerprint_files(run, RUN_ARTIFACTS)
    assert_fingerprints_equal(
        after, expected_fingerprints, f"{run_name(seed, core)} validation"
    )
    return {
        "run": str(run),
        "seed": seed,
        "core": core,
        "config": config,
        "data_report": data_report,
        "parameter_report": parameter_report,
        "summary": summary,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "model": model,
        "endpoint_predictions": last_predictions,
        "endpoint_state_sha256": state_tensor_sha256(last["model"]),
        "last_checkpoint_sha256": expected_fingerprints["last.pt"]["sha256"],
        "best_checkpoint_sha256": expected_fingerprints["best.pt"]["sha256"],
        "initializer": {
            "path": str(initializer),
            **initializer_fingerprint,
        },
        "initializer_report": {
            "path": str(initializer_report),
            **initializer_report_fingerprint,
        },
        "paired_initializer_binding": paired_binding,
        "training_rng_report": rng_report,
        "stochastic_trace": stochastic_trace,
    }


def fingerprint_all_runs(runs_root: Path) -> Dict[str, Dict[str, Dict]]:
    result = {}
    for seed, core in RUN_KEYS:
        name = run_name(seed, core)
        result[name] = fingerprint_files(runs_root / name, RUN_ARTIFACTS)
    return result


def load_six_runs(
    runs_root: Path,
    labels: np.ndarray,
    expected_fingerprints: Mapping[str, Mapping],
) -> Dict[Tuple[int, str], Dict]:
    runs_root = assert_canonical_directory(
        guard_forbidden_path(runs_root, "runs root"), "runs root"
    )
    loaded = {}
    for seed, core in RUN_KEYS:
        name = run_name(seed, core)
        loaded[(seed, core)] = validate_run(
            runs_root / name,
            labels,
            seed,
            core,
            expected_fingerprints[name],
        )
    reference_train = loaded[(0, "quantum")]["train_indices"]
    reference_val = loaded[(0, "quantum")]["val_indices"]
    for key, record in loaded.items():
        if not np.array_equal(record["train_indices"], reference_train):
            raise RuntimeError(f"{run_name(*key)} training split order differs")
        if not np.array_equal(record["val_indices"], reference_val):
            raise RuntimeError(f"{run_name(*key)} validation split order differs")
    for seed in SEEDS:
        quantum = loaded[(seed, "quantum")]
        classical = loaded[(seed, "classical")]
        if quantum["training_rng_report"] != classical["training_rng_report"]:
            raise RuntimeError(f"Seed {seed} paired training RNG reports differ")
        if quantum["stochastic_trace"] != classical["stochastic_trace"]:
            raise RuntimeError(f"Seed {seed} paired stochastic traces differ")
        q_binding = quantum["paired_initializer_binding"]
        c_binding = classical["paired_initializer_binding"]
        if (
            q_binding["pair_binding_sha256"] != c_binding["pair_binding_sha256"]
            or q_binding["common_noncore_state_sha256"]
            != c_binding["common_noncore_state_sha256"]
            or q_binding["core_state_sha256"] == c_binding["core_state_sha256"]
            or q_binding["checkpoint_sha256"] == c_binding["checkpoint_sha256"]
        ):
            raise RuntimeError(f"Seed {seed} paired initializer semantics drifted")
    shared_reports = {
        (
            record["paired_initializer_binding"]["report"],
            record["paired_initializer_binding"]["report_sha256"],
            record["paired_initializer_binding"]["report_payload_sha256"],
        )
        for record in loaded.values()
    }
    if len(shared_reports) != 1:
        raise RuntimeError("Six runs do not bind to one shared initializer report")
    return loaded


def derive_development_partitions(
    labels: np.ndarray,
    frozen_half: np.ndarray,
    stored_validation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Derive and SHA-lock full-train-minus-half complement membership."""

    labels = np.asarray(labels, dtype=np.int64)
    half = np.asarray(frozen_half, dtype=np.int64)
    stored_validation = np.asarray(stored_validation, dtype=np.int64)
    canonical_train, canonical_validation = canonical_model_i_split(labels)
    full_sorted = np.sort(canonical_train)
    half_sorted = np.sort(half)
    complement = np.setdiff1d(full_sorted, half_sorted, assume_unique=True)
    memberships = {
        "canonical_full_train": index_membership_sha256(full_sorted),
        "frozen_half_train": index_membership_sha256(half_sorted),
        "development_complement": index_membership_sha256(complement),
        "canonical_validation": index_membership_sha256(canonical_validation),
    }
    expected = {
        "canonical_full_train": FULL_TRAIN_MEMBERSHIP_SHA256,
        "frozen_half_train": HALF_TRAIN_MEMBERSHIP_SHA256,
        "development_complement": COMPLEMENT_MEMBERSHIP_SHA256,
        "canonical_validation": VALIDATION_MEMBERSHIP_SHA256,
    }
    if memberships != expected:
        raise RuntimeError(
            f"Development partition membership drifted: actual={memberships} expected={expected}"
        )
    if (
        len(full_sorted) != FULL_TRAIN_SAMPLES
        or len(half_sorted) != HALF_TRAIN_SAMPLES
        or len(complement) != COMPLEMENT_SAMPLES
        or len(canonical_validation) != VALIDATION_SAMPLES
    ):
        raise RuntimeError("Development partition sizes drifted")
    if not np.array_equal(stored_validation, canonical_validation):
        raise RuntimeError("Stored validation order differs from the canonical split")
    if len(np.unique(half_sorted)) != len(half_sorted) or not np.isin(
        half_sorted, full_sorted, assume_unique=True
    ).all():
        raise RuntimeError("Frozen half is not a unique subset of canonical training")
    if np.intersect1d(complement, half_sorted, assume_unique=True).size:
        raise RuntimeError("Development complement overlaps the frozen half")
    if np.intersect1d(complement, canonical_validation, assume_unique=True).size:
        raise RuntimeError("Development complement overlaps canonical validation")
    if not np.array_equal(
        np.sort(np.concatenate((half_sorted, complement))), full_sorted
    ):
        raise RuntimeError("Frozen half and complement do not partition canonical training")
    full_counts = tuple(int((labels[full_sorted] == value).sum()) for value in range(3))
    complement_counts = tuple(
        int((labels[complement] == value).sum()) for value in range(3)
    )
    validation_counts = tuple(
        int((labels[canonical_validation] == value).sum()) for value in range(3)
    )
    if complement_counts != COMPLEMENT_COUNTS or validation_counts != VALIDATION_COUNTS:
        raise RuntimeError("Complement or canonical-validation class counts drifted")
    return complement, canonical_validation, {
        "membership_sha256": memberships,
        "samples": {
            "canonical_full_train": len(full_sorted),
            "frozen_half_train": len(half_sorted),
            "development_complement": len(complement),
            "canonical_validation": len(canonical_validation),
        },
        "class_counts": {
            "canonical_full_train": dict(zip(CLASS_NAMES, full_counts)),
            "development_complement": dict(zip(CLASS_NAMES, complement_counts)),
            "canonical_validation": dict(zip(CLASS_NAMES, validation_counts)),
        },
        "half_plus_complement_exactly_partitions_canonical_train": True,
        "complement_disjoint_from_all_six_training_sets": True,
        "complement_disjoint_from_canonical_validation": True,
    }


def clustered_fixed_seed_bootstrap(
    labels: np.ndarray,
    correctness_outcomes: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 512,
) -> Dict:
    """Bootstrap examples as six-outcome clusters within true-class strata.

    Columns are ``q0,c0,q1,c1,q2,c2``.  Each resampled example therefore
    carries every fixed-seed/arm correctness outcome.  Seeds are never
    resampled, and the resulting interval is not an inference about a random
    population of seeds.

    Correctness has at most 64 unique six-bit patterns.  Drawing multinomial
    counts over those empirical patterns is exactly equivalent to drawing the
    original examples with replacement and is substantially cheaper for the
    predeclared 100,000 replicates.
    """

    labels = np.asarray(labels, dtype=np.int64)
    outcomes = np.asarray(correctness_outcomes)
    if outcomes.shape != (len(labels), 2 * len(SEEDS)):
        raise ValueError("Correctness outcomes must have shape [samples, 6]")
    if outcomes.dtype != np.bool_:
        if not np.isin(outcomes, (0, 1)).all():
            raise ValueError("Correctness outcomes must be binary")
        outcomes = outcomes.astype(bool)
    if samples <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap samples and chunk size must be positive")
    strata = [np.flatnonzero(labels == label) for label in range(len(CLASS_NAMES))]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("Cluster bootstrap encountered an empty class stratum")

    integer_outcomes = outcomes.astype(np.int8)
    delta = integer_outcomes[:, 0::2] - integer_outcomes[:, 1::2]
    per_seed_point = delta.mean(axis=0, dtype=np.float64)
    fixed_seed_mean_point = float(per_seed_point.mean())
    per_seed_estimates = np.empty((samples, len(SEEDS)), dtype=np.float64)
    fixed_seed_mean_estimates = np.empty(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pattern_audit = []
    patterns_by_class = []
    for label, indices in enumerate(strata):
        patterns, counts = np.unique(
            integer_outcomes[indices], axis=0, return_counts=True
        )
        pattern_delta = patterns[:, 0::2] - patterns[:, 1::2]
        patterns_by_class.append((len(indices), counts, pattern_delta))
        pattern_audit.append(
            {
                "class": CLASS_NAMES[label],
                "samples": int(len(indices)),
                "unique_six_outcome_patterns": int(len(patterns)),
                "pattern_counts_sha256": hashlib.sha256(
                    counts.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
        )
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        totals = np.zeros((stop - start, len(SEEDS)), dtype=np.float64)
        for stratum_size, counts, pattern_delta in patterns_by_class:
            draws = rng.multinomial(
                stratum_size,
                counts.astype(np.float64) / stratum_size,
                size=stop - start,
            )
            totals += draws @ pattern_delta
        estimates = totals / len(labels)
        per_seed_estimates[start:stop] = estimates
        fixed_seed_mean_estimates[start:stop] = estimates.mean(axis=1)

    per_seed = {}
    for column, seed_value in enumerate(SEEDS):
        per_seed[str(seed_value)] = {
            "difference": float(per_seed_point[column]),
            "ci95_low": float(np.quantile(per_seed_estimates[:, column], 0.025)),
            "ci95_high": float(np.quantile(per_seed_estimates[:, column], 0.975)),
        }
    return {
        "estimand": (
            "mean over the three predeclared fixed seeds of "
            "accuracy(quantum)-accuracy(classical)"
        ),
        "resampling_unit": (
            "one development example carrying q0,c0,q1,c1,q2,c2 correctness; "
            "resampled once within true class"
        ),
        "implementation": "exact empirical six-pattern multinomial bootstrap",
        "fixed_seed_mean_difference": fixed_seed_mean_point,
        "fixed_seed_mean_ci95_low": float(
            np.quantile(fixed_seed_mean_estimates, 0.025)
        ),
        "fixed_seed_mean_ci95_high": float(
            np.quantile(fixed_seed_mean_estimates, 0.975)
        ),
        "per_seed": per_seed,
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "class_pattern_audit": pattern_audit,
        "seed_resampling": False,
        "random_seed_population_claim": False,
        "scope": (
            "conditional on the three fixed seeds; sampling uncertainty over "
            "development examples only"
        ),
    }


def accuracy_decision_gates(
    bootstrap: Mapping,
    per_seed: Mapping[str, Mapping],
    seed_mean_recall_delta_by_class: Mapping[str, float],
) -> Dict:
    point = float(bootstrap["fixed_seed_mean_difference"])
    low = float(bootstrap["fixed_seed_mean_ci95_low"])
    per_seed_effects = {
        int(seed): float(record["difference"])
        for seed, record in bootstrap["per_seed"].items()
    }
    noninferiority_conditions = {
        "fixed_seed_cluster_bootstrap_ci95_low_strictly_above_minus_0_005": (
            low > -NONINFERIORITY_MARGIN
        ),
        "every_fixed_seed_accuracy_delta_strictly_above_minus_0_005": all(
            value > -NONINFERIORITY_MARGIN for value in per_seed_effects.values()
        ),
        "every_class_seed_mean_recall_delta_strictly_above_minus_0_005": all(
            float(value) > -NONINFERIORITY_MARGIN
            for value in seed_mean_recall_delta_by_class.values()
        ),
    }
    one_sided_mcnemar = {
        int(seed): float(record["mcnemar_exact_one_sided_quantum_greater_p"])
        for seed, record in per_seed.items()
    }
    superiority_conditions = {
        "noninferiority_gate_passed": all(noninferiority_conditions.values()),
        "fixed_seed_mean_difference_at_least_0_002": (
            point >= SUPERIORITY_PRACTICAL_FLOOR
        ),
        "fixed_seed_cluster_bootstrap_ci95_low_strictly_above_zero": low > 0.0,
        "all_three_paired_seed_point_effects_strictly_positive": all(
            value > 0.0 for value in per_seed_effects.values()
        ),
        "all_three_one_sided_exact_mcnemar_p_strictly_below_0_05": all(
            value < 0.05 for value in one_sided_mcnemar.values()
        ),
    }
    return {
        "noninferiority": {
            "margin": NONINFERIORITY_MARGIN,
            "conditions": noninferiority_conditions,
            "passed": all(noninferiority_conditions.values()),
        },
        "strict_superiority": {
            "practical_floor": SUPERIORITY_PRACTICAL_FLOOR,
            "conditions": superiority_conditions,
            "per_seed_one_sided_exact_mcnemar_p": one_sided_mcnemar,
            "passed": all(superiority_conditions.values()),
        },
        "hierarchy": (
            "Strict superiority is considered only after noninferiority and "
            "requires all three fixed paired effects and all three predeclared "
            "directional exact McNemar checks."
        ),
    }


def binomial_half_upper_tail(successes: int, trials: int) -> float:
    """Exact Binomial(trials, 1/2) upper tail with stable log accumulation."""

    successes = int(successes)
    trials = int(trials)
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial upper-tail arguments")
    if successes == 0:
        return 1.0

    # By symmetry P[X >= successes] = P[X <= trials - successes].
    cutoff = trials - successes

    def lower_tail(k: int) -> float:
        if k < 0:
            return 0.0
        if k >= trials:
            return 1.0
        if 2 * k >= trials:
            return max(0.0, 1.0 - lower_tail(trials - k - 1))
        log_probability_at_k = (
            math.lgamma(trials + 1)
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            - trials * math.log(2.0)
        )
        relative_term = 1.0
        relative_sum = 1.0
        for index in range(k, 0, -1):
            relative_term *= index / (trials - index + 1)
            relative_sum += relative_term
        log_tail = log_probability_at_k + math.log(relative_sum)
        return math.exp(log_tail) if log_tail > -746.0 else 0.0

    return min(1.0, max(0.0, lower_tail(cutoff)))


def analyze_dataset(
    labels: np.ndarray,
    logits_by_run: Mapping[Tuple[int, str], np.ndarray],
    *,
    dataset_name: str,
    role: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Dict:
    labels = np.asarray(labels, dtype=np.int64)
    if set(logits_by_run) != set(RUN_KEYS):
        raise ValueError("Dataset analysis requires exactly six fixed run outputs")
    probabilities = {}
    metrics = {}
    outcomes = np.empty((len(labels), 2 * len(SEEDS)), dtype=bool)
    for seed_index, seed in enumerate(SEEDS):
        for core_index, core in enumerate(CORES):
            key = (seed, core)
            logits = np.asarray(logits_by_run[key])
            if (
                logits.dtype != np.float32
                or logits.shape != (len(labels), len(CLASS_NAMES))
                or not np.isfinite(logits).all()
            ):
                raise ValueError(f"Invalid logits for {run_name(seed, core)}")
            probability = softmax_numpy(logits)
            probabilities[key] = probability
            metrics[key] = metrics_from_probabilities(
                labels, probability, CLASS_NAMES
            )
            outcomes[:, 2 * seed_index + core_index] = (
                probability.argmax(axis=1) == labels
            )

    per_seed = {}
    for seed_index, seed in enumerate(SEEDS):
        quantum_metrics = metrics[(seed, "quantum")]
        classical_metrics = metrics[(seed, "classical")]
        quantum_correct = outcomes[:, 2 * seed_index]
        classical_correct = outcomes[:, 2 * seed_index + 1]
        per_class = {}
        for label, name in enumerate(CLASS_NAMES):
            mask = labels == label
            q_value = int((quantum_correct & mask).sum())
            c_value = int((classical_correct & mask).sum())
            per_class[name] = {
                "quantum_correct": q_value,
                "classical_correct": c_value,
                "quantum_minus_classical_correct": q_value - c_value,
            }
        mcnemar = mcnemar_exact(
            labels,
            probabilities[(seed, "quantum")],
            probabilities[(seed, "classical")],
        )
        one_sided_quantum_greater = binomial_half_upper_tail(
            int(mcnemar["a_correct_b_wrong"]),
            int(mcnemar["discordant"]),
        )
        per_seed[str(seed)] = {
            "quantum": quantum_metrics,
            "classical": classical_metrics,
            "metric_tradeoffs_quantum_minus_classical": {
                metric: float(quantum_metrics[metric])
                - float(classical_metrics[metric])
                for metric in SCALAR_METRICS
            },
            "quantum_correct": int(quantum_correct.sum()),
            "classical_correct": int(classical_correct.sum()),
            "quantum_minus_classical_correct": int(
                quantum_correct.sum() - classical_correct.sum()
            ),
            "per_class_correct": per_class,
            "mcnemar_exact_two_sided": mcnemar,
            "mcnemar_exact_one_sided_quantum_greater_p": (
                one_sided_quantum_greater
            ),
        }

    fixed_mean_quantum = {
        metric: float(np.mean([metrics[(seed, "quantum")][metric] for seed in SEEDS]))
        for metric in SCALAR_METRICS
    }
    fixed_mean_classical = {
        metric: float(
            np.mean([metrics[(seed, "classical")][metric] for seed in SEEDS])
        )
        for metric in SCALAR_METRICS
    }
    bootstrap = clustered_fixed_seed_bootstrap(
        labels,
        outcomes,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    seed_mean_recall_delta_by_class = {
        name: float(
            np.mean(
                [
                    metrics[(seed, "quantum")]["per_class"][name]["recall"]
                    - metrics[(seed, "classical")]["per_class"][name]["recall"]
                    for seed in SEEDS
                ]
            )
        )
        for name in CLASS_NAMES
    }
    gates = accuracy_decision_gates(
        bootstrap, per_seed, seed_mean_recall_delta_by_class
    )
    return {
        "dataset": dataset_name,
        "role": role,
        "samples": int(len(labels)),
        "class_counts": {
            name: int((labels == label).sum())
            for label, name in enumerate(CLASS_NAMES)
        },
        "per_seed": per_seed,
        "fixed_three_seed_mean_metrics": {
            "quantum": fixed_mean_quantum,
            "classical": fixed_mean_classical,
            "quantum_minus_classical": {
                metric: fixed_mean_quantum[metric] - fixed_mean_classical[metric]
                for metric in SCALAR_METRICS
            },
            "interpretation": {
                "higher_is_better": [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "macro_auc_ovr",
                ],
                "lower_is_better": ["nll", "brier", "ece_15"],
            },
        },
        "clustered_accuracy_bootstrap": bootstrap,
        "seed_mean_recall_delta_by_class_quantum_minus_classical": (
            seed_mean_recall_delta_by_class
        ),
        "accuracy_gates": gates,
        "inference_scope": (
            "fixed seeds 0,1,2 only; examples are resampled, seeds are not; "
            "no random-seed population claim"
        ),
        "prediction_ensemble_used": False,
    }


def canonical_consistency_gates(canonical_analysis: Mapping) -> Dict:
    bootstrap = canonical_analysis["clustered_accuracy_bootstrap"]
    mean_effect = float(bootstrap["fixed_seed_mean_difference"])
    per_seed = {
        int(seed): float(record["difference"])
        for seed, record in bootstrap["per_seed"].items()
    }
    noninferiority_conditions = {
        "canonical_fixed_seed_mean_strictly_above_minus_0_005": (
            mean_effect > -NONINFERIORITY_MARGIN
        ),
        "canonical_minimum_seed_effect_strictly_above_minus_0_005": (
            min(per_seed.values()) > -NONINFERIORITY_MARGIN
        ),
    }
    superiority_conditions = {
        "canonical_fixed_seed_mean_strictly_positive": mean_effect > 0.0,
        "every_canonical_seed_effect_nonnegative": min(per_seed.values()) >= 0.0,
    }
    return {
        "role": (
            "directional consistency only; no canonical-validation confidence "
            "interval or p-value enters the primary decision"
        ),
        "noninferiority_consistency": {
            "conditions": noninferiority_conditions,
            "passed": all(noninferiority_conditions.values()),
        },
        "superiority_consistency": {
            "conditions": superiority_conditions,
            "passed": all(superiority_conditions.values()),
        },
    }


def configure_deterministic_runtime(seed: int = LOADER_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def evaluate_six_models(
    loaded: Mapping[Tuple[int, str], Mapping],
    cache_dir: Path,
    indices: np.ndarray,
    expected_labels: np.ndarray,
) -> Dict:
    """Evaluate six frozen endpoints without any test-set code path."""

    if set(loaded) != set(RUN_KEYS):
        raise RuntimeError("Exactly six fixed spatial endpoints are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Spatial replication evaluation requires exactly one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Spatial replication evaluation requires CUDA bfloat16 support")
    indices = np.asarray(indices, dtype=np.int64)
    expected_labels = np.asarray(expected_labels, dtype=np.int64)
    device = torch.device("cuda")
    loader = make_loader(
        CachedNPYDataset(cache_dir, indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        workers=WORKERS,
        seed=LOADER_SEED,
    )
    logits_by_run = {}
    observed_indices_reference = None
    observed_labels_reference = None
    for seed, core in RUN_KEYS:
        model = loaded[(seed, core)]["model"]
        model.to(device, memory_format=torch.channels_last).eval()
        logits_parts = []
        indices_parts = []
        labels_parts = []
        for images, labels, batch_indices in loader:
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(
                    f"Non-finite logits from {run_name(seed, core)}"
                )
            logits_parts.append(logits.float().cpu().numpy())
            indices_parts.append(batch_indices.numpy())
            labels_parts.append(labels.numpy())
        torch.cuda.synchronize()
        observed_indices = np.concatenate(indices_parts).astype(np.int64, copy=False)
        observed_labels = np.concatenate(labels_parts).astype(np.int64, copy=False)
        logits = np.concatenate(logits_parts).astype(np.float32, copy=False)
        if not np.array_equal(observed_indices, indices):
            raise RuntimeError("Evaluation loader changed fixed membership/order")
        if not np.array_equal(observed_labels, expected_labels[indices]):
            raise RuntimeError("Evaluation labels disagree with locked cache")
        if logits.shape != (len(indices), len(CLASS_NAMES)):
            raise RuntimeError("Evaluation logits have an invalid shape")
        if observed_indices_reference is None:
            observed_indices_reference = observed_indices
            observed_labels_reference = observed_labels
        elif not np.array_equal(observed_indices_reference, observed_indices) or not np.array_equal(
            observed_labels_reference, observed_labels
        ):
            raise RuntimeError("Six endpoint evaluation orders differ")
        logits_by_run[(seed, core)] = logits
        model.cpu()
        torch.cuda.empty_cache()
    return {
        "indices": observed_indices_reference,
        "labels": observed_labels_reference,
        "logits_by_run": logits_by_run,
    }


def validate_exact_canonical_replay(
    loaded: Mapping[Tuple[int, str], Mapping], replayed: Mapping
) -> Dict[str, Dict]:
    report = {}
    for seed, core in RUN_KEYS:
        expected = loaded[(seed, core)]["endpoint_predictions"]
        actual_logits = replayed["logits_by_run"][(seed, core)]
        if not np.array_equal(replayed["indices"], expected["indices"]):
            raise RuntimeError(f"{run_name(seed, core)} replay membership differs")
        if not np.array_equal(replayed["labels"], expected["labels"]):
            raise RuntimeError(f"{run_name(seed, core)} replay labels differ")
        if not np.array_equal(actual_logits, expected["logits"]):
            difference = np.abs(
                actual_logits.astype(np.float64) - expected["logits"].astype(np.float64)
            )
            raise RuntimeError(
                f"{run_name(seed, core)} canonical replay is not bitwise exact: "
                f"max={float(difference.max())} mean={float(difference.mean())}"
            )
        shifted = actual_logits - actual_logits.max(axis=1, keepdims=True)
        actual_probabilities = np.exp(shifted)
        actual_probabilities /= actual_probabilities.sum(axis=1, keepdims=True)
        if not np.array_equal(actual_probabilities, expected["probabilities"]):
            raise RuntimeError(
                f"{run_name(seed, core)} canonical probability replay is not exact"
            )
        report[run_name(seed, core)] = {
            "indices_exact": True,
            "labels_exact": True,
            "logits_bitwise_exact": True,
            "probabilities_bitwise_exact": True,
            "logits_sha256": hashlib.sha256(actual_logits.tobytes()).hexdigest(),
            "samples": int(len(actual_logits)),
        }
    return report


def prediction_arrays(result: Mapping) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "indices": np.asarray(result["indices"], dtype=np.int64),
        "labels": np.asarray(result["labels"], dtype=np.int64),
    }
    for seed, core in RUN_KEYS:
        prefix = f"seed_{seed}_{core}"
        logits = np.asarray(result["logits_by_run"][(seed, core)], dtype=np.float32)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        arrays[f"{prefix}_logits"] = logits
        arrays[f"{prefix}_probabilities"] = probabilities.astype(
            np.float32, copy=False
        )
    return arrays


def fingerprint_external_inputs(loaded: Mapping[Tuple[int, str], Mapping]) -> Dict:
    result = {}
    for seed, core in RUN_KEYS:
        record = loaded[(seed, core)]
        values = {}
        for name in ("initializer", "initializer_report"):
            path = Path(record[name]["path"])
            values[name] = {"path": str(path), **file_fingerprint(path)}
        result[run_name(seed, core)] = values
    return result


def assert_external_fingerprints_equal(
    loaded: Mapping[Tuple[int, str], Mapping], expected: Mapping
) -> None:
    actual = fingerprint_external_inputs(loaded)
    if actual != expected:
        raise RuntimeError("Initializer or paired-initializer report changed during evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-cache", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cache_dir = assert_canonical_directory(
        guard_forbidden_path(args.development_cache, "development cache"),
        "development cache",
    )
    runs_root = assert_canonical_directory(
        guard_forbidden_path(args.runs_root, "runs root"), "runs root"
    )
    output_dir = refuse_existing_output(args.output_dir)

    cache_fingerprints = fingerprint_cache(cache_dir)
    cache = validate_cache_structure(cache_dir, CLASS_NAMES)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    if len(labels) != DEVELOPMENT_SAMPLES or tuple(
        int((labels == label).sum()) for label in range(len(CLASS_NAMES))
    ) != DEVELOPMENT_COUNTS:
        raise RuntimeError("Development cache class identity drifted")
    run_fingerprints = fingerprint_all_runs(runs_root)
    loaded = load_six_runs(runs_root, labels, run_fingerprints)
    external_fingerprints = fingerprint_external_inputs(loaded)
    complement_indices, canonical_val_indices, partition_audit = (
        derive_development_partitions(
            labels,
            loaded[(0, "quantum")]["train_indices"],
            loaded[(0, "quantum")]["val_indices"],
        )
    )

    configure_deterministic_runtime()
    canonical = evaluate_six_models(
        loaded, cache_dir, canonical_val_indices, labels
    )
    replay_audit = validate_exact_canonical_replay(loaded, canonical)
    complement = evaluate_six_models(
        loaded, cache_dir, complement_indices, labels
    )
    canonical_analysis = analyze_dataset(
        canonical["labels"],
        canonical["logits_by_run"],
        dataset_name="canonical_development_validation",
        role=(
            "consistency-only endpoint assessment; not used for epoch or model selection"
        ),
    )
    complement_analysis = analyze_dataset(
        complement["labels"],
        complement["logits_by_run"],
        dataset_name="development_training_complement",
        role=(
            "primary confirmation set: disjoint from all six training subsets, "
            "but previously exposed to the broader research process"
        ),
    )
    canonical_consistency = canonical_consistency_gates(canonical_analysis)

    after_cache = fingerprint_cache(cache_dir)
    assert_fingerprints_equal(after_cache, cache_fingerprints, "development cache")
    after_runs = fingerprint_all_runs(runs_root)
    if after_runs != run_fingerprints:
        raise RuntimeError("One or more six-run artifacts changed during evaluation")
    assert_external_fingerprints_equal(loaded, external_fingerprints)
    protocol_gate = {
        "conditions": {
            "six_fixed_seed_core_runs_strictly_validated": True,
            "last_and_best_endpoint_states_bitwise_equal": True,
            "one_validation_only_at_fixed_epoch_40": True,
            "paired_training_rng_and_stochastic_traces_equal_within_seed": True,
            "canonical_validation_replay_bitwise_exact_for_all_six": True,
            "development_complement_sha_locked_and_training_disjoint": True,
            "all_input_sha256_fingerprints_unchanged": True,
            "official_test_not_referenced_or_evaluated": True,
        },
        "passed": True,
    }
    primary_ni = bool(
        complement_analysis["accuracy_gates"]["noninferiority"]["passed"]
        and canonical_consistency["noninferiority_consistency"]["passed"]
    )
    primary_superiority = bool(
        complement_analysis["accuracy_gates"]["strict_superiority"]["passed"]
        and canonical_consistency["superiority_consistency"]["passed"]
    )
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "scope": "Model-I development data only",
        "official_test_opened": False,
        "primary_dataset": "development_training_complement",
        "primary_dataset_disclosure": (
            "training-disjoint for these six models but previously exposed by a "
            "different development experiment; not an unseen holdout"
        ),
        "canonical_validation_role": (
            "bitwise endpoint replay and consistency assessment only"
        ),
        "partition_audit": partition_audit,
        "canonical_replay_audit": replay_audit,
        "protocol_gate": protocol_gate,
        "canonical_validation": canonical_analysis,
        "canonical_consistency_gates": canonical_consistency,
        "development_complement": complement_analysis,
        "primary_decisions": {
            "noninferiority_passed": protocol_gate["passed"] and primary_ni,
            "strict_superiority_passed": (
                protocol_gate["passed"] and primary_superiority
            ),
            "canonical_validation_used_only_as_predeclared_directional_consistency": True,
            "canonical_validation_p_value_used": False,
            "random_seed_population_claim": False,
        },
        "parameter_statement": {
            "parameters_per_model": 122_573,
            "quantum_parameters_in_quantum_arm": 132,
            "classical_core_parameters_in_control_arm": 132,
            "prediction_ensemble_used": False,
        },
    }
    input_audit = {
        "development_cache": cache_fingerprints,
        "runs": run_fingerprints,
        "external_initializers_and_reports": external_fingerprints,
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "seeds": list(SEEDS),
        "cores": list(CORES),
        "fixed_epoch": EPOCHS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferiority_also_requires_every_seed_and_every_class_mean_recall_above_margin": True,
        "strict_superiority_practical_floor": SUPERIORITY_PRACTICAL_FLOOR,
        "strict_superiority_requires_all_three_seed_effects_positive": True,
        "strict_superiority_requires_all_three_directional_exact_mcnemar_p_below_0_05": True,
        "canonical_noninferiority_consistency": (
            "fixed-seed mean and minimum seed effect both strictly above -0.005"
        ),
        "canonical_superiority_consistency": (
            "fixed-seed mean strictly positive and every seed effect nonnegative"
        ),
        "canonical_validation_p_value_used": False,
        "bootstrap_resamples_examples_not_seeds": True,
        "random_seed_population_claim": False,
        "official_test_argument_or_code_path": False,
    }

    staging = output_dir.with_name(
        f".{output_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if os.path.lexists(staging):  # pragma: no cover - UUID collision guard
        raise RuntimeError(f"Unexpected staging-path collision: {staging}")
    staging.mkdir()
    try:
        atomic_npz(
            staging / "canonical_validation_predictions.npz",
            **prediction_arrays(canonical),
        )
        atomic_npz(
            staging / "development_complement_predictions.npz",
            **prediction_arrays(complement),
        )
        atomic_json(staging / "analysis.json", analysis)
        atomic_json(staging / "protocol.json", protocol)
        atomic_json(staging / "input_sha256_audit.json", input_audit)

        # Re-audit after output serialization as well as after inference.
        assert_fingerprints_equal(
            fingerprint_cache(cache_dir), cache_fingerprints, "development cache"
        )
        if fingerprint_all_runs(runs_root) != run_fingerprints:
            raise RuntimeError("Run artifacts changed while serializing results")
        assert_external_fingerprints_equal(loaded, external_fingerprints)

        output_artifacts = (
            "canonical_validation_predictions.npz",
            "development_complement_predictions.npz",
            "analysis.json",
            "protocol.json",
            "input_sha256_audit.json",
        )
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "artifacts": {
                name: file_fingerprint(staging / name) for name in output_artifacts
            },
        }
        atomic_json(staging / "output_manifest.json", output_manifest)
        fsync_directory(staging)
        os.replace(staging, output_dir)
        fsync_directory(output_dir.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        "SPATIAL_REPLICATION_ANALYSIS "
        + json.dumps(analysis["primary_decisions"], sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
