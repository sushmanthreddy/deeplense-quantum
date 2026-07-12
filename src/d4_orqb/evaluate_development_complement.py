"""Fail-closed paired evaluation on the unused Model-I development complement.

This module evaluates exactly two SHA-locked fixed-epoch checkpoints: the
15-weight Haar-subtype-residual candidate and its allocated zero-residual
control.  The evaluation membership is derived as the canonical Model-I
development-training membership minus their frozen hash-v1 half subset.

There is deliberately no official-test argument or code path.  Scientific
gate failure is a valid negative result and exits normally; protocol drift
raises before results are committed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .data import CachedNPYDataset, index_membership_sha256, make_loader
from .evaluate_locked import (
    MAX_VALIDATION_REPLAY_METRIC_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
    assert_canonical_directory,
    assert_no_symlink_components,
    atomic_json,
    atomic_npz,
    canonical_model_i_split,
    file_fingerprint,
    fsync_directory,
    load_validation_predictions,
    metrics_from_probabilities,
    mcnemar_exact,
    probability_replay_diagnostics,
    softmax_numpy,
    stratified_paired_bootstrap_accuracy,
    validate_cache_structure,
)
from .model import D4OrbitClassifier


SCHEMA_VERSION = 1
PROTOCOL_ID = "sha-locked-development-training-complement-paired-v1"
CLASS_NAMES = ("axion", "cdm", "no_sub")
DEVELOPMENT_COUNTS = (28_897, 29_772, 28_856)
DEVELOPMENT_SAMPLES = 87_525
IMAGE_SIZE = 96
FULL_TRAIN_SAMPLES = 70_021
HALF_TRAIN_SAMPLES = 35_001
COMPLEMENT_SAMPLES = 35_020
COMPLEMENT_COUNTS = (11_451, 12_151, 11_418)
VALIDATION_SAMPLES = 17_504
FIXED_EPOCH = 15
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20260805
LOADER_SEED = 20260805

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
PAIRED_ANALYSIS_SHA256 = (
    "1edecbea2ed4bcd4b3f4cbb93f85b9191a6580e3a80ba6115ca427f50d906dd1"
)
PROVENANCE_SHA256 = (
    "8c2ee79b30f5af198a5ad12dbe872025556a8ee2cf0cf5d5cab2d3274b058db6"
)

CANDIDATE_NAME = "subtype-trainable"
CONTROL_NAME = "subtype-zero-control"
ARM_NAMES = (CANDIDATE_NAME, CONTROL_NAME)
ARM_ARTIFACT_SHA256 = {
    CANDIDATE_NAME: {
        "best.pt": "73360992a6d63777ec6df90b419ced77e69dc54960e61b036f1da3b2c73030e1",
        "last.pt": "4c49dfdcbf18730bbcc4fae4437df34ee50ea47cf0a1e54789a205becfcf6a24",
        "best_validation_predictions.npz": "12df213c94f619b91773103527151380c614ffa74a8a30774b46419ef763148c",
        "config.json": "ce56a6707d6b9d482ab875d5a24e2f2943fb319836533336326159920ff605bb",
        "data_report.json": "6f8e03e63ad06113d4ab879110e1e0787882fac1e4c8ee6397c2a697e7beef19",
        "parameter_report.json": "e46ae8ee5a885575b0f8e70fd8cc7ad84a5a0f7ed8df5fce2b229fa71dcd1870",
        "summary.json": "04658e8c7379e5f24689dc1690901a8222ef05ca3cff412b366f75618810034b",
        "initialization_report.json": "0d0fb400cc536d860757269b0eaa25be9b6d5a685cc9799ffdaf13504fe92e82",
        "haar_subtype_selection.json": "02878d3cb807af9f449f3e482f446bfae6558e5233d372687b7497cfe7652b9e",
        "distillation_report.json": "3d51175e5cbef82bc66bc6a042f3b49a7fa232b474f665c694169316b101f134",
        "fixed_summary_normalization.json": "4ab10793c0ceeeae4a3c4e55be3b519a495e252cf28c3f5701814ac0fa22f527",
        "haar_normalization.json": "91702ed7097fbcb678a0f3a2d99d1daa2cc4c7a373fbece76674eedd9b1571d7",
        "morphology_normalization.json": "803b9947924a41b2d22e0402622bdebbcf7becc074ff36b9c0c3c45160e95eaf",
        "history.json": "42b505c99d3457eee25e1569722eafb6dfc568e8fd25c5b1a18e8409b638219e",
        "split_indices.npz": "6fba1c6f36aa7b37775d7525ae0690c75a4814dabb3d2e2c9dd93d667b877eb3",
        "symmetry_audit.json": "fa5141dcdc11a503285f8feba5d8345f594bc883788acb1fe94473cd2ddb2917",
    },
    CONTROL_NAME: {
        "best.pt": "2902876e5e870324cc8ed4c2ccef09d520ad7e17fbd4dbff9fe34ada12319fdb",
        "last.pt": "36dbf1ea256108aa2b89be04d646f9d0f9a6fc6d7ebb49ace337956e3d61d96b",
        "best_validation_predictions.npz": "b0256c0afea1177e501d9a9ada9fecb2efcea05171988cea00ee496d62a31283",
        "config.json": "48f9c8e482ec16e638176192f4ba11de6a5783befd07cd07504e3cd455f5580f",
        "data_report.json": "6f8e03e63ad06113d4ab879110e1e0787882fac1e4c8ee6397c2a697e7beef19",
        "parameter_report.json": "5efdd5b4a603d147aa6ea6af86ee788f01fadd6a65650a600d1a080f430d29ad",
        "summary.json": "70eff78dee2bf0ae0f6be3a2ddae443a0ab7767821da49f06e59d9a3fdfdcfb0",
        "initialization_report.json": "0d0fb400cc536d860757269b0eaa25be9b6d5a685cc9799ffdaf13504fe92e82",
        "haar_subtype_selection.json": "02878d3cb807af9f449f3e482f446bfae6558e5233d372687b7497cfe7652b9e",
        "distillation_report.json": "3d51175e5cbef82bc66bc6a042f3b49a7fa232b474f665c694169316b101f134",
        "fixed_summary_normalization.json": "4ab10793c0ceeeae4a3c4e55be3b519a495e252cf28c3f5701814ac0fa22f527",
        "haar_normalization.json": "91702ed7097fbcb678a0f3a2d99d1daa2cc4c7a373fbece76674eedd9b1571d7",
        "morphology_normalization.json": "803b9947924a41b2d22e0402622bdebbcf7becc074ff36b9c0c3c45160e95eaf",
        "history.json": "3b7f3066c5f5ff8be33a9b210dcfb9528f68f4f5855f4c70390edf6c4b58584b",
        "split_indices.npz": "6fba1c6f36aa7b37775d7525ae0690c75a4814dabb3d2e2c9dd93d667b877eb3",
        "symmetry_audit.json": "82bd27ef7d8a20d63fc1d62d8a9a113ac159d21277bfa774b858151ea752582e",
    },
}

MODEL_CONFIG_KEYS = (
    "heads",
    "reuploads",
    "core",
    "include_context",
    "dropout",
    "encoder_variant",
    "physics_variant",
    "physics_summary",
    "quantum_encoding",
    "observable_readout",
    "tied_mean_dispersion",
    "haar_subtype_residual",
    "shared_late_refinement",
    "haar_subtype_max_envelope",
    "r2_entanglers",
    "equatorial_readout",
    "meridional_readout",
)


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
    except TypeError:  # pragma: no cover - older torch compatibility
        return torch.load(path, map_location="cpu")


def require_sha256(path: Path, expected: str, context: str) -> Dict[str, int | str]:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"Invalid expected SHA-256 for {context}")
    actual = file_fingerprint(path)
    if actual["sha256"] != expected:
        raise RuntimeError(
            f"{context} SHA-256 mismatch: actual={actual['sha256']} expected={expected}"
        )
    return actual


def guard_forbidden_test_path(path: str | Path, context: str) -> Path:
    """Reject the official-test name and every symlink/lexical path alias."""

    path = Path(path).expanduser()
    if any(part.casefold() == "model_i_test" for part in path.parts):
        raise RuntimeError(f"{context} must never reference the official Model-I test path")
    return assert_no_symlink_components(path, context)


def refuse_existing_output(path: str | Path) -> Path:
    path = guard_forbidden_test_path(path, "output")
    if os.path.lexists(path):
        raise RuntimeError(f"Refusing to replace an existing output path: {path}")
    assert_canonical_directory(path.parent, "output parent")
    return path


def configure_deterministic_runtime(seed: int = LOADER_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Match the frozen training runtime exactly so the canonical-validation
    # replay is a meaningful reconstruction check rather than a backend-mode
    # comparison.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def fingerprint_locked_artifacts(paired_root: Path) -> Dict[str, Dict]:
    paired_root = assert_canonical_directory(
        guard_forbidden_test_path(paired_root, "paired root"), "paired root"
    )
    top = {
        "paired_analysis.json": require_sha256(
            paired_root / "paired_analysis.json",
            PAIRED_ANALYSIS_SHA256,
            "paired analysis",
        ),
        "provenance.json": require_sha256(
            paired_root / "provenance.json", PROVENANCE_SHA256, "paired provenance"
        ),
    }
    arms: Dict[str, Dict] = {}
    for name in ARM_NAMES:
        run = assert_canonical_directory(paired_root / name, f"{name} run")
        arms[name] = {
            artifact: require_sha256(run / artifact, digest, f"{name}/{artifact}")
            for artifact, digest in ARM_ARTIFACT_SHA256[name].items()
        }
    return {"top": top, "arms": arms}


def fingerprint_locked_cache(cache_dir: Path) -> Dict[str, Dict]:
    cache_dir = assert_canonical_directory(
        guard_forbidden_test_path(cache_dir, "development cache"),
        "development cache",
    )
    return {
        name: require_sha256(cache_dir / name, digest, f"development cache/{name}")
        for name, digest in DEVELOPMENT_CACHE_SHA256.items()
    }


def assert_fingerprints_unchanged(
    before: Mapping[str, Mapping], after: Mapping[str, Mapping], context: str
) -> None:
    if before != after:
        raise RuntimeError(f"{context} changed during evaluation")


def _validate_model_config(config: Mapping, *, expected_frozen: bool) -> None:
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise RuntimeError(f"Subtype config lacks model fields: {missing}")
    expected = {
        "image_size": IMAGE_SIZE,
        "heads": 4,
        "reuploads": 2,
        "core": "quantum",
        "include_context": False,
        "dropout": 0.1,
        "encoder_variant": "deep-se-haar-morph",
        "physics_variant": "base",
        "physics_summary": "moments-morphology-haar",
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "tied_mean_dispersion": False,
        "haar_subtype_residual": True,
        "haar_subtype_max_envelope": False,
        "shared_late_refinement": False,
        "r2_entanglers": False,
        "equatorial_readout": False,
        "meridional_readout": False,
        "freeze_haar_subtype_residual_at_zero": expected_frozen,
        "freeze_base_for_haar_subtype_residual": False,
        "evaluate_test": False,
        "deterministic": True,
        "epochs": FIXED_EPOCH,
        "validation_interval": FIXED_EPOCH,
        "fixed_final_validation_only": True,
        "seed": 0,
        "split_seed": 42,
        "max_train_per_class": 11_667,
        "train_subset_protocol": "hash-v1",
    }
    drift = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise RuntimeError(f"Subtype config protocol drift: {drift}")


def build_subtype_model_strict(
    config: Mapping, state: Mapping[str, torch.Tensor], *, expected_frozen: bool
) -> D4OrbitClassifier:
    """Construct from every architecture field and strict-load the full state."""

    _validate_model_config(config, expected_frozen=expected_frozen)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Checkpoint model state is empty or invalid")
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise RuntimeError("Checkpoint state must map string keys to tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise RuntimeError(f"Checkpoint contains a non-finite tensor: {name}")

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
    model.load_state_dict(state, strict=True)
    report = model.parameter_report()
    if (
        sum(parameter.numel() for parameter in model.parameters()) != 122_610
        or int(report.get("total", -1)) != 122_610
        or int(report.get("quantum", -1)) != 88
        or int(report.get("haar_subtype_residual_trainable", -1)) != 15
    ):
        raise RuntimeError(f"Subtype model parameter identity drifted: {report}")
    weight = model.haar_subtype_residual.weight.detach().cpu()
    if expected_frozen and not torch.equal(weight, torch.zeros_like(weight)):
        raise RuntimeError("Zero-residual control checkpoint is not exact zero")
    if not expected_frozen and torch.equal(weight, torch.zeros_like(weight)):
        raise RuntimeError("Trainable subtype checkpoint has an exact-zero residual")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def load_locked_arm(run: Path, *, expected_frozen: bool) -> Tuple[D4OrbitClassifier, Dict]:
    config = _read_json(run / "config.json")
    checkpoint = _torch_load(run / "best.pt")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model", "epoch", "record"}:
        raise RuntimeError("Checkpoint must contain exactly model/epoch/record")
    if int(checkpoint["epoch"]) != FIXED_EPOCH or not isinstance(
        checkpoint["record"], Mapping
    ):
        raise RuntimeError("Checkpoint is not the fixed epoch-15 endpoint")
    model = build_subtype_model_strict(
        config, checkpoint["model"], expected_frozen=expected_frozen
    )
    return model, config


def load_frozen_half_membership(paired_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    for name in ARM_NAMES:
        with np.load(paired_root / name / "split_indices.npz", allow_pickle=False) as split:
            if set(split.files) != {"train", "val"}:
                raise RuntimeError(f"{name} split artifact schema drifted")
            train_raw, val_raw = np.asarray(split["train"]), np.asarray(split["val"])
        if train_raw.dtype.kind not in "iu" or val_raw.dtype.kind not in "iu":
            raise RuntimeError(f"{name} split indices are not integers")
        values.append(
            (
                train_raw.astype(np.int64, copy=False),
                val_raw.astype(np.int64, copy=False),
            )
        )
    if not np.array_equal(values[0][0], values[1][0]) or not np.array_equal(
        values[0][1], values[1][1]
    ):
        raise RuntimeError("Candidate and control split arrays differ")
    return values[0]


def derive_complement_membership(
    labels: np.ndarray, frozen_half: np.ndarray, stored_validation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
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
        "complement": index_membership_sha256(complement),
        "canonical_validation": index_membership_sha256(canonical_validation),
    }
    expected_memberships = {
        "canonical_full_train": FULL_TRAIN_MEMBERSHIP_SHA256,
        "frozen_half_train": HALF_TRAIN_MEMBERSHIP_SHA256,
        "complement": COMPLEMENT_MEMBERSHIP_SHA256,
        "canonical_validation": VALIDATION_MEMBERSHIP_SHA256,
    }
    if memberships != expected_memberships:
        raise RuntimeError(
            f"Development complement membership drift: actual={memberships} "
            f"expected={expected_memberships}"
        )
    if len(full_sorted) != FULL_TRAIN_SAMPLES or len(half_sorted) != HALF_TRAIN_SAMPLES:
        raise RuntimeError("Canonical full/half training sizes drifted")
    if len(complement) != COMPLEMENT_SAMPLES or len(canonical_validation) != VALIDATION_SAMPLES:
        raise RuntimeError("Complement/validation sizes drifted")
    if not np.array_equal(stored_validation, canonical_validation):
        raise RuntimeError("Stored validation order differs from the canonical split")
    if len(np.unique(half_sorted)) != len(half_sorted) or not np.isin(
        half_sorted, full_sorted, assume_unique=True
    ).all():
        raise RuntimeError("Frozen half is not a unique subset of canonical training")
    if np.intersect1d(complement, half_sorted, assume_unique=True).size:
        raise RuntimeError("Complement overlaps the frozen half")
    if not np.array_equal(np.sort(np.concatenate((half_sorted, complement))), full_sorted):
        raise RuntimeError("Frozen half and complement do not partition canonical training")
    if np.intersect1d(complement, canonical_validation, assume_unique=True).size:
        raise RuntimeError("Complement overlaps canonical validation")
    counts = tuple(int((labels[complement] == label).sum()) for label in range(3))
    if counts != COMPLEMENT_COUNTS:
        raise RuntimeError(f"Complement class counts drifted: {counts}")
    return complement, canonical_validation, {
        "membership_sha256": memberships,
        "samples": {
            "canonical_full_train": len(full_sorted),
            "frozen_half_train": len(half_sorted),
            "complement": len(complement),
            "canonical_validation": len(canonical_validation),
        },
        "complement_class_counts": dict(zip(CLASS_NAMES, counts)),
        "half_plus_complement_exactly_partitions_canonical_train": True,
        "complement_disjoint_from_canonical_validation": True,
    }


@torch.inference_mode()
def infer_pair_once(
    candidate: D4OrbitClassifier,
    control: D4OrbitClassifier,
    cache_dir: Path,
    indices: np.ndarray,
    expected_labels: np.ndarray,
    *,
    batch_size: int,
    workers: int,
) -> Dict[str, np.ndarray]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Complement evaluation requires exactly one allocated CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Complement evaluation requires CUDA bfloat16 support")
    device = torch.device("cuda")
    candidate = candidate.to(device, memory_format=torch.channels_last).eval()
    control = control.to(device, memory_format=torch.channels_last).eval()
    loader = make_loader(
        CachedNPYDataset(cache_dir, indices),
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        seed=LOADER_SEED,
    )
    labels_parts = []
    index_parts = []
    candidate_parts = []
    control_parts = []
    for images, labels, batch_indices in loader:
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            candidate_logits = candidate(images)
            control_logits = control(images)
        if not bool(torch.isfinite(candidate_logits).all()) or not bool(
            torch.isfinite(control_logits).all()
        ):
            raise RuntimeError("Complement inference produced non-finite logits")
        labels_parts.append(labels.numpy())
        index_parts.append(batch_indices.numpy())
        candidate_parts.append(candidate_logits.float().cpu().numpy())
        control_parts.append(control_logits.float().cpu().numpy())
    torch.cuda.synchronize()
    labels = np.concatenate(labels_parts).astype(np.int64, copy=False)
    observed_indices = np.concatenate(index_parts).astype(np.int64, copy=False)
    candidate_logits = np.concatenate(candidate_parts).astype(np.float32, copy=False)
    control_logits = np.concatenate(control_parts).astype(np.float32, copy=False)
    if not np.array_equal(observed_indices, indices):
        raise RuntimeError("Complement loader changed the fixed evaluation order")
    if not np.array_equal(labels, expected_labels[indices]):
        raise RuntimeError("Complement loader labels disagree with the locked cache")
    if candidate_logits.shape != (len(indices), 3) or control_logits.shape != (
        len(indices),
        3,
    ):
        raise RuntimeError("Complement logits have an invalid shape")
    return {
        "indices": observed_indices,
        "labels": labels,
        "candidate_logits": candidate_logits,
        "control_logits": control_logits,
        "candidate_probabilities": softmax_numpy(candidate_logits),
        "control_probabilities": softmax_numpy(control_logits),
    }


def paired_gate(
    *,
    candidate_correct: int,
    control_correct: int,
    bootstrap: Mapping,
    mcnemar: Mapping,
    per_class_delta: Mapping[str, int],
    candidate_metrics: Mapping,
    control_metrics: Mapping,
) -> Dict:
    conditions = {
        "candidate_correct_strictly_greater_than_control": (
            int(candidate_correct) > int(control_correct)
        ),
        "bootstrap_ci95_low_strictly_greater_than_zero": (
            float(bootstrap["ci95_low"]) > 0.0
        ),
        "mcnemar_exact_two_sided_p_strictly_less_than_0_05": (
            float(mcnemar["two_sided_exact_p"]) < 0.05
        ),
        "every_class_correct_delta_nonnegative": min(per_class_delta.values()) >= 0,
        "candidate_macro_auc_not_less_than_control": (
            float(candidate_metrics["macro_auc_ovr"])
            >= float(control_metrics["macro_auc_ovr"])
        ),
        "candidate_nll_not_greater_than_control": (
            float(candidate_metrics["nll"]) <= float(control_metrics["nll"])
        ),
    }
    return {"conditions": conditions, "passed": all(conditions.values())}


def analyze_pair(result: Mapping[str, np.ndarray]) -> Dict:
    labels = result["labels"]
    candidate_probabilities = result["candidate_probabilities"]
    control_probabilities = result["control_probabilities"]
    metrics = {
        CANDIDATE_NAME: metrics_from_probabilities(
            labels, candidate_probabilities, CLASS_NAMES
        ),
        CONTROL_NAME: metrics_from_probabilities(labels, control_probabilities, CLASS_NAMES),
    }
    candidate_predictions = candidate_probabilities.argmax(axis=1)
    control_predictions = control_probabilities.argmax(axis=1)
    candidate_correct_mask = candidate_predictions == labels
    control_correct_mask = control_predictions == labels
    candidate_correct = int(candidate_correct_mask.sum())
    control_correct = int(control_correct_mask.sum())
    per_class_correct = {}
    per_class_delta = {}
    for label, name in enumerate(CLASS_NAMES):
        mask = labels == label
        candidate_value = int((candidate_correct_mask & mask).sum())
        control_value = int((control_correct_mask & mask).sum())
        per_class_correct[name] = {
            CANDIDATE_NAME: candidate_value,
            CONTROL_NAME: control_value,
        }
        per_class_delta[name] = candidate_value - control_value
    bootstrap = stratified_paired_bootstrap_accuracy(
        labels,
        candidate_probabilities,
        control_probabilities,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    mcnemar = mcnemar_exact(labels, candidate_probabilities, control_probabilities)
    gate = paired_gate(
        candidate_correct=candidate_correct,
        control_correct=control_correct,
        bootstrap=bootstrap,
        mcnemar=mcnemar,
        per_class_delta=per_class_delta,
        candidate_metrics=metrics[CANDIDATE_NAME],
        control_metrics=metrics[CONTROL_NAME],
    )
    return {
        "metrics": metrics,
        "candidate_correct": candidate_correct,
        "control_correct": control_correct,
        "candidate_minus_control_correct": candidate_correct - control_correct,
        "per_class_correct": per_class_correct,
        "per_class_candidate_minus_control_correct": per_class_delta,
        "paired_bootstrap_accuracy": bootstrap,
        "mcnemar_exact": mcnemar,
        "predeclared_complement_gate": gate,
        "passed_full_gate": gate["passed"],
    }


def validate_canonical_validation_replay(
    paired_root: Path,
    val_indices: np.ndarray,
    development_labels: np.ndarray,
    replayed: Mapping[str, np.ndarray],
) -> Dict[str, Dict]:
    """Anchor both reconstructed models to their frozen validation artifacts."""

    if not np.array_equal(replayed["indices"], val_indices):
        raise RuntimeError("Validation replay order differs from the frozen split")
    if not np.array_equal(replayed["labels"], development_labels[val_indices]):
        raise RuntimeError("Validation replay labels differ from the locked cache")
    limits = {
        "max_probability_absolute_difference": MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
        "mean_probability_absolute_difference": (
            MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL
        ),
        "p99_probability_absolute_difference": (
            MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL
        ),
    }
    report = {}
    for name, actual_key in (
        (CANDIDATE_NAME, "candidate_probabilities"),
        (CONTROL_NAME, "control_probabilities"),
    ):
        frozen = load_validation_predictions(
            paired_root / name / "best_validation_predictions.npz",
            val_indices,
            development_labels,
            CLASS_NAMES,
        )
        diagnostics = probability_replay_diagnostics(
            replayed[actual_key], frozen["probabilities"]
        )
        violations = {
            key: {"actual": float(diagnostics[key]), "maximum": maximum}
            for key, maximum in limits.items()
            if float(diagnostics[key]) > maximum
        }
        actual_metrics = metrics_from_probabilities(
            frozen["labels"], replayed[actual_key], CLASS_NAMES
        )
        frozen_metrics = metrics_from_probabilities(
            frozen["labels"], frozen["probabilities"], CLASS_NAMES
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
            key: abs(float(actual_metrics[key]) - float(frozen_metrics[key]))
            for key in scalar_keys
        }
        maximum_metric_drift = max(metric_drift.values())
        confusion_exact = (
            actual_metrics["confusion_matrix"] == frozen_metrics["confusion_matrix"]
        )
        if (
            violations
            or diagnostics["predicted_classes_exact"] is not True
            or maximum_metric_drift > MAX_VALIDATION_REPLAY_METRIC_ATOL
            or not confusion_exact
        ):
            raise RuntimeError(
                f"{name} canonical-validation replay failed: "
                f"probability_violations={violations} diagnostics={diagnostics} "
                f"maximum_metric_drift={maximum_metric_drift} "
                f"confusion_exact={confusion_exact}"
            )
        report[name] = {
            "probability_diagnostics": diagnostics,
            "probability_limits": limits,
            "metric_absolute_drift": metric_drift,
            "maximum_metric_absolute_drift": maximum_metric_drift,
            "metric_tolerance": MAX_VALIDATION_REPLAY_METRIC_ATOL,
            "confusion_matrix_exact": True,
            "predicted_classes_exact": True,
        }
    return report


def _commit_results(output_dir: Path, arrays: Mapping[str, np.ndarray], analysis: Mapping) -> None:
    staging = output_dir.with_name(
        f".{output_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if os.path.lexists(staging):
        raise RuntimeError(f"Unexpected staging collision: {staging}")
    staging.mkdir(mode=0o750)
    try:
        atomic_npz(staging / "complement_predictions.npz", **arrays)
        atomic_json(staging / "analysis.json", analysis)
        fsync_directory(staging)
        if os.path.lexists(output_dir):
            raise RuntimeError(f"Output appeared during evaluation: {output_dir}")
        os.rename(staging, output_dir)
        fsync_directory(output_dir.parent)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                if child.is_file() and not child.is_symlink():
                    child.unlink()
            staging.rmdir()


def run(args: argparse.Namespace) -> Dict:
    cache_dir = guard_forbidden_test_path(args.development_cache, "development cache")
    paired_root = guard_forbidden_test_path(args.paired_root, "paired root")
    output_dir = refuse_existing_output(args.output_dir)
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    if len(args.source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.source_sha256
    ):
        raise ValueError("source SHA-256 must be 64 lowercase hexadecimal characters")

    artifact_before = fingerprint_locked_artifacts(paired_root)
    cache_before = fingerprint_locked_cache(cache_dir)
    cache = validate_cache_structure(cache_dir, expected_classes=CLASS_NAMES)
    metadata = cache["metadata"]
    expected_metadata = {
        "complete": True,
        "classes": list(CLASS_NAMES),
        "class_counts": dict(zip(CLASS_NAMES, DEVELOPMENT_COUNTS)),
        "samples": DEVELOPMENT_SAMPLES,
        "image_size": IMAGE_SIZE,
        "dtype": "float16",
    }
    metadata_drift = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if metadata_drift:
        raise RuntimeError(f"Development cache metadata drift: {metadata_drift}")

    half, stored_validation = load_frozen_half_membership(paired_root)
    complement, _, membership_audit = derive_complement_membership(
        cache["labels"], half, stored_validation
    )
    candidate, candidate_config = load_locked_arm(
        paired_root / CANDIDATE_NAME, expected_frozen=False
    )
    control, control_config = load_locked_arm(
        paired_root / CONTROL_NAME, expected_frozen=True
    )
    ignored = {"output_dir", "freeze_haar_subtype_residual_at_zero"}
    if {key: value for key, value in candidate_config.items() if key not in ignored} != {
        key: value for key, value in control_config.items() if key not in ignored
    }:
        raise RuntimeError("Candidate and control configs differ beyond the paired intervention")

    configure_deterministic_runtime()
    validation_replayed = infer_pair_once(
        candidate,
        control,
        cache_dir,
        stored_validation,
        cache["labels"],
        batch_size=args.batch_size,
        workers=args.workers,
    )
    validation_replay = validate_canonical_validation_replay(
        paired_root,
        stored_validation,
        cache["labels"],
        validation_replayed,
    )
    # Construct/read the complementary-membership dataset only after both
    # canonical-validation replays have passed all fixed numerical bounds.
    del validation_replayed
    arrays = infer_pair_once(
        candidate,
        control,
        cache_dir,
        complement,
        cache["labels"],
        batch_size=args.batch_size,
        workers=args.workers,
    )
    pair_analysis = analyze_pair(arrays)

    artifact_after = fingerprint_locked_artifacts(paired_root)
    cache_after = fingerprint_locked_cache(cache_dir)
    assert_fingerprints_unchanged(artifact_before, artifact_after, "Paired artifacts")
    assert_fingerprints_unchanged(cache_before, cache_after, "Development cache")
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_sha256": args.source_sha256,
        "scope": (
            "canonical Model-I development-training complement unused by the "
            "two evaluated arms, their base initializer, and their teachers"
        ),
        "comparators": [CANDIDATE_NAME, CONTROL_NAME],
        "q0_comparison_performed": False,
        "official_test_opened": False,
        "gate_provenance": {
            "scope": (
                "new complement-only gate frozen before the first complement "
                "inference by this source digest"
            ),
            "relationship_to_prior_validation_gate": (
                "stricter complementary-holdout confirmation rule; it does "
                "not replace or alter the prior negative validation decision"
            ),
            "prior_validation_passed_full_gate": False,
        },
        "inference": {
            "device": "cuda",
            "autocast_dtype": "bfloat16",
            "single_shared_loader_pass": True,
            "identical_sample_order": True,
            "loader_seed": LOADER_SEED,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "deterministic_algorithms": True,
        },
        "membership": membership_audit,
        "canonical_validation_replay": validation_replay,
        "source_pair": {
            "paired_analysis_sha256": PAIRED_ANALYSIS_SHA256,
            "provenance_sha256": PROVENANCE_SHA256,
            "fixed_checkpoint_epoch": FIXED_EPOCH,
            "artifacts": artifact_before,
        },
        "development_cache": cache_before,
        **pair_analysis,
    }
    output_arrays = {
        "indices": arrays["indices"],
        "labels": arrays["labels"],
        "candidate_logits": arrays["candidate_logits"],
        "candidate_probabilities": arrays["candidate_probabilities"],
        "control_logits": arrays["control_logits"],
        "control_probabilities": arrays["control_probabilities"],
    }
    _commit_results(output_dir, output_arrays, analysis)
    print("DEVELOPMENT_COMPLEMENT_ANALYSIS " + json.dumps(analysis, sort_keys=True), flush=True)
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-cache", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    # A failed scientific gate is intentionally a successful, retained result.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
