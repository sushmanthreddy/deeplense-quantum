"""Train and evaluate the D4 orbit-reuploading Model-I classifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    CachedNPYDataset,
    deterministic_subset,
    fixed_stratified_split,
    hash_ranked_subset,
    index_membership_sha256,
    make_loader,
    prepare_cache,
    stratified_hash_folds,
    verify_cache_disjoint,
)
from .metrics import classification_metrics
from .model import (
    HAAR_MORPHOLOGY_CONTEXT_INDICES,
    D4OrbitClassifier,
    annular_haar_scattering_summary,
    cross_scale_scattering_summary,
    d4_transform,
    d4_views,
    invariant_annular_haar_coefficients,
    lens_morphology_summary,
)
from .quantum import D4_ELEMENTS, right_regular_permutation


OOF_DEVELOPMENT_MANIFEST_SHA256 = (
    "c04a3c62afebe3f660ffaad4333b6632471a91c6f5f239f84e68b4b94c330025"
)
OOF_DEVELOPMENT_IMAGES_SHA256 = (
    "c3c639584e0a9e2d6ba369e3fb41ba0451b2170a362ec419fa1182e55d5ce070"
)
OOF_DEVELOPMENT_LABELS_SHA256 = (
    "3a12100d1df155738b57255e6625ba1287c2d74c5bab53bce2cd0597afe89b17"
)
OOF_DEVELOPMENT_METADATA_SHA256 = (
    "9f36faff5fc3300b97512b1c29439b2e993888318a31835938749e10fbf12379"
)
OOF_FULL_HALF_MEMBERSHIP_SHA256 = (
    "571d23ced25095cf0cfb57216654f9b7be289b0589a95489a5a815a866aaee71"
)
OOF_CANONICAL_VAL_MEMBERSHIP_SHA256 = (
    "454935a294c5bb0f7c66c5b5c61072e469575b1ad68fda9fa3efb057db97ec52"
)
ANNULAR_HAAR_BASE_CHECKPOINT_SHA256 = (
    "9850a7d0c53b6332739696a9952b94a85a9ecea9c567fe5e0e85bdeef296b144"
)
HAAR_SUBTYPE_SELECTION_SPEC_SHA256 = (
    "8c8aca6a0cf66ce4d547ec746851c7c698fbc9fe85b6262d70c94085d098d470"
)
CROSS_SCALE_STATE_KEYS = frozenset(
    {
        "cross_scale_reupload_gates",
        "cross_scale_mean",
        "cross_scale_scale",
        "cross_scale_walsh",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--core",
        choices=("quantum", "classical", "hybrid", "classical-fusion"),
        default="quantum",
    )
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument(
        "--encoder-variant",
        choices=(
            "micro",
            "micro-stat",
            "deep-se",
            "deep-se-morph",
            "deep-se-haar-morph",
            "deep-se-mscorr",
            "eca",
            "tiny",
            "small",
        ),
        default="tiny",
    )
    parser.add_argument("--physics-variant", choices=("base", "radial"), default="base")
    parser.add_argument(
        "--physics-summary",
        choices=(
            "none",
            "moments",
            "moments-spectral",
            "moments-morphology",
            "moments-morphology-haar",
        ),
        default="none",
    )
    parser.add_argument(
        "--quantum-encoding",
        choices=("angle", "boltzmann", "gibbs"),
        default="angle",
    )
    parser.add_argument(
        "--observable-readout",
        choices=("pair", "plaquette", "cayley-complete"),
        default="pair",
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--reuploads", type=int, default=2)
    parser.add_argument(
        "--tied-mean-dispersion",
        action="store_true",
        help=(
            "Opt-in annular-Haar candidate that reuses the learned mean "
            "projection columns for final-map dispersion with eight zero-start gates"
        ),
    )
    parser.add_argument(
        "--haar-subtype-residual",
        action="store_true",
        help=(
            "Opt-in 15-parameter train-selected invariant Haar correction "
            "for axion-versus-CDM logits"
        ),
    )
    parser.add_argument(
        "--haar-subtype-max-envelope",
        action="store_true",
        help=(
            "Apply the subtype residual through an exact max-preserving "
            "parent envelope; requires --haar-subtype-residual"
        ),
    )
    parser.add_argument(
        "--freeze-haar-subtype-residual-at-zero",
        action="store_true",
        help=(
            "Matched continuation control that keeps only the allocated "
            "15-weight Haar subtype residual fixed at exact zero while the "
            "shared base model remains trainable"
        ),
    )
    parser.add_argument(
        "--freeze-base-for-haar-subtype-residual",
        action="store_true",
        help=(
            "Optimize only the 15 Haar subtype weights while preserving every "
            "base-model parameter and buffer bitwise; requires the max envelope"
        ),
    )
    parser.add_argument(
        "--shared-late-refinement",
        action="store_true",
        help=(
            "Opt-in base-Haar extension with four zero-start scalar gates "
            "over weight-shared late refinement blocks"
        ),
    )
    parser.add_argument(
        "--cross-scale-reupload",
        action="store_true",
        help=(
            "Opt-in base-Haar quantum/classical extension with a frozen "
            "32-D cross-scale scattering bank and four zero-start reupload gates"
        ),
    )
    parser.add_argument(
        "--r2-entanglers",
        action="store_true",
        help=(
            "Opt-in exact-budget annular-Haar quantum circuit with zero-start "
            "R2-edge ZZ/XX rotations and a gauge-fixed classifier bias"
        ),
    )
    parser.add_argument(
        "--freeze-r2-entanglers-at-zero",
        action="store_true",
        help=(
            "Matched continuation control that keeps the allocated R2 ZZ/XX "
            "angles fixed at their exact-zero initialization"
        ),
    )
    parser.add_argument(
        "--equatorial-readout",
        action="store_true",
        help=(
            "Opt-in exact-budget annular-Haar quantum model with sixteen "
            "zero-start D4-tied equatorial measurement-basis phases"
        ),
    )
    parser.add_argument(
        "--freeze-equatorial-readout-at-zero",
        action="store_true",
        help=(
            "Matched continuation control that allocates but freezes all "
            "sixteen equatorial measurement phases at exact zero"
        ),
    )
    parser.add_argument(
        "--meridional-readout",
        action="store_true",
        help=(
            "Opt-in exact-budget annular-Haar quantum model with sixteen "
            "zero-start D4-tied XZ-plane measurement-basis phases"
        ),
    )
    parser.add_argument(
        "--freeze-meridional-readout-at-zero",
        action="store_true",
        help=(
            "Matched continuation control that allocates but freezes all "
            "sixteen meridional measurement phases at exact zero"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Run full validation every N epochs (epoch 1 and the final epoch are always evaluated)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--io-workers", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2.5e-3)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--core-learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hierarchical-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--branch-loss-weight",
        type=float,
        default=0.0,
        help="Deep supervision for both branches of hybrid/fusion cores",
    )
    parser.add_argument("--max-translation-pixels", type=int, default=0)
    parser.add_argument("--translation-probability", type=float, default=1.0)
    parser.add_argument("--photon-noise-probability", type=float, default=0.0)
    parser.add_argument("--photon-count-min", type=float, default=256.0)
    parser.add_argument("--photon-count-max", type=float, default=2048.0)
    parser.add_argument("--psf-blur-probability", type=float, default=0.0)
    parser.add_argument("--read-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--subtype-mixup-probability",
        type=float,
        default=0.0,
        help="Train-only cross-class MixUp probability for axion/CDM samples",
    )
    parser.add_argument("--subtype-mixup-alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--max-train-per-class", type=int)
    parser.add_argument("--max-val-per-class", type=int)
    parser.add_argument(
        "--train-subset-protocol",
        choices=("legacy-rng", "hash-v1"),
        default="legacy-rng",
        help="Use hash-v1 for model-seed-independent nested reduced-data subsets",
    )
    parser.add_argument(
        "--oof-teacher-fold-index",
        type=int,
        choices=(0, 1),
        help=(
            "Leakage-safe fixed-epoch teacher mode: train on this fold of the "
            "fixed half subset and validate/predict only the complementary fold"
        ),
    )
    parser.add_argument(
        "--save-last-validation-predictions",
        action="store_true",
        help="Persist the most recent validation logits independently of best-checkpoint selection",
    )
    parser.add_argument(
        "--fixed-final-validation-only",
        action="store_true",
        help="Disable intermediate validation and select only the predeclared final epoch",
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--subtype-specialist",
        action="store_true",
        help=(
            "Train a validation-only binary axion/CDM specialist after the "
            "canonical three-class split and reduced-data subset are fixed"
        ),
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--training-rng-seed",
        type=int,
        help=(
            "Reset Python, NumPy, CPU, and CUDA RNG state immediately before "
            "epoch 1. This is distinct from the data-subset/loader seed and "
            "supports paired-core experiments."
        ),
    )
    parser.add_argument(
        "--save-stochastic-trace",
        action="store_true",
        help=(
            "Persist ordered sample and pre-augmentation RNG-state digests for "
            "every training batch. Requires deterministic training and an "
            "explicit --training-rng-seed."
        ),
    )
    parser.add_argument(
        "--paired-spatial-init-report",
        help=(
            "SHA-audited paired initializer report for the fixed spatial-stat "
            "quantum/classical replication protocol"
        ),
    )
    parser.add_argument("--init-backbone-checkpoint")
    parser.add_argument(
        "--init-compatible-backbone-checkpoint",
        help="Load only exact-shape tensors from a declared compatible encoder prefix",
    )
    parser.add_argument(
        "--init-full-checkpoint",
        help="Resume all compatible model weights (useful for resolution-only fine-tuning)",
    )
    parser.add_argument(
        "--reinitialize-core-after-init",
        action="store_true",
        help=(
            "For a matched annular-Haar core control, restore the freshly "
            "seeded target core after the exact full-checkpoint remap"
        ),
    )
    parser.add_argument(
        "--distillation-teacher-checkpoint",
        action="append",
        help="Frozen same-subset teacher checkpoint; repeat for a probability-ensemble teacher",
    )
    parser.add_argument(
        "--oof-distillation-artifact",
        help="Leakage-audited fixed-half OOF morphology/spatial teacher logits (.npz)",
    )
    parser.add_argument(
        "--oof-distillation-report",
        help="JSON provenance report paired with --oof-distillation-artifact",
    )
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    return parser.parse_args()


def validate_meridional_flag_contract(args: argparse.Namespace) -> None:
    """Fail closed on an orphan freeze flag or combined architecture extension."""

    enabled = bool(getattr(args, "meridional_readout", False))
    freeze = bool(getattr(args, "freeze_meridional_readout_at_zero", False))
    if freeze and not enabled:
        raise ValueError(
            "--freeze-meridional-readout-at-zero requires --meridional-readout"
        )
    if not enabled:
        return
    conflicts = tuple(
        name
        for name in (
            "tied_mean_dispersion",
            "haar_subtype_residual",
            "shared_late_refinement",
            "r2_entanglers",
            "equatorial_readout",
            "cross_scale_reupload",
            "reinitialize_core_after_init",
        )
        if bool(getattr(args, name, False))
    )
    if conflicts:
        raise ValueError(
            "Meridional readout is mutually exclusive with other annular-Haar "
            f"extensions: {conflicts}"
        )


def validate_cross_scale_reupload_contract(args: argparse.Namespace) -> None:
    """Fail closed unless CSSR is the only fixed-half base-Haar extension."""

    if not bool(getattr(args, "cross_scale_reupload", False)):
        return
    conflicts = tuple(
        name
        for name in (
            "tied_mean_dispersion",
            "haar_subtype_residual",
            "haar_subtype_max_envelope",
            "freeze_haar_subtype_residual_at_zero",
            "freeze_base_for_haar_subtype_residual",
            "shared_late_refinement",
            "r2_entanglers",
            "freeze_r2_entanglers_at_zero",
            "equatorial_readout",
            "freeze_equatorial_readout_at_zero",
            "meridional_readout",
            "freeze_meridional_readout_at_zero",
            "reinitialize_core_after_init",
        )
        if bool(getattr(args, name, False))
    )
    if conflicts:
        raise ValueError(
            "--cross-scale-reupload is mutually exclusive with every other "
            f"annular-Haar extension: {conflicts}"
        )
    exact = {
        "image_size": 96,
        "encoder_variant": "deep-se-haar-morph",
        "physics_variant": "base",
        "physics_summary": "moments-morphology-haar",
        "include_context": False,
        "heads": 4,
        "reuploads": 2,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "dropout": 0.10,
        "train_subset_protocol": "hash-v1",
        "max_train_per_class": 11_667,
        "max_val_per_class": None,
        "split_seed": 42,
        "evaluate_test": False,
    }
    drift = {
        key: {"actual": getattr(args, key, None), "expected": expected}
        for key, expected in exact.items()
        if getattr(args, key, None) != expected
    }
    val_fraction = getattr(args, "val_fraction", None)
    if val_fraction is None or not math.isclose(
        float(val_fraction), 0.20, rel_tol=0.0, abs_tol=1e-12
    ):
        drift["val_fraction"] = {"actual": val_fraction, "expected": 0.20}
    if getattr(args, "core", None) not in ("quantum", "classical"):
        drift["core"] = {
            "actual": getattr(args, "core", None),
            "expected": "quantum or classical",
        }
    if drift:
        raise ValueError(f"Cross-scale reupload architecture/data drift: {drift}")
    if not getattr(args, "init_full_checkpoint", None):
        raise ValueError(
            "--cross-scale-reupload requires an exact base-Haar full checkpoint"
        )
    if getattr(args, "init_backbone_checkpoint", None) or getattr(
        args, "init_compatible_backbone_checkpoint", None
    ):
        raise ValueError(
            "--cross-scale-reupload forbids backbone and compatible-prefix "
            "checkpoint modes"
        )
    forbidden_training = {
        "subtype_specialist": bool(getattr(args, "subtype_specialist", False)),
        "oof_teacher_fold_index": getattr(args, "oof_teacher_fold_index", None)
        is not None,
        "distillation_teacher_checkpoint": bool(
            getattr(args, "distillation_teacher_checkpoint", None)
        ),
        "oof_distillation_artifact": bool(
            getattr(args, "oof_distillation_artifact", None)
        ),
    }
    enabled = sorted(name for name, value in forbidden_training.items() if value)
    if enabled:
        raise ValueError(
            "Cross-scale reupload is a development-only undistilled three-class "
            f"study and forbids: {enabled}"
        )
    for context, raw_path in (
        ("development root", getattr(args, "development_root", "")),
        ("unused test sentinel", getattr(args, "test_root", "")),
        ("output", getattr(args, "output_dir", "")),
        ("base-Haar checkpoint", getattr(args, "init_full_checkpoint", "")),
    ):
        if any(part.casefold() == "model_i_test" for part in Path(raw_path).parts):
            raise ValueError(
                f"Cross-scale reupload {context} must not reference Model_I_test"
            )


def validate_haar_subtype_freeze_contract(args: argparse.Namespace) -> None:
    """Fail closed unless the paired control freezes only the plain residual."""

    freeze = bool(
        getattr(args, "freeze_haar_subtype_residual_at_zero", False)
    )
    if not freeze:
        return
    if not bool(getattr(args, "haar_subtype_residual", False)):
        raise ValueError(
            "--freeze-haar-subtype-residual-at-zero requires "
            "--haar-subtype-residual"
        )
    if bool(getattr(args, "haar_subtype_max_envelope", False)):
        raise ValueError(
            "--freeze-haar-subtype-residual-at-zero forbids the max envelope"
        )
    if bool(getattr(args, "freeze_base_for_haar_subtype_residual", False)):
        raise ValueError(
            "--freeze-haar-subtype-residual-at-zero freezes only the residual "
            "and cannot be combined with frozen-base optimization"
        )


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_bytes_sha256(value: torch.Tensor) -> str:
    """Hash a tensor's exact ordered CPU bytes for protocol auditing."""

    array = value.detach().cpu().contiguous().numpy()
    return _sha256_bytes(array.tobytes(order="C"))


def training_rng_state_digests() -> Dict[str, str]:
    """Return reproducible digests of every RNG stream used by training."""

    result = {
        "python": _sha256_bytes(pickle.dumps(random.getstate(), protocol=4)),
        "numpy": _sha256_bytes(pickle.dumps(np.random.get_state(), protocol=4)),
        "torch_cpu": tensor_bytes_sha256(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        result["torch_cuda_device_0"] = tensor_bytes_sha256(
            torch.cuda.get_rng_state(0)
        )
    return result


def validate_paired_spatial_training_contract(args: argparse.Namespace) -> None:
    """Fail closed on drift from the fixed spatial-stat replication design."""

    report = getattr(args, "paired_spatial_init_report", None)
    if not report:
        if getattr(args, "save_stochastic_trace", False) and (
            not getattr(args, "deterministic", False)
            or getattr(args, "training_rng_seed", None) is None
        ):
            raise ValueError(
                "--save-stochastic-trace requires deterministic training and "
                "--training-rng-seed"
            )
        return

    exact = {
        "image_size": 96,
        "encoder_variant": "micro-stat",
        "physics_variant": "base",
        "physics_summary": "moments",
        "include_context": False,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "heads": 4,
        "reuploads": 3,
        "epochs": 40,
        "patience": 41,
        "validation_interval": 40,
        "batch_size": 256,
        "workers": 4,
        "io_workers": 8,
        "split_seed": 42,
        "max_train_per_class": 11_667,
        "train_subset_protocol": "hash-v1",
        "evaluate_test": False,
        "deterministic": True,
        "fixed_final_validation_only": True,
        "save_last_validation_predictions": True,
        "save_stochastic_trace": True,
    }
    drift = {
        key: {"actual": getattr(args, key, None), "expected": expected}
        for key, expected in exact.items()
        if getattr(args, key, None) != expected
    }
    floats = {
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
    }
    for key, expected in floats.items():
        actual = getattr(args, key, None)
        if actual is None or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            drift[key] = {"actual": actual, "expected": expected}
    if drift:
        raise ValueError(f"Paired spatial replication protocol drift: {drift}")
    if args.core not in ("quantum", "classical"):
        raise ValueError("Paired spatial replication requires quantum or classical core")
    if args.seed not in (0, 1, 2):
        raise ValueError("Paired spatial replication seed must be one of 0, 1, 2")
    if args.training_rng_seed != 20_000 + args.seed:
        raise ValueError(
            "Paired spatial replication training RNG seed must equal 20000 + seed"
        )
    if not args.init_full_checkpoint:
        raise ValueError("Paired spatial replication requires a full initializer")
    if args.init_backbone_checkpoint or args.init_compatible_backbone_checkpoint:
        raise ValueError("Paired spatial replication forbids another initializer mode")
    forbidden_flags = {
        "tied_mean_dispersion": args.tied_mean_dispersion,
        "haar_subtype_residual": args.haar_subtype_residual,
        "shared_late_refinement": args.shared_late_refinement,
        "r2_entanglers": args.r2_entanglers,
        "equatorial_readout": args.equatorial_readout,
        "meridional_readout": args.meridional_readout,
        "cross_scale_reupload": bool(
            getattr(args, "cross_scale_reupload", False)
        ),
        "subtype_specialist": args.subtype_specialist,
        "oof_teacher_fold_index": args.oof_teacher_fold_index is not None,
        "distillation_teacher_checkpoint": bool(
            args.distillation_teacher_checkpoint
        ),
        "oof_distillation_artifact": bool(args.oof_distillation_artifact),
    }
    enabled = sorted(key for key, value in forbidden_flags.items() if value)
    if enabled:
        raise ValueError(
            "Paired spatial replication forbids architecture/training extensions: "
            f"{enabled}"
        )
    for context, raw_path in (
        ("development root", args.development_root),
        ("unused test root", args.test_root),
        ("output", args.output_dir),
        ("initializer", args.init_full_checkpoint),
        ("initializer report", report),
    ):
        if any(part.casefold() == "model_i_test" for part in Path(raw_path).parts):
            raise ValueError(
                f"Paired spatial replication {context} must not reference Model_I_test"
            )


def validate_paired_spatial_initializer_binding(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint: Dict,
    state: Dict[str, torch.Tensor],
) -> Dict:
    """Bind a paired run to the declared seed/core initializer, not just shape."""

    report_path = Path(args.paired_spatial_init_report)
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid paired initializer report: {report_path}") from error
    if not isinstance(report, dict):
        raise RuntimeError("Paired initializer report must contain an object")
    expected_protocol = "model-i-spatial-stat-paired-initializer-v1"
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id") != expected_protocol
        or report.get("seeds") != [0, 1, 2]
        or report.get("official_test_opened") is not False
        or report.get("official_test_reference_accepted") is not False
    ):
        raise RuntimeError("Paired initializer report protocol identity drifted")
    payload_digest = report.get("report_payload_sha256")
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
    if payload_digest != _sha256_bytes(canonical):
        raise RuntimeError("Paired initializer report payload digest drifted")
    try:
        seed_report = report["per_seed"][str(args.seed)]
        arm = seed_report["arms"][args.core]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Paired initializer report lacks the requested arm") from error
    expected_path = (report_path.parent / arm["checkpoint_path"]).resolve()
    if checkpoint_path.resolve() != expected_path:
        raise RuntimeError(
            "Paired initializer path is cross-wired: "
            f"actual={checkpoint_path.resolve()} expected={expected_path}"
        )
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != arm.get("checkpoint_sha256"):
        raise RuntimeError("Paired initializer checkpoint digest drifted")
    from .spatial_paired_init import state_sha256

    full_state_sha256 = state_sha256(state)
    core_state = {key: value for key, value in state.items() if key.startswith("core.")}
    noncore_state = {
        key: value for key, value in state.items() if not key.startswith("core.")
    }
    core_state_sha256 = state_sha256(core_state)
    noncore_state_sha256 = state_sha256(noncore_state)
    expected_values = {
        "full_state_sha256": full_state_sha256,
        "core_state_sha256": core_state_sha256,
        "native_core_state_sha256": core_state_sha256,
        "noncore_state_sha256": noncore_state_sha256,
        "core_name": args.core,
    }
    drift = {
        key: {"actual": arm.get(key), "expected": expected}
        for key, expected in expected_values.items()
        if arm.get(key) != expected
    }
    checkpoint_expected = {
        "schema_version": 1,
        "protocol_id": expected_protocol,
        "epoch": 0,
        "seed": args.seed,
        "core_name": args.core,
        "full_state_sha256": full_state_sha256,
        "core_state_sha256": core_state_sha256,
        "native_core_state_sha256": core_state_sha256,
        "noncore_state_sha256": noncore_state_sha256,
        "common_noncore_state_sha256": noncore_state_sha256,
    }
    for key, expected in checkpoint_expected.items():
        if checkpoint.get(key) != expected:
            drift[f"checkpoint.{key}"] = {
                "actual": checkpoint.get(key),
                "expected": expected,
            }
    if seed_report.get("common_noncore_state_sha256") != noncore_state_sha256:
        drift["seed_report.common_noncore_state_sha256"] = {
            "actual": seed_report.get("common_noncore_state_sha256"),
            "expected": noncore_state_sha256,
        }
    if drift:
        raise RuntimeError(f"Paired initializer seed/core binding drift: {drift}")
    parameters = checkpoint.get("parameters", {})
    if (
        parameters.get("total") != 122_573
        or parameters.get("core") != 132
        or parameters.get("core_architecture") != args.core
        or parameters.get("quantum") != (132 if args.core == "quantum" else 0)
    ):
        raise RuntimeError("Paired initializer parameter identity drifted")
    return {
        "protocol_id": expected_protocol,
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "report_payload_sha256": payload_digest,
        "seed": int(args.seed),
        "core": args.core,
        "checkpoint_sha256": checkpoint_sha256,
        "full_state_sha256": full_state_sha256,
        "core_state_sha256": core_state_sha256,
        "common_noncore_state_sha256": noncore_state_sha256,
        "pair_binding_sha256": seed_report.get("pair_binding_sha256"),
        "cross_wire_rejected": True,
    }


def atomic_json(path: Path, value: Dict | list) -> None:
    tmp = path.with_name(f"{path.name}.building-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def atomic_checkpoint(path: Path, payload: Dict) -> None:
    tmp = path.with_name(f"{path.name}.building-{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_oof_student_data_contract(
    development_cache: Path,
    development_metadata: Dict,
    class_names: list[str],
    train_indices: np.ndarray,
    canonical_val_indices: np.ndarray,
    development_manifest_sha256: str,
) -> Dict[str, str]:
    """Fail closed unless an OOF student uses the exact locked development data."""

    cache_fingerprints = {
        "images.npy": file_sha256(development_cache / "images.npy"),
        "labels.npy": file_sha256(development_cache / "labels.npy"),
        "metadata.json": file_sha256(development_cache / "metadata.json"),
        "manifest.csv": development_manifest_sha256,
    }
    expected_cache_fingerprints = {
        "images.npy": OOF_DEVELOPMENT_IMAGES_SHA256,
        "labels.npy": OOF_DEVELOPMENT_LABELS_SHA256,
        "metadata.json": OOF_DEVELOPMENT_METADATA_SHA256,
        "manifest.csv": OOF_DEVELOPMENT_MANIFEST_SHA256,
    }
    if cache_fingerprints != expected_cache_fingerprints:
        raise RuntimeError(
            "OOF student development cache identity drifted: "
            f"actual={cache_fingerprints} expected={expected_cache_fingerprints}"
        )
    if (
        class_names != ["axion", "cdm", "no_sub"]
        or development_metadata.get("class_counts")
        != {"axion": 28897, "cdm": 29772, "no_sub": 28856}
        or index_membership_sha256(train_indices)
        != OOF_FULL_HALF_MEMBERSHIP_SHA256
        or index_membership_sha256(canonical_val_indices)
        != OOF_CANONICAL_VAL_MEMBERSHIP_SHA256
        or np.intersect1d(
            train_indices, canonical_val_indices, assume_unique=True
        ).size
    ):
        raise RuntimeError(
            "OOF student class, parent-half, or canonical-validation contract drifted"
        )
    return cache_fingerprints


def zero_extend_input_weight(
    state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    key: str,
    adapted_tensors: list,
    insert_before_tail: int = 0,
) -> None:
    """Zero-extend compatible convolutional or linear input columns in-place.

    ``insert_before_tail`` preserves a semantic suffix when new features are
    inserted between an encoder vector and an existing physics-summary vector.
    """

    source = state.get(key)
    target = target_state.get(key)
    if source is None or target is None:
        return
    source_shape, target_shape = tuple(source.shape), tuple(target.shape)
    compatible_expansion = (
        len(source_shape) == len(target_shape)
        and len(source_shape) >= 2
        and source_shape[0] == target_shape[0]
        and source_shape[2:] == target_shape[2:]
        and source_shape[1] < target_shape[1]
    )
    if not compatible_expansion:
        return
    if not 0 <= insert_before_tail <= source_shape[1]:
        raise ValueError("insert_before_tail exceeds source input width")
    expanded = target.detach().cpu().clone().zero_()
    prefix = source_shape[1] - insert_before_tail
    expanded[:, :prefix] = source[:, :prefix]
    if insert_before_tail:
        expanded[:, -insert_before_tail:] = source[:, -insert_before_tail:]
    state[key] = expanded
    adapted_tensors.append(
        {
            "key": key,
            "source_shape": source_shape,
            "target_shape": target_shape,
            "method": (
                "copy-prefix-and-tail-zero-inserted-inputs"
                if insert_before_tail
                else "copy-existing-inputs-zero-new-inputs"
            ),
            "insert_before_tail": insert_before_tail,
        }
    )


def prefix_slice_output_tensor(
    state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    key: str,
    adapted_tensors: list,
) -> None:
    """Copy a leading output-channel slice into a narrower compatible tensor."""

    source = state.get(key)
    target = target_state.get(key)
    if source is None or target is None:
        return
    source_shape, target_shape = tuple(source.shape), tuple(target.shape)
    compatible = (
        len(source_shape) == len(target_shape)
        and source_shape[0] > target_shape[0]
        and source_shape[1:] == target_shape[1:]
    )
    if not compatible:
        return
    state[key] = source[: target_shape[0]].detach().cpu().clone()
    adapted_tensors.append(
        {
            "key": key,
            "source_shape": source_shape,
            "target_shape": target_shape,
            "method": "copy-leading-output-channels",
        }
    )


def remap_projection_encoder_and_summary(
    state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    key: str,
    adapted_tensors: list,
    source_encoder_dim: int,
    target_encoder_dim: int,
    preserved_summary_dim: int,
) -> None:
    """Remap a projection when encoder width shrinks and summaries are appended.

    Leading encoder columns and the semantic fixed-moment suffix are copied;
    new morphology columns start at exactly zero.
    """

    source = state.get(key)
    target = target_state.get(key)
    if source is None or target is None:
        return
    source_shape, target_shape = tuple(source.shape), tuple(target.shape)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source_shape[0] != target_shape[0]
        or source_shape[1] != source_encoder_dim + preserved_summary_dim
        or target_shape[1] < target_encoder_dim + preserved_summary_dim
    ):
        return
    remapped = target.detach().cpu().clone().zero_()
    copied_encoder_dim = min(source_encoder_dim, target_encoder_dim)
    remapped[:, :copied_encoder_dim] = source[:, :copied_encoder_dim]
    remapped[
        :,
        target_encoder_dim : target_encoder_dim + preserved_summary_dim,
    ] = source[
        :,
        source_encoder_dim : source_encoder_dim + preserved_summary_dim,
    ]
    state[key] = remapped
    adapted_tensors.append(
        {
            "key": key,
            "source_shape": source_shape,
            "target_shape": target_shape,
            "method": "copy-encoder-prefix-and-semantic-summary-zero-new-features",
            "source_encoder_dim": source_encoder_dim,
            "target_encoder_dim": target_encoder_dim,
            "preserved_summary_dim": preserved_summary_dim,
            "copied_encoder_dim": copied_encoder_dim,
        }
    )


def remap_projection_to_multiscale_correlation(
    state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    key: str,
    adapted_tensors: list,
    source_encoder_dim: int,
    target_multiscale_dim: int,
    target_final_dim: int,
    preserved_summary_dim: int,
) -> None:
    """Map a mean-pooled deep-SE projection into a multiscale descriptor.

    New block-3/block-5 statistics occupy the leading multiscale columns and
    start at zero.  Surviving final-channel means retain the source projection
    weights, and the fixed physics-moment suffix is copied by semantic role.
    """

    source = state.get(key)
    target = target_state.get(key)
    if source is None or target is None:
        return
    source_shape, target_shape = tuple(source.shape), tuple(target.shape)
    expected_target_dim = (
        target_multiscale_dim + target_final_dim + preserved_summary_dim
    )
    if (
        source.ndim != 2
        or target.ndim != 2
        or source_shape[0] != target_shape[0]
        or source_shape[1] != source_encoder_dim + preserved_summary_dim
        or target_shape[1] != expected_target_dim
        or source_encoder_dim < target_final_dim
    ):
        return
    remapped = target.detach().cpu().clone().zero_()
    final_start = target_multiscale_dim
    final_stop = final_start + target_final_dim
    remapped[:, final_start:final_stop] = source[:, :target_final_dim]
    remapped[:, final_stop:] = source[
        :, source_encoder_dim : source_encoder_dim + preserved_summary_dim
    ]
    state[key] = remapped
    adapted_tensors.append(
        {
            "key": key,
            "source_shape": source_shape,
            "target_shape": target_shape,
            "method": "zero-new-multiscale-copy-final-prefix-and-semantic-summary",
            "source_encoder_dim": source_encoder_dim,
            "target_multiscale_dim": target_multiscale_dim,
            "target_final_dim": target_final_dim,
            "preserved_summary_dim": preserved_summary_dim,
        }
    )


def morphology_path_sensitivity_order(
    state: Dict[str, torch.Tensor], count: int = 15
) -> Tuple[int, ...]:
    """Rank morphology bypass columns without labels or validation replay."""

    projection = state.get("head.projection.weight")
    classifier = state.get("head.classifier.weight")
    if projection is None or classifier is None:
        raise ValueError("Checkpoint lacks morphology fusion head tensors")
    if tuple(projection.shape) != (18, 108) or tuple(classifier.shape) != (3, 18):
        raise ValueError(
            "Morphology-KD remap requires an 18x108 fusion projection and "
            "3x18 classifier"
        )
    if not 0 < count <= 60:
        raise ValueError("Morphology sensitivity count must be in [1, 60]")
    with torch.no_grad():
        morphology = projection[:, 48:].detach().double().cpu()
        outgoing = classifier.detach().double().cpu().abs().sum(dim=0)
        score = (morphology.abs() * outgoing[:, None]).sum(dim=0)
    return tuple(
        sorted(range(60), key=lambda index: (-float(score[index]), index))[:count]
    )


def remap_morphology_kd_to_haar_candidate(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Exactly warm-start the compact annular-Haar model from morphology-KD.

    All equal-shape tensors are copied.  The existing 268 projection columns
    retain their encoder/moment/morphology semantics and the 104 new Haar
    columns start at zero.  The fusion head keeps all quantum columns and the
    frozen 15 morphology columns in their declared sensitivity order.
    """

    frozen = tuple(HAAR_MORPHOLOGY_CONTEXT_INDICES)
    actual = morphology_path_sensitivity_order(source_state, len(frozen))
    if actual != frozen:
        raise ValueError(
            "Checkpoint morphology sensitivity order does not match the "
            f"frozen candidate: expected={frozen}, actual={actual}"
        )
    required_shapes = {
        "orbit_projection.weight": ((8, 268), (8, 372)),
        "head.projection.weight": ((18, 108), (18, 63)),
        "morphology_mean": ((60,), (60,)),
        "morphology_scale": ((60,), (60,)),
    }
    for key, (source_shape, target_shape) in required_shapes.items():
        if key not in source_state or key not in target_state:
            raise ValueError(f"Missing required morphology-KD tensor: {key}")
        if tuple(source_state[key].shape) != source_shape:
            raise ValueError(
                f"Unexpected source shape for {key}: {tuple(source_state[key].shape)}"
            )
        if tuple(target_state[key].shape) != target_shape:
            raise ValueError(
                f"Unexpected target shape for {key}: {tuple(target_state[key].shape)}"
            )

    manually_remapped = {
        "orbit_projection.weight",
        "head.projection.weight",
    }
    remapped = {
        key: value.detach().cpu().clone() for key, value in target_state.items()
    }
    copied = []
    for key, value in source_state.items():
        if key in manually_remapped:
            continue
        target = target_state.get(key)
        if target is None:
            raise ValueError(f"Unexpected source morphology-KD tensor: {key}")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"Non-remappable morphology-KD tensor {key}: "
                f"source={tuple(value.shape)} target={tuple(target.shape)}"
            )
        remapped[key] = value.detach().cpu().clone()
        copied.append(key)

    source_projection = source_state["orbit_projection.weight"].detach().cpu()
    target_projection = remapped["orbit_projection.weight"].zero_()
    target_projection[:, :268] = source_projection

    source_head = source_state["head.projection.weight"].detach().cpu()
    target_head = remapped["head.projection.weight"].zero_()
    target_head[:, :48] = source_head[:, :48]
    context_indices = torch.tensor(frozen, dtype=torch.long)
    target_head[:, 48:] = source_head[:, 48:].index_select(1, context_indices)
    remapped["morphology_context_indices"] = context_indices

    adapted = [
        {
            "key": "orbit_projection.weight",
            "source_shape": (8, 268),
            "target_shape": (8, 372),
            "method": "copy-encoder-moments-morphology-zero-new-haar",
            "copied_columns": 268,
            "zero_columns": 104,
        },
        {
            "key": "head.projection.weight",
            "source_shape": (18, 108),
            "target_shape": (18, 63),
            "method": "copy-quantum-and-frozen-morphology-context",
            "morphology_context_indices": list(frozen),
        },
    ]
    if "dispersion_gates" in remapped:
        gates = remapped["dispersion_gates"]
        if tuple(gates.shape) != (8,):
            raise ValueError(
                "The tied mean-dispersion Haar candidate requires eight gates"
            )
        if not torch.equal(gates, torch.zeros_like(gates)):
            raise ValueError(
                "Tied mean-dispersion gates must be zero before warm-start remap"
            )
        adapted.append(
            {
                "key": "dispersion_gates",
                "source_shape": None,
                "target_shape": (8,),
                "method": "zero-new-tied-mean-dispersion-gates",
                "zero_parameters": 8,
            }
        )
    if not copied:
        raise RuntimeError("Morphology-KD remap copied no exact-shape tensors")
    return remapped, adapted


def remap_haar_to_subtype_residual(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Add only a zero-initialized 15-parameter subtype residual to Haar."""

    new_keys = {
        "haar_subtype_residual.weight",
        "haar_subtype_residual.selected_indices",
        "haar_subtype_residual.center",
        "haar_subtype_residual.scale",
    }
    if any(key.startswith("haar_subtype_residual.") for key in source_state):
        raise ValueError("Subtype residual warm start requires a base-Haar source")
    if "dispersion_gates" in source_state or "dispersion_gates" in target_state:
        raise ValueError("Subtype residual cannot be combined with tied dispersion")
    if set(target_state) != set(source_state).union(new_keys):
        missing = sorted(set(source_state).union(new_keys) - set(target_state))
        unexpected = sorted(set(target_state) - set(source_state).union(new_keys))
        raise ValueError(
            "Base-Haar/subtype-residual state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    required = {
        "orbit_projection.weight": (8, 372),
        "head.projection.weight": (18, 63),
        "haar_mean": (104,),
        "haar_scale": (104,),
        "morphology_mean": (60,),
        "morphology_scale": (60,),
    }
    for key, shape in required.items():
        if key not in source_state or tuple(source_state[key].shape) != shape:
            raise ValueError(f"Base-Haar source has invalid {key} shape")
    remapped = {
        key: value.detach().cpu().clone() for key, value in target_state.items()
    }
    for key, value in source_state.items():
        if tuple(value.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"Base-Haar tensor shape drift for {key}: "
                f"source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
        remapped[key] = value.detach().cpu().clone()
    residual = remapped["haar_subtype_residual.weight"]
    if tuple(residual.shape) != (15,) or not torch.equal(
        residual, torch.zeros_like(residual)
    ):
        raise ValueError("New Haar subtype residual must contain 15 exact zeros")
    return remapped, [
        {
            "key": "haar_subtype_residual.weight",
            "source_shape": None,
            "target_shape": (15,),
            "method": "zero-new-invariant-haar-subtype-residual",
            "zero_parameters": 15,
            "exact_base_replay": True,
        }
    ]


def remap_haar_to_cross_scale_reupload(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Copy an exact base-Haar state and retain only fresh CSSR state.

    The target is required to differ by precisely the four declared CSSR
    tensors.  This prevents an extension checkpoint, a partial checkpoint, or
    an unrelated architecture from being accepted through shape coincidence.
    """

    if any(key.startswith("cross_scale_") for key in source_state):
        raise ValueError("CSSR warm start requires a base-Haar source")
    expected_target_keys = set(source_state).union(CROSS_SCALE_STATE_KEYS)
    if set(target_state) != expected_target_keys:
        missing = sorted(expected_target_keys - set(target_state))
        unexpected = sorted(set(target_state) - expected_target_keys)
        raise ValueError(
            "Base-Haar/CSSR state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    required_base = {
        "orbit_projection.weight": (8, 372),
        "head.projection.weight": (18, 63),
        "haar_mean": (104,),
        "haar_scale": (104,),
        "morphology_mean": (60,),
        "morphology_scale": (60,),
    }
    for key, shape in required_base.items():
        value = source_state.get(key)
        if value is None or tuple(value.shape) != shape:
            raise ValueError(f"Base-Haar source has invalid {key} shape")
    expected_new_shapes = {
        "cross_scale_reupload_gates": (4,),
        "cross_scale_mean": (32,),
        "cross_scale_scale": (32,),
        "cross_scale_walsh": (8, 32),
    }
    for key, shape in expected_new_shapes.items():
        value = target_state.get(key)
        if value is None or tuple(value.shape) != shape:
            raise ValueError(f"CSSR target has invalid {key} shape")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"CSSR target contains nonfinite {key}")
    if not torch.equal(
        target_state["cross_scale_reupload_gates"],
        torch.zeros_like(target_state["cross_scale_reupload_gates"]),
    ):
        raise ValueError("CSSR reupload gates must start at exact zero")
    if not torch.equal(
        target_state["cross_scale_mean"],
        torch.zeros_like(target_state["cross_scale_mean"]),
    ):
        raise ValueError("CSSR mean buffer must start at exact zero")
    if not torch.equal(
        target_state["cross_scale_scale"],
        torch.ones_like(target_state["cross_scale_scale"]),
    ):
        raise ValueError("CSSR scale buffer must start at exact one")
    walsh = target_state["cross_scale_walsh"].detach().float()
    if not torch.allclose(
        walsh.square().sum(dim=1),
        torch.ones(8, device=walsh.device),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("CSSR Walsh rows must have unit Euclidean norm")

    remapped = {
        key: value.detach().cpu().clone() for key, value in target_state.items()
    }
    for key, value in source_state.items():
        target = target_state[key]
        if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
            raise ValueError(
                f"Base-Haar tensor identity drift for {key}: "
                f"source={tuple(value.shape)}/{value.dtype} "
                f"target={tuple(target.shape)}/{target.dtype}"
            )
        remapped[key] = value.detach().cpu().clone()
    return remapped, [
        {
            "key": key,
            "source_shape": None,
            "target_shape": list(expected_new_shapes[key]),
            "method": (
                "zero-new-cross-scale-reupload-gates"
                if key == "cross_scale_reupload_gates"
                else "retain-frozen-target-cssr-buffer"
            ),
            "trainable_parameters": 4
            if key == "cross_scale_reupload_gates"
            else 0,
            "exact_base_replay": True,
        }
        for key in sorted(CROSS_SCALE_STATE_KEYS)
    ]


def validate_cross_scale_source_contract(
    source_config: Dict,
    source_parameters: Dict,
    source_data: Dict,
    source_summary: Dict,
    target_data: Dict,
    target_core: str,
) -> Dict:
    """Validate an exact same-core base-Haar fixed-half development source."""

    if target_core not in ("quantum", "classical"):
        raise ValueError("CSSR target core must be quantum or classical")
    expected_config = {
        "image_size": 96,
        "encoder_variant": "deep-se-haar-morph",
        "physics_variant": "base",
        "physics_summary": "moments-morphology-haar",
        "heads": 4,
        "reuploads": 2,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "include_context": False,
        "core": target_core,
        "dropout": 0.10,
        "evaluate_test": False,
        "train_subset_protocol": "hash-v1",
        "max_train_per_class": 11_667,
    }
    drift = {
        key: {"actual": source_config.get(key), "expected": expected}
        for key, expected in expected_config.items()
        if source_config.get(key) != expected
    }
    extensions = (
        "tied_mean_dispersion",
        "haar_subtype_residual",
        "haar_subtype_max_envelope",
        "shared_late_refinement",
        "r2_entanglers",
        "equatorial_readout",
        "meridional_readout",
        "cross_scale_reupload",
    )
    active_extensions = [
        key for key in extensions if bool(source_config.get(key, False))
    ]
    if drift or active_extensions:
        raise RuntimeError(
            "CSSR source is not exact same-core base Haar: "
            f"drift={drift}, active_extensions={active_extensions}"
        )
    expected_quantum = 88 if target_core == "quantum" else 0
    expected_classical = 88 if target_core == "classical" else 0
    if (
        int(source_parameters.get("total", -1)) != 122_595
        or int(source_parameters.get("core", -1)) != 88
        or source_parameters.get("core_architecture") != target_core
        or int(source_parameters.get("quantum", -1)) != expected_quantum
        or int(source_parameters.get("parallel_classical", -1))
        != expected_classical
        or any(
            int(source_parameters.get(key, 0)) != 0
            for key in (
                "haar_subtype_residual_trainable",
                "dispersion_gate_trainable",
                "shared_late_refinement_gate_trainable",
                "r2_entangler_trainable",
                "equatorial_readout_trainable",
                "meridional_readout_trainable",
                "cross_scale_reupload_gate_trainable",
            )
        )
    ):
        raise RuntimeError("CSSR source parameter report is not base Haar")

    def locked_half(report: Dict) -> bool:
        return bool(
            report.get("train_size") == 35_001
            and report.get("train_membership_sha256")
            == OOF_FULL_HALF_MEMBERSHIP_SHA256
            and report.get("development_manifest_sha256")
            == OOF_DEVELOPMENT_MANIFEST_SHA256
            and report.get("class_names") == ["axion", "cdm", "no_sub"]
            and report.get("official_test_cache_opened") is False
            and "test" not in report
        )

    if (
        not locked_half(source_data)
        or not locked_half(target_data)
        or source_data.get("train_membership_sha256")
        != target_data.get("train_membership_sha256")
        or source_data.get("development_manifest_sha256")
        != target_data.get("development_manifest_sha256")
        or source_summary.get("official_test_evaluated") is not False
        or "test" in source_summary
    ):
        raise RuntimeError(
            "CSSR source/target violates fixed-half membership, manifest, class, "
            "or official-test lock"
        )
    return {
        "source_parameters": 122_595,
        "source_core_parameters": 88,
        "source_core": target_core,
        "same_training_membership": True,
        "same_development_manifest": True,
        "source_official_test_opened": False,
        "target_official_test_opened": False,
    }


def cross_scale_initialization_record(model: D4OrbitClassifier) -> Dict:
    """Describe and verify the four-parameter zero-function CSSR extension."""

    gates = getattr(model, "cross_scale_reupload_gates", None)
    if not bool(getattr(model, "cross_scale_reupload", False)) or gates is None:
        raise ValueError("Model does not enable cross-scale reupload")
    if tuple(gates.shape) != (4,) or not torch.equal(
        gates.detach(), torch.zeros_like(gates.detach())
    ):
        raise RuntimeError("CSSR requires four exact-zero initial gates")
    return {
        "enabled": True,
        "feature_dimensions": 32,
        "gate_parameters": 4,
        "gate_initialization": "zeros",
        "all_gates_zero_after_remap": True,
        "zero_gate_exact_base_replay": True,
        "normalization": "clean fixed-half train; all eight D4 views",
        "validation_samples_used": 0,
        "official_test_samples_used": 0,
    }


def configure_cross_scale_optimization_and_report(
    model: D4OrbitClassifier,
) -> Dict:
    """Validate CSSR's four-parameter allocation before optimizer creation."""

    record = cross_scale_initialization_record(model)
    gates = model.cross_scale_reupload_gates
    if not gates.requires_grad:
        raise RuntimeError("CSSR gates must be trainable")
    allocated = model.parameter_report()
    forbidden_counts = (
        "haar_subtype_residual_trainable",
        "dispersion_gate_trainable",
        "shared_late_refinement_gate_trainable",
        "r2_entangler_trainable",
        "equatorial_readout_trainable",
        "meridional_readout_trainable",
    )
    if (
        int(allocated.get("total", -1)) != 122_599
        or int(allocated.get("core", -1)) != 88
        or int(allocated.get("cross_scale_reupload_gate_trainable", -1)) != 4
        or any(int(allocated.get(key, 0)) != 0 for key in forbidden_counts)
    ):
        raise RuntimeError(f"Invalid exact-budget CSSR candidate: {allocated}")
    allocated.update(
        {
            "inference_total": 122_599,
            "optimization_trainable_total": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "cross_scale_reupload_gate_optimization_trainable": 4,
            "cross_scale_reupload_gate_optimizer_group": "head",
            "cross_scale_reupload_initialization": record,
        }
    )
    if allocated["optimization_trainable_total"] != 122_599:
        raise RuntimeError(f"CSSR optimizer budget drifted: {allocated}")
    return allocated


def cross_scale_gate_update_record(
    model: D4OrbitClassifier,
    initial_gates: torch.Tensor,
) -> Dict:
    """Record the learned CSSR gates and require a real optimizer update."""

    gates = getattr(model, "cross_scale_reupload_gates", None)
    if gates is None or tuple(gates.shape) != (4,):
        raise ValueError("Model does not expose four CSSR gates")
    initial = initial_gates.detach().float().cpu()
    if tuple(initial.shape) != (4,) or not torch.equal(
        initial, torch.zeros_like(initial)
    ):
        raise ValueError("CSSR update audit requires four exact-zero initial gates")
    final = gates.detach().float().cpu()
    update_l2 = float((final - initial).norm())
    if update_l2 <= 0.0:
        raise RuntimeError("Trainable CSSR gates received no parameter update")
    return {
        "parameters": 4,
        "optimization_trainable": 4,
        "gate_update_l2": update_l2,
        "gate_l2": float(final.norm()),
        "gates": final.tolist(),
    }


def cross_scale_zero_gate_replay(
    source: D4OrbitClassifier,
    target: D4OrbitClassifier,
    images: torch.Tensor,
) -> Dict:
    """Require bitwise source logits/probabilities with CSSR gates at zero."""

    gates = getattr(target, "cross_scale_reupload_gates", None)
    if gates is None or tuple(gates.shape) != (4,) or not torch.equal(
        gates.detach(), torch.zeros_like(gates.detach())
    ):
        raise ValueError("CSSR replay requires four exact-zero gates")
    source_training, target_training = source.training, target.training
    source.eval()
    target.eval()
    try:
        with torch.inference_mode():
            source_logits = source(images)
            target_logits = target(images)
            source_probabilities = F.softmax(source_logits.float(), dim=1)
            target_probabilities = F.softmax(target_logits.float(), dim=1)
    finally:
        source.train(source_training)
        target.train(target_training)
    logit_difference = (target_logits.float() - source_logits.float()).abs()
    probability_difference = (
        target_probabilities - source_probabilities
    ).abs()
    logits_exact = bool(torch.equal(target_logits, source_logits))
    probabilities_exact = bool(
        torch.equal(target_probabilities, source_probabilities)
    )
    predictions_equal = bool(
        torch.equal(target_logits.argmax(dim=1), source_logits.argmax(dim=1))
    )
    if not logits_exact or not probabilities_exact or not predictions_equal:
        raise RuntimeError(
            "Zero-gate CSSR failed exact base-Haar replay: "
            f"logit_max={float(logit_difference.max())}, "
            f"probability_max={float(probability_difference.max())}"
        )
    return {
        "samples": int(images.shape[0]),
        "gates_all_zero": True,
        "logits_bitwise_exact": logits_exact,
        "probabilities_bitwise_exact": probabilities_exact,
        "predictions_equal": predictions_equal,
        "max_logit_absolute_difference": float(logit_difference.max()),
        "max_probability_absolute_difference": float(
            probability_difference.max()
        ),
    }


def configure_haar_subtype_optimization_and_report(
    model: D4OrbitClassifier,
    freeze_at_zero: bool,
) -> Dict:
    """Report allocated size while optionally freezing only the residual."""

    residual = model.haar_subtype_residual
    if (
        not model.haar_subtype_residual_enabled
        or residual is None
        or tuple(residual.weight.shape) != (15,)
        or model.haar_subtype_max_envelope
    ):
        raise ValueError(
            "Paired Haar subtype optimization requires the plain 15-weight residual"
        )
    if not torch.equal(
        residual.weight.detach(), torch.zeros_like(residual.weight.detach())
    ):
        raise RuntimeError(
            "Haar subtype residual changed before optimizer configuration"
        )
    allocated = model.parameter_report()
    if (
        int(allocated["total"]) != 122610
        or int(allocated["quantum"]) != 88
        or int(allocated.get("haar_subtype_residual_trainable", -1)) != 15
        or int(allocated.get("dispersion_gate_trainable", -1)) != 0
        or int(
            allocated.get("shared_late_refinement_gate_trainable", -1)
        )
        != 0
    ):
        raise RuntimeError(
            f"Invalid exact-budget Haar subtype candidate: {allocated}"
        )
    residual.weight.requires_grad_(not freeze_at_zero)
    allocated["inference_total"] = 122610
    allocated["inference_quantum"] = 88
    allocated["optimization_trainable_total"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    allocated["quantum_optimization_trainable"] = sum(
        parameter.numel()
        for parameter in model.core.parameters()
        if parameter.requires_grad
    )
    allocated["haar_subtype_residual_optimization_trainable"] = int(
        residual.weight.numel() if residual.weight.requires_grad else 0
    )
    allocated["haar_subtype_residual_frozen_at_zero"] = bool(freeze_at_zero)
    expected_optimization = 122595 if freeze_at_zero else 122610
    expected_residual_optimization = 0 if freeze_at_zero else 15
    if (
        allocated["optimization_trainable_total"] != expected_optimization
        or allocated["quantum_optimization_trainable"] != 88
        or allocated["haar_subtype_residual_optimization_trainable"]
        != expected_residual_optimization
    ):
        raise RuntimeError(
            f"Haar subtype optimizer budget drifted: {allocated}"
        )
    return allocated


def haar_subtype_residual_update_record(
    model: D4OrbitClassifier,
    initial_weight: torch.Tensor,
    freeze_at_zero: bool,
) -> Dict:
    """Audit the paired endpoint: exact-zero control or updated candidate."""

    residual = model.haar_subtype_residual
    if residual is None or tuple(residual.weight.shape) != (15,):
        raise ValueError("Model does not contain the 15-weight Haar subtype residual")
    initial = initial_weight.detach().float().cpu()
    if tuple(initial.shape) != (15,) or not torch.equal(
        initial, torch.zeros_like(initial)
    ):
        raise ValueError("Haar subtype paired audit requires exact-zero initialization")
    final = residual.weight.detach().float().cpu()
    update_l2 = float((final - initial).norm())
    exact_zero = bool(torch.equal(final, torch.zeros_like(final)))
    if freeze_at_zero and (update_l2 != 0.0 or not exact_zero):
        raise RuntimeError(
            "Frozen Haar subtype control changed an exact-zero residual weight"
        )
    if not freeze_at_zero and update_l2 <= 0.0:
        raise RuntimeError(
            "Trainable Haar subtype residual received no parameter update"
        )
    return {
        "parameters": int(final.numel()),
        "optimization_trainable": int(
            final.numel() if residual.weight.requires_grad else 0
        ),
        "frozen_at_zero_control": bool(freeze_at_zero),
        "weights_exact_zero": exact_zero,
        "weight_update_l2": update_l2,
        "weight_l2": float(final.norm()),
        "weights": final.tolist(),
    }


def remap_haar_to_shared_late_refinement(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Add only four zero gates around shared late encoder refinements."""

    gate_key = "encoder.shared_refinement_gates"
    if gate_key in source_state:
        raise ValueError("Shared late refinement requires a base-Haar source")
    if "dispersion_gates" in source_state or "dispersion_gates" in target_state:
        raise ValueError("Shared late refinement cannot use tied dispersion")
    if any(key.startswith("haar_subtype_residual.") for key in source_state) or any(
        key.startswith("haar_subtype_residual.") for key in target_state
    ):
        raise ValueError("Shared late refinement cannot use a Haar subtype residual")
    if set(target_state) != set(source_state).union({gate_key}):
        missing = sorted(set(source_state).union({gate_key}) - set(target_state))
        unexpected = sorted(set(target_state) - set(source_state).union({gate_key}))
        raise ValueError(
            "Base-Haar/shared-refinement state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    required = {
        "encoder.blocks.5.block.0.weight": (64, 1, 5, 5),
        "encoder.blocks.7.block.0.weight": (96, 1, 5, 5),
        "orbit_projection.weight": (8, 372),
        "head.projection.weight": (18, 63),
        "haar_mean": (104,),
        "haar_scale": (104,),
        "morphology_mean": (60,),
        "morphology_scale": (60,),
    }
    for key, shape in required.items():
        if key not in source_state or tuple(source_state[key].shape) != shape:
            raise ValueError(f"Base-Haar source has invalid {key} shape")
    remapped = {
        key: value.detach().cpu().clone() for key, value in target_state.items()
    }
    for key, value in source_state.items():
        if tuple(value.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"Base-Haar tensor shape drift for {key}: "
                f"source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
        remapped[key] = value.detach().cpu().clone()
    gates = remapped[gate_key]
    if tuple(gates.shape) != (4,) or not torch.equal(
        gates, torch.zeros_like(gates)
    ):
        raise ValueError("New shared-refinement gates must contain four exact zeros")
    return remapped, [
        {
            "key": gate_key,
            "source_shape": None,
            "target_shape": (4,),
            "method": "zero-new-shared-late-refinement-gates",
            "zero_parameters": 4,
            "shared_block_applications": [5, 5, 7, 7],
            "exact_base_replay": True,
        }
    ]


def remap_haar_to_r2_entanglers(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Add zero R2 entanglers and remove only the softmax-common bias gauge."""

    r2_key = "core.r2_params"
    bias_key = "head.classifier.bias"
    if r2_key in source_state:
        raise ValueError("R2 warm start requires a base-Haar source")
    if r2_key not in target_state:
        raise ValueError("R2 target is missing its new entangler parameters")
    if set(target_state) != set(source_state).union({r2_key}):
        missing = sorted(set(source_state).union({r2_key}) - set(target_state))
        unexpected = sorted(set(target_state) - set(source_state).union({r2_key}))
        raise ValueError(
            "Base-Haar/R2 state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    required = {
        "core.params": (4, 2, 11),
        r2_key: (4, 2, 2),
        bias_key: (3,),
        "orbit_projection.weight": (8, 372),
        "head.projection.weight": (18, 63),
        "haar_mean": (104,),
        "haar_scale": (104,),
        "morphology_mean": (60,),
        "morphology_scale": (60,),
    }
    for key, shape in required.items():
        state = target_state if key == r2_key else source_state
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"Base-Haar/R2 source has invalid {key} shape")
    if tuple(target_state[bias_key].shape) != (2,):
        raise ValueError("R2 target classifier must use a two-parameter bias gauge")

    remapped = {
        key: value.detach().cpu().clone() for key, value in target_state.items()
    }
    for key, value in source_state.items():
        if key == bias_key:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"Base-Haar/R2 tensor shape drift for {key}: "
                f"source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
        remapped[key] = value.detach().cpu().clone()
    r2 = remapped[r2_key]
    if not torch.equal(r2, torch.zeros_like(r2)):
        raise ValueError("New R2 entangler angles must contain 16 exact zeros")
    source_bias = source_state[bias_key].detach().cpu()
    remapped[bias_key] = (source_bias[:2] - source_bias[2]).clone()
    return remapped, [
        {
            "key": r2_key,
            "source_shape": None,
            "target_shape": (4, 2, 2),
            "method": "zero-new-r2-edge-zz-xx-entanglers",
            "zero_parameters": 16,
            "mathematical_probability_equivalence": True,
        },
        {
            "key": bias_key,
            "source_shape": (3,),
            "target_shape": (2,),
            "method": "remove-softmax-common-logit-gauge",
            "removed_parameters": 1,
            "mathematical_probability_equivalence": True,
        },
    ]


def r2_entangler_initialization_record(model: D4OrbitClassifier) -> Dict:
    """Describe and verify the exact-function R2 circuit extension."""

    r2 = getattr(model.core, "r2_params", None)
    if not model.r2_entanglers or r2 is None:
        raise ValueError("Model does not enable R2 entanglers")
    detached = r2.detach()
    all_zero = bool(torch.equal(detached, torch.zeros_like(detached)))
    if tuple(detached.shape) != (4, 2, 2) or not all_zero:
        raise RuntimeError("R2 initialization no longer replays base Haar")
    return {
        "enabled": True,
        "parameters": int(detached.numel()),
        "initialization": "zeros",
        "all_angles_zero_after_remap": all_zero,
        "edge_family": "R2 half-turn complete left-Cayley orbit",
        "pauli_rotations": ["ZZ", "XX"],
        "classifier_bias_gauge_degrees": 2,
        "zero_angle_algebraic_probability_equivalence": True,
        "mixed_precision_replay": "tolerance-verified rather than bitwise exact",
    }


def configure_r2_optimization_and_report(
    model: D4OrbitClassifier,
    freeze_at_zero: bool,
) -> Dict:
    """Separate allocated/inference size from the frozen control's optimizer size."""

    r2 = getattr(model.core, "r2_params", None)
    if not model.r2_entanglers or r2 is None or tuple(r2.shape) != (4, 2, 2):
        raise ValueError("Model is not the exact R2 annular-Haar candidate")
    if not torch.equal(r2.detach(), torch.zeros_like(r2.detach())):
        raise RuntimeError("R2 angles changed before optimizer configuration")
    allocated = model.parameter_report()
    if (
        int(allocated["total"]) != 122610
        or int(allocated["quantum"]) != 104
        or int(allocated.get("r2_entangler_trainable", -1)) != 16
        or int(allocated.get("classifier_bias_trainable", -1)) != 2
        or int(allocated.get("classifier_bias_gauge_degrees", -1)) != 2
    ):
        raise RuntimeError(f"Invalid exact-budget R2 candidate: {allocated}")
    r2.requires_grad_(not freeze_at_zero)
    allocated["inference_total"] = 122610
    allocated["optimization_trainable_total"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    allocated["quantum_optimization_trainable"] = sum(
        parameter.numel()
        for parameter in model.core.parameters()
        if parameter.requires_grad
    )
    allocated["r2_entangler_optimization_trainable"] = int(
        r2.numel() if r2.requires_grad else 0
    )
    allocated["r2_entanglers_frozen_at_zero"] = bool(freeze_at_zero)
    expected_optimization = 122594 if freeze_at_zero else 122610
    expected_quantum_optimization = 88 if freeze_at_zero else 104
    if (
        allocated["optimization_trainable_total"] != expected_optimization
        or allocated["quantum_optimization_trainable"]
        != expected_quantum_optimization
    ):
        raise RuntimeError(f"R2 optimizer budget drifted: {allocated}")
    return allocated


def r2_entangler_probability_replay(
    source: D4OrbitClassifier,
    target: D4OrbitClassifier,
    images: torch.Tensor,
) -> Dict:
    """Require prediction/softmax replay modulo the removed common logit gauge."""

    r2 = getattr(target.core, "r2_params", None)
    if r2 is None or not torch.equal(r2.detach(), torch.zeros_like(r2.detach())):
        raise ValueError("R2 replay requires sixteen zero entangler angles")
    source_training, target_training = source.training, target.training
    source.eval()
    target.eval()

    def compare(autocast_enabled: bool) -> Dict:
        with torch.inference_mode(), torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            source_logits = source(images).float()
            target_logits = target(images).float()
        source_probabilities = source_logits.softmax(dim=1)
        target_probabilities = target_logits.softmax(dim=1)
        probability_difference = (
            target_probabilities - source_probabilities
        ).abs()
        predictions_equal = bool(
            torch.equal(target_logits.argmax(dim=1), source_logits.argmax(dim=1))
        )
        probability_tolerance = 2e-6 if not autocast_enabled else 2e-3
        common_shift_tolerance = 2e-6 if not autocast_enabled else 5e-2
        probabilities_equal = bool(
            torch.allclose(
                target_probabilities,
                source_probabilities,
                rtol=probability_tolerance,
                atol=probability_tolerance,
            )
        )
        common_shift = target_logits - source_logits
        shift_spread = (
            common_shift - common_shift[:, :1]
        ).abs()
        if (
            not predictions_equal
            or not probabilities_equal
            or float(shift_spread.max()) > common_shift_tolerance
        ):
            mode = "bfloat16" if autocast_enabled else "float32"
            raise RuntimeError(
                f"Zero R2 entanglers failed {mode} probability replay: "
                f"probability_max={float(probability_difference.max())}, "
                f"common_shift_spread={float(shift_spread.max())}"
            )
        return {
            "predictions_equal": predictions_equal,
            "probabilities_equal_within_tolerance": probabilities_equal,
            "probability_tolerance": probability_tolerance,
            "common_shift_spread_tolerance": common_shift_tolerance,
            "max_probability_absolute_difference": float(
                probability_difference.max()
            ),
            "max_common_shift_spread": float(shift_spread.max()),
        }

    try:
        float32 = compare(False)
        bfloat16 = compare(True)
    finally:
        source.train(source_training)
        target.train(target_training)
    return {
        "samples": int(images.shape[0]),
        "r2_angles_all_zero": True,
        "functional_equivalence": (
            "algebraically softmax-equivalent; float32 and bfloat16 execution "
            "are tolerance-verified and need not be bitwise identical"
        ),
        "float32": float32,
        "bfloat16_autocast": bfloat16,
    }


def remap_haar_to_equatorial_readout(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Add zero measurement phases and remove the softmax-common bias gauge."""

    phase_key = "core.readout_phases"
    bias_key = "head.classifier.bias"
    if phase_key in source_state:
        raise ValueError("Equatorial warm start requires a base-Haar source")
    if phase_key not in target_state:
        raise ValueError("Equatorial target is missing its measurement phases")
    if set(target_state) != set(source_state).union({phase_key}):
        missing = sorted(set(source_state).union({phase_key}) - set(target_state))
        unexpected = sorted(set(target_state) - set(source_state).union({phase_key}))
        raise ValueError(
            "Base-Haar/equatorial state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    required = {
        "core.params": (4, 2, 11),
        phase_key: (4, 4),
        bias_key: (3,),
        "orbit_projection.weight": (8, 372),
        "head.projection.weight": (18, 63),
        "haar_mean": (104,),
        "haar_scale": (104,),
        "morphology_mean": (60,),
        "morphology_scale": (60,),
    }
    for key, shape in required.items():
        state = target_state if key == phase_key else source_state
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(
                f"Base-Haar/equatorial source has invalid {key} shape"
            )
    if tuple(target_state[bias_key].shape) != (2,):
        raise ValueError(
            "Equatorial target classifier must use a two-parameter bias gauge"
        )

    remapped = {
        key: value.detach().cpu().clone() for key, value in target_state.items()
    }
    for key, value in source_state.items():
        if key == bias_key:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"Base-Haar/equatorial tensor shape drift for {key}: "
                f"source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
        remapped[key] = value.detach().cpu().clone()
    phases = remapped[phase_key]
    if not torch.equal(phases, torch.zeros_like(phases)):
        raise ValueError("New equatorial phases must contain 16 exact zeros")
    source_bias = source_state[bias_key].detach().cpu()
    remapped[bias_key] = (source_bias[:2] - source_bias[2]).clone()
    return remapped, [
        {
            "key": phase_key,
            "source_shape": None,
            "target_shape": (4, 4),
            "method": "zero-new-d4-equatorial-measurement-phases",
            "zero_parameters": 16,
            "settings_per_head": ["local", "R", "R2", "S"],
            "mathematical_feature_equivalence": True,
        },
        {
            "key": bias_key,
            "source_shape": (3,),
            "target_shape": (2,),
            "method": "remove-softmax-common-logit-gauge",
            "removed_parameters": 1,
            "mathematical_probability_equivalence": True,
        },
    ]


def equatorial_readout_initialization_record(
    model: D4OrbitClassifier,
) -> Dict:
    """Describe and verify the exact-function equatorial readout extension."""

    phases = getattr(model.core, "readout_phases", None)
    if not model.equatorial_readout or phases is None:
        raise ValueError("Model does not enable equatorial readout")
    detached = phases.detach()
    all_zero = bool(torch.equal(detached, torch.zeros_like(detached)))
    if tuple(detached.shape) != (4, 4) or not all_zero:
        raise RuntimeError(
            "Equatorial initialization no longer replays base Haar"
        )
    return {
        "enabled": True,
        "parameters": int(detached.numel()),
        "initialization": "zeros",
        "all_phases_zero_after_remap": all_zero,
        "settings_per_head": ["local", "R", "R2", "S"],
        "observable": "P(phi)=cos(phi)X+sin(phi)Y",
        "classifier_bias_gauge_degrees": 2,
        "zero_phase_algebraic_feature_equivalence": True,
        "mixed_precision_replay": "tolerance-verified rather than bitwise exact",
    }


def configure_equatorial_optimization_and_report(
    model: D4OrbitClassifier,
    freeze_at_zero: bool,
) -> Dict:
    """Separate allocated/inference size from the frozen phase optimizer size."""

    phases = getattr(model.core, "readout_phases", None)
    if (
        not model.equatorial_readout
        or phases is None
        or tuple(phases.shape) != (4, 4)
    ):
        raise ValueError("Model is not the exact EQR-16 annular-Haar candidate")
    if not torch.equal(phases.detach(), torch.zeros_like(phases.detach())):
        raise RuntimeError("Equatorial phases changed before optimizer configuration")
    allocated = model.parameter_report()
    if (
        int(allocated["total"]) != 122610
        or int(allocated["quantum"]) != 104
        or int(allocated.get("quantum_state_preparation_trainable", -1)) != 88
        or int(allocated.get("equatorial_readout_trainable", -1)) != 16
        or int(allocated.get("classifier_bias_trainable", -1)) != 2
        or int(allocated.get("classifier_bias_gauge_degrees", -1)) != 2
    ):
        raise RuntimeError(f"Invalid exact-budget EQR-16 candidate: {allocated}")
    phases.requires_grad_(not freeze_at_zero)
    allocated["inference_total"] = 122610
    allocated["optimization_trainable_total"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    allocated["quantum_optimization_trainable"] = sum(
        parameter.numel()
        for parameter in model.core.parameters()
        if parameter.requires_grad
    )
    allocated["equatorial_readout_optimization_trainable"] = int(
        phases.numel() if phases.requires_grad else 0
    )
    allocated["equatorial_readout_frozen_at_zero"] = bool(freeze_at_zero)
    expected_optimization = 122594 if freeze_at_zero else 122610
    expected_quantum_optimization = 88 if freeze_at_zero else 104
    if (
        allocated["optimization_trainable_total"] != expected_optimization
        or allocated["quantum_optimization_trainable"]
        != expected_quantum_optimization
    ):
        raise RuntimeError(f"EQR-16 optimizer budget drifted: {allocated}")
    return allocated


def equatorial_readout_probability_replay(
    source: D4OrbitClassifier,
    target: D4OrbitClassifier,
    images: torch.Tensor,
) -> Dict:
    """Verify zero-phase feature replay and gauge-equivalent probabilities."""

    phases = getattr(target.core, "readout_phases", None)
    if phases is None or not torch.equal(
        phases.detach(), torch.zeros_like(phases.detach())
    ):
        raise ValueError("Equatorial replay requires sixteen zero phases")
    source_training, target_training = source.training, target.training
    source.eval()
    target.eval()

    def compare(autocast_enabled: bool) -> Dict:
        with torch.inference_mode(), torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            source_angles = source.orbit_encode(images)[1]
            source_features = source.core(source_angles).float()
            target_features = target.core(source_angles).float()
            source_logits = source(images).float()
            target_logits = target(images).float()
        feature_difference = (target_features - source_features).abs()
        features_bitwise_equal = bool(
            torch.equal(target_features, source_features)
        )
        source_probabilities = source_logits.softmax(dim=1)
        target_probabilities = target_logits.softmax(dim=1)
        probability_difference = (
            target_probabilities - source_probabilities
        ).abs()
        predictions_equal = bool(
            torch.equal(target_logits.argmax(dim=1), source_logits.argmax(dim=1))
        )
        probability_tolerance = 2e-6 if not autocast_enabled else 2e-3
        common_shift_tolerance = 2e-6 if not autocast_enabled else 5e-2
        probabilities_equal = bool(
            torch.allclose(
                target_probabilities,
                source_probabilities,
                rtol=probability_tolerance,
                atol=probability_tolerance,
            )
        )
        common_shift = target_logits - source_logits
        shift_spread = (common_shift - common_shift[:, :1]).abs()
        if (
            not features_bitwise_equal
            or not predictions_equal
            or not probabilities_equal
            or float(shift_spread.max()) > common_shift_tolerance
        ):
            mode = "bfloat16" if autocast_enabled else "float32"
            raise RuntimeError(
                f"Zero equatorial phases failed {mode} replay: "
                f"feature_max={float(feature_difference.max())}, "
                f"probability_max={float(probability_difference.max())}, "
                f"common_shift_spread={float(shift_spread.max())}"
            )
        return {
            "features_bitwise_equal": features_bitwise_equal,
            "max_feature_absolute_difference": float(feature_difference.max()),
            "predictions_equal": predictions_equal,
            "probabilities_equal_within_tolerance": probabilities_equal,
            "probability_tolerance": probability_tolerance,
            "common_shift_spread_tolerance": common_shift_tolerance,
            "max_probability_absolute_difference": float(
                probability_difference.max()
            ),
            "max_common_shift_spread": float(shift_spread.max()),
        }

    try:
        float32 = compare(False)
        bfloat16 = compare(True)
    finally:
        source.train(source_training)
        target.train(target_training)
    return {
        "samples": int(images.shape[0]),
        "equatorial_phases_all_zero": True,
        "functional_equivalence": (
            "features replay bitwise; logits differ only by the removed common "
            "bias gauge; probabilities are tolerance-verified"
        ),
        "float32": float32,
        "bfloat16_autocast": bfloat16,
    }


def validate_equatorial_source_contract(
    checkpoint_sha256: str,
    source_config: Dict,
    source_parameters: Dict,
    source_data: Dict,
    source_summary: Dict,
    target_data: Dict,
) -> Dict:
    """Fail closed on checkpoint, architecture, split, and test-lock drift."""

    if checkpoint_sha256 != ANNULAR_HAAR_BASE_CHECKPOINT_SHA256:
        raise RuntimeError("EQR-16 source checkpoint identity drifted")
    expected_config = {
        "image_size": 96,
        "encoder_variant": "deep-se-haar-morph",
        "physics_variant": "base",
        "physics_summary": "moments-morphology-haar",
        "heads": 4,
        "reuploads": 2,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "include_context": False,
        "core": "quantum",
        "evaluate_test": False,
        "train_subset_protocol": "hash-v1",
        "max_train_per_class": 11667,
    }
    drift = {
        key: (source_config.get(key), expected)
        for key, expected in expected_config.items()
        if source_config.get(key) != expected
    }
    extensions = (
        "tied_mean_dispersion",
        "haar_subtype_residual",
        "haar_subtype_max_envelope",
        "shared_late_refinement",
        "r2_entanglers",
        "equatorial_readout",
        "meridional_readout",
    )
    active_extensions = [
        key for key in extensions if bool(source_config.get(key, False))
    ]
    if drift or active_extensions:
        raise RuntimeError(
            "EQR-16 source is not exact base Haar: "
            f"drift={drift}, active_extensions={active_extensions}"
        )
    if (
        int(source_parameters.get("total", -1)) != 122595
        or int(source_parameters.get("quantum", -1)) != 88
        or int(source_parameters.get("equatorial_readout_trainable", 0)) != 0
        or int(source_parameters.get("meridional_readout_trainable", 0)) != 0
        or int(source_parameters.get("r2_entangler_trainable", 0)) != 0
        or int(source_parameters.get("haar_subtype_residual_trainable", 0)) != 0
        or int(source_parameters.get("dispersion_gate_trainable", 0)) != 0
        or int(
            source_parameters.get("shared_late_refinement_gate_trainable", 0)
        )
        != 0
    ):
        raise RuntimeError("EQR-16 source parameter report is not base Haar")
    if (
        source_data.get("train_size") != 35001
        or source_data.get("train_membership_sha256")
        != OOF_FULL_HALF_MEMBERSHIP_SHA256
        or source_data.get("development_manifest_sha256")
        != OOF_DEVELOPMENT_MANIFEST_SHA256
        or source_data.get("class_names") != ["axion", "cdm", "no_sub"]
        or source_data.get("official_test_cache_opened") is not False
        or "test" in source_data
        or source_summary.get("official_test_evaluated") is not False
        or "test" in source_summary
    ):
        raise RuntimeError("EQR-16 source violates data or official-test locks")
    if (
        target_data.get("train_size") != 35001
        or target_data.get("train_membership_sha256")
        != source_data.get("train_membership_sha256")
        or target_data.get("development_manifest_sha256")
        != source_data.get("development_manifest_sha256")
        or target_data.get("class_names") != source_data.get("class_names")
        or target_data.get("official_test_cache_opened") is not False
        or "test" in target_data
    ):
        raise RuntimeError("EQR-16 target violates split, manifest, or test locks")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "source_parameters": 122595,
        "source_quantum_parameters": 88,
        "same_training_membership": True,
        "same_development_manifest": True,
        "source_official_test_opened": False,
        "target_official_test_opened": False,
    }


def remap_haar_to_meridional_readout(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Add zero XZ-plane phases and remove the softmax-common bias gauge."""

    phase_key = "core.meridional_phases"
    bias_key = "head.classifier.bias"
    if phase_key in source_state:
        raise ValueError("Meridional warm start requires a base-Haar source")
    if phase_key not in target_state:
        raise ValueError("Meridional target is missing its measurement phases")
    expected_keys = set(source_state).union({phase_key})
    if set(target_state) != expected_keys:
        missing = sorted(expected_keys - set(target_state))
        unexpected = sorted(set(target_state) - expected_keys)
        raise ValueError(
            "Base-Haar/meridional state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    required = {
        "core.params": (4, 2, 11),
        phase_key: (4, 4),
        bias_key: (3,),
        "orbit_projection.weight": (8, 372),
        "head.projection.weight": (18, 63),
        "haar_mean": (104,),
        "haar_scale": (104,),
        "morphology_mean": (60,),
        "morphology_scale": (60,),
    }
    for key, shape in required.items():
        state = target_state if key == phase_key else source_state
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(
                f"Base-Haar/meridional source has invalid {key} shape"
            )
    if tuple(target_state[bias_key].shape) != (2,):
        raise ValueError(
            "Meridional target classifier must use a two-parameter bias gauge"
        )

    remapped = {
        key: value.detach().cpu().clone() for key, value in target_state.items()
    }
    for key, value in source_state.items():
        if key == bias_key:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"Base-Haar/meridional tensor shape drift for {key}: "
                f"source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
        remapped[key] = value.detach().cpu().clone()
    phases = remapped[phase_key]
    if not torch.equal(phases, torch.zeros_like(phases)):
        raise ValueError("New meridional phases must contain 16 exact zeros")
    source_bias = source_state[bias_key].detach().cpu()
    remapped[bias_key] = (source_bias[:2] - source_bias[2]).clone()
    return remapped, [
        {
            "key": phase_key,
            "source_shape": None,
            "target_shape": (4, 4),
            "method": "zero-new-d4-meridional-measurement-phases",
            "zero_parameters": 16,
            "settings_per_head": ["local", "R", "R2", "S"],
            "observable": "P(phi)=cos(phi)X+sin(phi)Z",
            "mathematical_feature_equivalence": True,
        },
        {
            "key": bias_key,
            "source_shape": (3,),
            "target_shape": (2,),
            "method": "remove-softmax-common-logit-gauge",
            "removed_parameters": 1,
            "mathematical_probability_equivalence": True,
        },
    ]


def meridional_readout_initialization_record(
    model: D4OrbitClassifier,
) -> Dict:
    """Describe and verify the exact-function XZ-plane readout extension."""

    phases = getattr(model.core, "meridional_phases", None)
    if not model.meridional_readout or phases is None:
        raise ValueError("Model does not enable meridional readout")
    detached = phases.detach()
    all_zero = bool(torch.equal(detached, torch.zeros_like(detached)))
    if tuple(detached.shape) != (4, 4) or not all_zero:
        raise RuntimeError(
            "Meridional initialization no longer replays base Haar"
        )
    return {
        "enabled": True,
        "parameters": int(detached.numel()),
        "initialization": "zeros",
        "all_phases_zero_after_remap": all_zero,
        "settings_per_head": ["local", "R", "R2", "S"],
        "observable": "P(phi)=cos(phi)X+sin(phi)Z",
        "mixed_pair_sector": ["XZ", "ZX"],
        "classifier_bias_gauge_degrees": 2,
        "zero_phase_algebraic_feature_equivalence": True,
        "mixed_precision_replay": "tolerance-verified rather than bitwise exact",
    }


def configure_meridional_optimization_and_report(
    model: D4OrbitClassifier,
    freeze_at_zero: bool,
) -> Dict:
    """Separate allocated size from the frozen meridional optimizer size."""

    phases = getattr(model.core, "meridional_phases", None)
    if (
        not model.meridional_readout
        or phases is None
        or tuple(phases.shape) != (4, 4)
    ):
        raise ValueError(
            "Model is not the exact meridional-16 annular-Haar candidate"
        )
    if not torch.equal(phases.detach(), torch.zeros_like(phases.detach())):
        raise RuntimeError("Meridional phases changed before optimizer configuration")
    allocated = model.parameter_report()
    if (
        int(allocated["total"]) != 122610
        or int(allocated["quantum"]) != 104
        or int(allocated.get("quantum_state_preparation_trainable", -1)) != 88
        or int(allocated.get("meridional_readout_trainable", -1)) != 16
        or int(allocated.get("equatorial_readout_trainable", -1)) != 0
        or int(allocated.get("classifier_bias_trainable", -1)) != 2
        or int(allocated.get("classifier_bias_gauge_degrees", -1)) != 2
    ):
        raise RuntimeError(
            f"Invalid exact-budget meridional-16 candidate: {allocated}"
        )
    phases.requires_grad_(not freeze_at_zero)
    allocated["inference_total"] = 122610
    allocated["optimization_trainable_total"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    allocated["quantum_optimization_trainable"] = sum(
        parameter.numel()
        for parameter in model.core.parameters()
        if parameter.requires_grad
    )
    allocated["meridional_readout_optimization_trainable"] = int(
        phases.numel() if phases.requires_grad else 0
    )
    allocated["meridional_readout_frozen_at_zero"] = bool(freeze_at_zero)
    expected_optimization = 122594 if freeze_at_zero else 122610
    expected_quantum_optimization = 88 if freeze_at_zero else 104
    if (
        allocated["optimization_trainable_total"] != expected_optimization
        or allocated["quantum_optimization_trainable"]
        != expected_quantum_optimization
    ):
        raise RuntimeError(
            f"Meridional-16 optimizer budget drifted: {allocated}"
        )
    return allocated


def meridional_readout_probability_replay(
    source: D4OrbitClassifier,
    target: D4OrbitClassifier,
    images: torch.Tensor,
) -> Dict:
    """Verify zero-phase feature replay and gauge-equivalent probabilities."""

    phases = getattr(target.core, "meridional_phases", None)
    if phases is None or not torch.equal(
        phases.detach(), torch.zeros_like(phases.detach())
    ):
        raise ValueError("Meridional replay requires sixteen zero phases")
    source_training, target_training = source.training, target.training
    source.eval()
    target.eval()

    def compare(autocast_enabled: bool) -> Dict:
        with torch.inference_mode(), torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            source_angles = source.orbit_encode(images)[1]
            source_features = source.core(source_angles).float()
            target_features = target.core(source_angles).float()
            source_logits = source(images).float()
            target_logits = target(images).float()
        feature_difference = (target_features - source_features).abs()
        features_bitwise_equal = bool(torch.equal(target_features, source_features))
        source_probabilities = source_logits.softmax(dim=1)
        target_probabilities = target_logits.softmax(dim=1)
        probability_difference = (
            target_probabilities - source_probabilities
        ).abs()
        predictions_equal = bool(
            torch.equal(target_logits.argmax(dim=1), source_logits.argmax(dim=1))
        )
        probability_tolerance = 2e-6 if not autocast_enabled else 2e-3
        common_shift_tolerance = 2e-6 if not autocast_enabled else 5e-2
        probabilities_equal = bool(
            torch.allclose(
                target_probabilities,
                source_probabilities,
                rtol=probability_tolerance,
                atol=probability_tolerance,
            )
        )
        common_shift = target_logits - source_logits
        shift_spread = (common_shift - common_shift[:, :1]).abs()
        if (
            not features_bitwise_equal
            or not predictions_equal
            or not probabilities_equal
            or float(shift_spread.max()) > common_shift_tolerance
        ):
            mode = "bfloat16" if autocast_enabled else "float32"
            raise RuntimeError(
                f"Zero meridional phases failed {mode} replay: "
                f"feature_max={float(feature_difference.max())}, "
                f"probability_max={float(probability_difference.max())}, "
                f"common_shift_spread={float(shift_spread.max())}"
            )
        return {
            "features_bitwise_equal": features_bitwise_equal,
            "max_feature_absolute_difference": float(feature_difference.max()),
            "predictions_equal": predictions_equal,
            "probabilities_equal_within_tolerance": probabilities_equal,
            "probability_tolerance": probability_tolerance,
            "common_shift_spread_tolerance": common_shift_tolerance,
            "max_probability_absolute_difference": float(
                probability_difference.max()
            ),
            "max_common_shift_spread": float(shift_spread.max()),
        }

    try:
        float32 = compare(False)
        bfloat16 = compare(True)
    finally:
        source.train(source_training)
        target.train(target_training)
    return {
        "samples": int(images.shape[0]),
        "meridional_phases_all_zero": True,
        "functional_equivalence": (
            "features replay bitwise; logits differ only by the removed common "
            "bias gauge; probabilities are tolerance-verified"
        ),
        "float32": float32,
        "bfloat16_autocast": bfloat16,
    }


def validate_meridional_source_contract(
    checkpoint_sha256: str,
    source_config: Dict,
    source_parameters: Dict,
    source_data: Dict,
    source_summary: Dict,
    target_data: Dict,
) -> Dict:
    """Fail closed on meridional checkpoint, split, and test-lock drift."""

    if checkpoint_sha256 != ANNULAR_HAAR_BASE_CHECKPOINT_SHA256:
        raise RuntimeError("Meridional-16 source checkpoint identity drifted")
    expected_config = {
        "image_size": 96,
        "encoder_variant": "deep-se-haar-morph",
        "physics_variant": "base",
        "physics_summary": "moments-morphology-haar",
        "heads": 4,
        "reuploads": 2,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "include_context": False,
        "core": "quantum",
        "evaluate_test": False,
        "train_subset_protocol": "hash-v1",
        "max_train_per_class": 11667,
    }
    drift = {
        key: (source_config.get(key), expected)
        for key, expected in expected_config.items()
        if source_config.get(key) != expected
    }
    extensions = (
        "tied_mean_dispersion",
        "haar_subtype_residual",
        "haar_subtype_max_envelope",
        "shared_late_refinement",
        "r2_entanglers",
        "equatorial_readout",
        "meridional_readout",
    )
    active_extensions = [
        key for key in extensions if bool(source_config.get(key, False))
    ]
    if drift or active_extensions:
        raise RuntimeError(
            "Meridional-16 source is not exact base Haar: "
            f"drift={drift}, active_extensions={active_extensions}"
        )
    if (
        int(source_parameters.get("total", -1)) != 122595
        or int(source_parameters.get("quantum", -1)) != 88
        or int(source_parameters.get("meridional_readout_trainable", 0)) != 0
        or int(source_parameters.get("equatorial_readout_trainable", 0)) != 0
        or int(source_parameters.get("r2_entangler_trainable", 0)) != 0
        or int(source_parameters.get("haar_subtype_residual_trainable", 0)) != 0
        or int(source_parameters.get("dispersion_gate_trainable", 0)) != 0
        or int(
            source_parameters.get("shared_late_refinement_gate_trainable", 0)
        )
        != 0
    ):
        raise RuntimeError(
            "Meridional-16 source parameter report is not base Haar"
        )
    if (
        source_data.get("train_size") != 35001
        or source_data.get("train_membership_sha256")
        != OOF_FULL_HALF_MEMBERSHIP_SHA256
        or source_data.get("development_manifest_sha256")
        != OOF_DEVELOPMENT_MANIFEST_SHA256
        or source_data.get("class_names") != ["axion", "cdm", "no_sub"]
        or source_data.get("official_test_cache_opened") is not False
        or "test" in source_data
        or source_summary.get("official_test_evaluated") is not False
        or "test" in source_summary
    ):
        raise RuntimeError(
            "Meridional-16 source violates data or official-test locks"
        )
    if (
        target_data.get("train_size") != 35001
        or target_data.get("train_membership_sha256")
        != source_data.get("train_membership_sha256")
        or target_data.get("development_manifest_sha256")
        != source_data.get("development_manifest_sha256")
        or target_data.get("class_names") != source_data.get("class_names")
        or target_data.get("official_test_cache_opened") is not False
        or "test" in target_data
    ):
        raise RuntimeError(
            "Meridional-16 target violates split, manifest, or test locks"
        )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "source_parameters": 122595,
        "source_quantum_parameters": 88,
        "same_training_membership": True,
        "same_development_manifest": True,
        "source_official_test_opened": False,
        "target_official_test_opened": False,
    }


def shared_late_refinement_initialization_record(
    model: D4OrbitClassifier,
) -> Dict:
    """Describe and verify the zero-function shared-depth extension."""

    gates = model.encoder.shared_refinement_gates
    if not model.shared_late_refinement or gates is None:
        raise ValueError("Model does not enable shared late refinement")
    detached = gates.detach()
    all_zero = bool(torch.equal(detached, torch.zeros_like(detached)))
    if not all_zero:
        raise RuntimeError(
            "Shared late-refinement initialization no longer replays base Haar"
        )
    return {
        "enabled": True,
        "gate_parameters": int(detached.numel()),
        "gate_initialization": "zeros",
        "all_gates_zero_after_remap": all_zero,
        "zero_gate_exact_base_replay": True,
        "shared_block_applications": [5, 5, 7, 7],
        "inference_path": "shared encoder refinement -> projected angles -> core only",
    }


def shared_late_refinement_exact_replay(
    source: D4OrbitClassifier,
    target: D4OrbitClassifier,
    images: torch.Tensor,
) -> Dict:
    """Require float32 and bfloat16 base-Haar replay before optimization."""

    gates = target.encoder.shared_refinement_gates
    if gates is None or not torch.equal(
        gates.detach(), torch.zeros_like(gates.detach())
    ):
        raise ValueError("Shared late-refinement replay requires four zero gates")
    source_training, target_training = source.training, target.training
    source.eval()
    target.eval()

    def compare(autocast_enabled: bool) -> Dict:
        with torch.inference_mode(), torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            source_logits = source(images)
            target_logits = target(images)
        difference = (target_logits.float() - source_logits.float()).abs()
        exact = bool(torch.equal(target_logits, source_logits))
        predictions_equal = bool(
            torch.equal(target_logits.argmax(dim=1), source_logits.argmax(dim=1))
        )
        if not exact or not predictions_equal:
            mode = "bfloat16" if autocast_enabled else "float32"
            raise RuntimeError(
                f"Zero shared refinement failed exact {mode} base replay: "
                f"max={float(difference.max())}, mean={float(difference.mean())}"
            )
        return {
            "exact_logits": exact,
            "predictions_equal": predictions_equal,
            "max_logit_absolute_difference": float(difference.max()),
            "mean_logit_absolute_difference": float(difference.mean()),
        }

    try:
        float32 = compare(False)
        bfloat16 = compare(True)
    finally:
        source.train(source_training)
        target.train(target_training)
    return {
        "samples": int(images.shape[0]),
        "gates_all_zero": True,
        "float32": float32,
        "bfloat16_autocast": bfloat16,
    }


def haar_subtype_exact_replay(
    source: D4OrbitClassifier,
    target: D4OrbitClassifier,
    images: torch.Tensor,
) -> Dict:
    """Require bitwise source replay before optimizing the new residual."""

    if target.haar_subtype_residual is None:
        raise ValueError("Target has no Haar subtype residual")
    if not torch.equal(
        target.haar_subtype_residual.weight.detach(),
        torch.zeros_like(target.haar_subtype_residual.weight),
    ):
        raise ValueError("Haar subtype replay requires zero residual weights")
    source_training, target_training = source.training, target.training
    source.eval()
    target.eval()
    with torch.inference_mode():
        source_logits = source(images)
        target_logits = target(images)
    source.train(source_training)
    target.train(target_training)
    difference = (target_logits.float() - source_logits.float()).abs()
    exact = bool(torch.equal(target_logits, source_logits))
    predictions_equal = bool(
        torch.equal(target_logits.argmax(dim=1), source_logits.argmax(dim=1))
    )
    if not exact or not predictions_equal:
        raise RuntimeError(
            "Zero Haar subtype residual failed exact base replay: "
            f"max={float(difference.max())}, mean={float(difference.mean())}"
        )
    return {
        "exact_logits": exact,
        "predictions_equal": predictions_equal,
        "samples": int(images.shape[0]),
        "max_logit_absolute_difference": float(difference.max()),
        "mean_logit_absolute_difference": float(difference.mean()),
        "residual_weights_all_zero": True,
    }


def tied_mean_dispersion_initialization_record(
    model: D4OrbitClassifier,
) -> Dict:
    """Describe and verify the zero-function extension used by the candidate."""

    if not model.tied_mean_dispersion or model.dispersion_gates is None:
        raise ValueError("Model does not enable tied mean-dispersion")
    gates = model.dispersion_gates.detach()
    all_zero = bool(torch.equal(gates, torch.zeros_like(gates)))
    if not all_zero:
        raise RuntimeError(
            "Tied mean-dispersion initialization no longer exactly replays the base model"
        )
    return {
        "enabled": True,
        "gate_parameters": int(gates.numel()),
        "gate_initialization": "zeros",
        "all_gates_zero_after_remap": all_zero,
        "zero_gate_exact_base_replay": True,
        "projection_tying": (
            "reuse orbit_projection encoder-mean columns for final-map population std"
        ),
        "inference_path": "dispersion -> projected angles -> invariant core only",
    }


def clone_module_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a freshly initialized module state before any checkpoint load."""

    state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    if not state:
        raise ValueError("Cannot preserve an empty module state")
    return state


def restore_fresh_haar_core(
    model: D4OrbitClassifier,
    fresh_core_state: Dict[str, torch.Tensor],
) -> Dict:
    """Restore only the seeded target core after exact Haar initialization."""

    if model.core_name not in ("quantum", "classical"):
        raise ValueError(
            "Core reinitialization supports only matched quantum/classical cores"
        )
    current = model.core.state_dict()
    if set(current) != set(fresh_core_state):
        raise ValueError(
            "Fresh core state keys differ from the constructed target core"
        )
    for key, value in fresh_core_state.items():
        if tuple(value.shape) != tuple(current[key].shape):
            raise ValueError(
                f"Fresh core tensor shape mismatch for {key}: "
                f"fresh={tuple(value.shape)} current={tuple(current[key].shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Fresh core tensor is nonfinite: {key}")
    checkpoint_core_differed = any(
        not torch.equal(current[key].detach().cpu(), value)
        for key, value in fresh_core_state.items()
    )
    model.core.load_state_dict(fresh_core_state, strict=True)
    restored = model.core.state_dict()
    if any(
        not torch.equal(restored[key].detach().cpu(), value)
        for key, value in fresh_core_state.items()
    ):
        raise RuntimeError("Failed to restore the freshly seeded target core")
    trainable = sum(
        parameter.numel()
        for parameter in model.core.parameters()
        if parameter.requires_grad
    )
    if trainable != 88:
        raise RuntimeError(
            f"Matched annular-Haar core must have 88 trainable values, got {trainable}"
        )
    parameter_names = set(dict(model.core.named_parameters()))
    return {
        "enabled": True,
        "core_architecture": model.core_name,
        "source": "fresh seeded target core captured before any checkpoint load",
        "restored_after": "morphology-kd-to-annular-haar-exact-remap",
        "state_tensors": sorted(fresh_core_state),
        "parameter_tensors": sorted(parameter_names.intersection(fresh_core_state)),
        "persistent_buffer_tensors": sorted(
            set(fresh_core_state) - parameter_names
        ),
        "trainable_parameters": trainable,
        "checkpoint_core_differed_from_fresh": checkpoint_core_differed,
        "restored_exactly": True,
    }


def shape_compatible_prefix_state(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    prefixes: Tuple[str, ...],
) -> Tuple[Dict[str, torch.Tensor], list]:
    """Select exact-shape checkpoint tensors under explicit safe prefixes."""

    compatible: Dict[str, torch.Tensor] = {}
    skipped = []
    for key, value in source_state.items():
        if not key.startswith(prefixes):
            continue
        target = target_state.get(key)
        if target is not None and tuple(value.shape) == tuple(target.shape):
            compatible[key] = value
        else:
            skipped.append(
                {
                    "key": key,
                    "source_shape": list(value.shape),
                    "target_shape": None if target is None else list(target.shape),
                    "reason": "missing-or-shape-mismatch",
                }
            )
    return compatible, skipped


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_morphology_normalization(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
) -> Dict:
    """Fit fixed morphology standardization from clean training images only."""

    if model.morphology_feature_dim != 60:
        raise ValueError("Morphology normalization requires the 60-feature bank")
    summaries = []
    model.physics.eval()
    with torch.inference_mode():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            physics = model.physics(images)
            summaries.append(lens_morphology_summary(physics).cpu())
    values = torch.cat(summaries, dim=0).float()
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    model.set_morphology_normalization(mean, scale)
    return {
        "features": int(values.shape[1]),
        "fit_samples": int(values.shape[0]),
        "source": "clean selected training subset only",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "minimum_scale": float(scale.min()),
        "maximum_scale": float(scale.max()),
    }


def _running_moments_update(
    count: int,
    mean: torch.Tensor | None,
    m2: torch.Tensor | None,
    values: torch.Tensor,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Merge one CPU float64 batch into population Welford moments."""

    batch = values.detach().double().cpu()
    if batch.ndim != 2 or batch.shape[0] == 0:
        raise ValueError("Running moments require a nonempty feature matrix")
    batch_count = int(batch.shape[0])
    batch_mean = batch.mean(dim=0)
    batch_m2 = (batch - batch_mean[None]).square().sum(dim=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    if mean is None or m2 is None or tuple(mean.shape) != tuple(batch_mean.shape):
        raise ValueError("Incompatible running-moment state")
    combined = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / combined)
    merged_m2 = (
        m2
        + batch_m2
        + delta.square() * (count * batch_count / combined)
    )
    return combined, merged_mean, merged_m2


def _finish_running_moments(
    count: int,
    mean: torch.Tensor | None,
    m2: torch.Tensor | None,
    minimum_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if count <= 0 or mean is None or m2 is None:
        raise ValueError("Cannot finish empty running moments")
    variance = (m2 / count).clamp_min(0.0)
    scale = variance.sqrt().clamp_min(minimum_scale)
    return mean.float(), scale.float()


def fit_cross_scale_normalization(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
) -> Dict:
    """Fit the frozen 32-D CSSR bank on clean half-train D4 views only."""

    if not bool(getattr(model, "cross_scale_reupload", False)):
        raise ValueError("Cross-scale normalization requires CSSR")
    gates = getattr(model, "cross_scale_reupload_gates", None)
    if gates is None or tuple(gates.shape) != (4,) or not torch.equal(
        gates.detach(), torch.zeros_like(gates.detach())
    ):
        raise ValueError("Cross-scale normalization requires four zero gates")
    preserved = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key not in {"cross_scale_mean", "cross_scale_scale"}
    }
    image_count = feature_count = 0
    feature_mean = feature_m2 = None
    model.physics.eval()
    with torch.inference_mode():
        for images, _, _ in loader:
            image_count += int(images.shape[0])
            images = images.to(device, non_blocking=True)
            views = d4_views(images)
            batch, group, channels, height, width = views.shape
            if group != 8:
                raise RuntimeError(
                    "Cross-scale normalization requires all eight D4 views"
                )
            flat = views.reshape(
                batch * group, channels, height, width
            ).contiguous(memory_format=torch.channels_last)
            physics = model.physics(flat)
            features = cross_scale_scattering_summary(physics)
            if tuple(features.shape) != (batch * group, 32):
                raise RuntimeError(
                    "Cross-scale scattering summary must produce 32 features "
                    "for every D4 view"
                )
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("Cross-scale normalization features are nonfinite")
            feature_count, feature_mean, feature_m2 = _running_moments_update(
                feature_count, feature_mean, feature_m2, features
            )
    expected_views = image_count * 8
    if feature_count != expected_views:
        raise RuntimeError("Cross-scale normalization omitted a D4 view")
    mean, scale = _finish_running_moments(
        feature_count,
        feature_mean,
        feature_m2,
        minimum_scale=1e-5,
    )
    model.set_cross_scale_normalization(mean, scale)
    if not torch.equal(
        model.cross_scale_reupload_gates.detach(),
        torch.zeros_like(model.cross_scale_reupload_gates.detach()),
    ):
        raise RuntimeError("Cross-scale normalization changed zero gates")
    final_state = model.state_dict()
    drifted = [
        key
        for key, expected in preserved.items()
        if key not in final_state
        or not torch.equal(final_state[key].detach().cpu(), expected)
    ]
    if drifted:
        raise RuntimeError(
            "Cross-scale normalization changed protected base/CSSR state: "
            f"{drifted[:8]}"
        )
    return {
        "schema_version": 1,
        "algorithm_version": "cross-scale-scattering-normalization-v1",
        "features": 32,
        "fit_images": image_count,
        "fit_views": feature_count,
        "d4_views_per_image": 8,
        "source": "clean selected training subset only; all eight D4 views",
        "accumulation": "CPU float64 parallel Welford population moments",
        "minimum_scale_floor": 1e-5,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "minimum_scale": float(scale.min()),
        "maximum_scale": float(scale.max()),
        "base_state_preserved_bitwise": True,
        "zero_gates_preserved": True,
        "validation_samples_used": 0,
        "official_test_samples_used": 0,
    }


def _haar_subtype_feature_definition(index: int) -> Dict:
    if not 0 <= index < 56:
        raise ValueError("Canonical Haar subtype index must be in [0,56)")
    if index < 48:
        scale_index = index // 12
        within_scale = index % 12
        family_index = within_scale // 6
        annulus_index = within_scale % 6
        direction_indices = (0, 1) if family_index == 0 else (2, 3)
        raw_indices = [
            scale_index * 24 + direction * 6 + annulus_index
            for direction in direction_indices
        ]
        return {
            "canonical_index": index,
            "kind": "first-order-annular",
            "offset_pixels": (1, 2, 4, 7)[scale_index],
            "direction_family": ("axis", "diagonal")[family_index],
            "annulus_index": annulus_index,
            "annulus_edges_pixels_at_96": (
                (0, 4),
                (4, 8),
                (8, 12),
                (12, 18),
                (18, 30),
                (30, 68),
            )[annulus_index],
            "raw_haar_indices": raw_indices,
        }
    subtype_index = index - 48
    return {
        "canonical_index": index,
        "kind": "intermittency",
        "offset_pixels": (1, 2, 4, 7)[subtype_index // 2],
        "region": ("inner-4-18", "outer-18-48")[subtype_index % 2],
        "raw_haar_indices": [96 + subtype_index],
    }


def select_haar_subtype_coefficients(
    features: torch.Tensor,
    labels: torch.Tensor,
    selected_count: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Select subtype features using axion/CDM training statistics only."""

    values = features.detach().double().cpu()
    targets = labels.detach().long().cpu()
    if values.ndim != 2 or values.shape[1] != 56:
        raise ValueError("Haar subtype selection requires an (N,56) matrix")
    if targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("Haar subtype labels do not align with features")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Haar subtype selection features must be finite")
    if not 0 < selected_count <= 56:
        raise ValueError("Invalid Haar subtype selection count")
    class_values = [values[targets == label] for label in (0, 1)]
    counts = [int(item.shape[0]) for item in class_values]
    if min(counts) < 2:
        raise ValueError("Haar subtype selection needs at least two samples per class")
    means = [item.mean(dim=0) for item in class_values]
    m2 = [
        (item - mean[None]).square().sum(dim=0)
        for item, mean in zip(class_values, means)
    ]
    pooled_variance = (m2[0] + m2[1]) / (counts[0] + counts[1] - 2)
    pooled_scale = pooled_variance.clamp_min(0.0).sqrt()
    signed_effect = (means[0] - means[1]) / pooled_scale.clamp_min(1e-8)
    score = signed_effect.abs()
    valid = torch.isfinite(score) & torch.isfinite(pooled_scale) & (
        pooled_scale >= 1e-8
    )
    valid_indices = [index for index in range(56) if bool(valid[index])]
    if len(valid_indices) < selected_count:
        raise ValueError("Too few nondegenerate Haar subtype coefficients")
    order = sorted(
        valid_indices, key=lambda index: (-float(score[index]), index)
    )[:selected_count]
    selected = torch.tensor(order, dtype=torch.long)
    center = (0.5 * (means[0] + means[1])).index_select(0, selected).float()
    scale = pooled_scale.index_select(0, selected).clamp_min(1e-5).float()
    report = {
        "class_counts": {"axion": counts[0], "cdm": counts[1]},
        "selection_samples": counts[0] + counts[1],
        "no_sub_selection_samples": 0,
        "score": "absolute pooled-standardized axion-minus-CDM mean",
        "variance": "unbiased within-class pooled variance",
        "tie_break": "descending score, then lower canonical index",
        "minimum_valid_pooled_scale": 1e-8,
        "normalization_scale_floor": 1e-5,
        "selected_indices": order,
        "selected_center": center.tolist(),
        "selected_scale": scale.tolist(),
        "selected_features": [
            {
                **_haar_subtype_feature_definition(index),
                "score": float(score[index]),
                "signed_effect_axion_minus_cdm": float(signed_effect[index]),
                "axion_mean": float(means[0][index]),
                "cdm_mean": float(means[1][index]),
                "pooled_scale": float(pooled_scale[index]),
            }
            for index in order
        ],
    }
    return selected, center, scale, report


def fit_haar_subtype_selection(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
) -> Dict:
    """Fit only residual selection buffers on clean fixed-half training data."""

    residual = model.haar_subtype_residual
    if residual is None or model.haar_summary_dim != 104:
        raise ValueError("Model does not enable the Haar subtype residual")
    preserved = {
        "morphology_mean": model.morphology_mean.detach().clone(),
        "morphology_scale": model.morphology_scale.detach().clone(),
        "haar_mean": model.haar_mean.detach().clone(),
        "haar_scale": model.haar_scale.detach().clone(),
    }
    feature_batches = []
    label_batches = []
    image_count = 0
    model.physics.eval()
    with torch.inference_mode():
        for images, labels, _ in loader:
            image_count += int(images.shape[0])
            images = images.to(device, non_blocking=True)
            views = d4_views(images)
            batch, group, channels, height, width = views.shape
            if group != 8:
                raise RuntimeError("Haar subtype selection requires all eight views")
            flat = views.reshape(batch * group, channels, height, width).contiguous(
                memory_format=torch.channels_last
            )
            physics = model.physics(flat)
            haar = annular_haar_scattering_summary(physics)
            normalized = (haar - model.haar_mean[None]) / model.haar_scale[None]
            invariant = invariant_annular_haar_coefficients(
                normalized.reshape(batch, group, 104)
            )
            feature_batches.append(invariant.cpu())
            label_batches.append(labels.detach().long().cpu())
    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    if len(features) != image_count:
        raise RuntimeError("Haar subtype selection lost training images")
    selected, center, scale, report = select_haar_subtype_coefficients(
        features, labels, selected_count=residual.feature_count
    )
    residual.set_selection(selected, center, scale)
    for name, expected in preserved.items():
        if not torch.equal(getattr(model, name), expected):
            raise RuntimeError(
                f"Haar subtype selection modified preserved fixed buffer: {name}"
            )
    if not torch.equal(residual.weight, torch.zeros_like(residual.weight)):
        raise RuntimeError("Haar subtype selection changed zero residual weights")
    report.update(
        {
            "algorithm_version": "invariant-haar-subtype-v1",
            "canonical_feature_count": 56,
            "selected_feature_count": residual.feature_count,
            "fit_images": image_count,
            "fit_views": image_count * 8,
            "d4_views_per_image": 8,
            "source": "clean selected training subset only",
            "fixed_morphology_normalization_preserved": True,
            "fixed_haar_normalization_preserved": True,
            "residual_weights_all_zero_after_selection": True,
            "validation_samples_used": 0,
            "official_test_samples_used": 0,
        }
    )
    selection_material = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    report["selection_spec_sha256"] = hashlib.sha256(selection_material).hexdigest()
    return report


def fit_haar_morphology_normalization(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
    preserve_morphology: bool = False,
) -> Dict:
    """Fit fixed banks on clean training images and all eight D4 views.

    A morphology-KD warm start already carries the exact train-only 60-D
    normalization used by its projection.  In that case only the new Haar bank
    is fitted, and the copied morphology buffers are asserted bitwise stable.
    """

    if model.morphology_feature_dim != 60 or model.haar_summary_dim != 104:
        raise ValueError(
            "Annular-Haar normalization requires the 60-D morphology and "
            "104-D Haar banks"
        )
    morphology_count = haar_count = image_count = 0
    morphology_mean = morphology_m2 = None
    haar_mean = haar_m2 = None
    preserved_morphology_mean = model.morphology_mean.detach().clone()
    preserved_morphology_scale = model.morphology_scale.detach().clone()
    model.physics.eval()
    with torch.inference_mode():
        for images, _, _ in loader:
            image_count += int(images.shape[0])
            images = images.to(device, non_blocking=True)
            views = d4_views(images)
            batch, group, channels, height, width = views.shape
            if group != 8:
                raise RuntimeError("Annular-Haar normalization requires all 8 D4 views")
            flat = views.reshape(batch * group, channels, height, width).contiguous(
                memory_format=torch.channels_last
            )
            physics = model.physics(flat)
            haar = annular_haar_scattering_summary(physics)
            if not preserve_morphology:
                morphology = lens_morphology_summary(physics)
                morphology_count, morphology_mean, morphology_m2 = (
                    _running_moments_update(
                        morphology_count,
                        morphology_mean,
                        morphology_m2,
                        morphology,
                    )
                )
            haar_count, haar_mean, haar_m2 = _running_moments_update(
                haar_count, haar_mean, haar_m2, haar
            )
    expected_views = image_count * 8
    if haar_count != expected_views:
        raise RuntimeError("Fixed-summary normalization omitted a D4 view")
    if not preserve_morphology and morphology_count != expected_views:
        raise RuntimeError("Morphology normalization omitted a D4 view")
    haar_mean, haar_scale = _finish_running_moments(
        haar_count,
        haar_mean,
        haar_m2,
        minimum_scale=1e-5,
    )
    if preserve_morphology:
        if not torch.equal(model.morphology_mean, preserved_morphology_mean) or not torch.equal(
            model.morphology_scale, preserved_morphology_scale
        ):
            raise RuntimeError("Copied morphology normalization changed before Haar fitting")
        morphology_mean = preserved_morphology_mean.float().cpu()
        morphology_scale = preserved_morphology_scale.float().cpu()
    else:
        morphology_mean, morphology_scale = _finish_running_moments(
            morphology_count,
            morphology_mean,
            morphology_m2,
            minimum_scale=1e-5,
        )
        model.set_morphology_normalization(morphology_mean, morphology_scale)
    model.set_haar_normalization(haar_mean, haar_scale)
    if preserve_morphology and (
        not torch.equal(model.morphology_mean, preserved_morphology_mean)
        or not torch.equal(model.morphology_scale, preserved_morphology_scale)
    ):
        raise RuntimeError("Haar fitting modified copied morphology normalization")

    def report(
        mean: torch.Tensor,
        scale: torch.Tensor,
        fit_views: int,
        source: str,
    ) -> Dict:
        return {
            "features": int(mean.numel()),
            "fit_images": image_count if fit_views else 0,
            "fit_views": fit_views,
            "d4_views_per_image": 8,
            "source": source,
            "accumulation": (
                "CPU float64 parallel Welford population moments"
                if fit_views
                else "preserved checkpoint buffers; no refit"
            ),
            "minimum_scale_floor": 1e-5,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "minimum_scale": float(scale.min()),
            "maximum_scale": float(scale.max()),
        }

    return {
        "morphology": report(
            morphology_mean,
            morphology_scale,
            morphology_count,
            (
                "preserved bitwise from morphology-KD initialization"
                if preserve_morphology
                else "clean selected training subset only; all eight D4 views"
            ),
        ),
        "haar": report(
            haar_mean,
            haar_scale,
            haar_count,
            "clean selected training subset only; all eight D4 views",
        ),
        "morphology_preserved_from_initialization": preserve_morphology,
    }


def translate_batch(
    images: torch.Tensor, max_pixels: int, probability: float = 1.0
) -> torch.Tensor:
    """Apply independent train-only integer translations with zero padding."""

    if max_pixels <= 0:
        return images
    if not 0.0 <= probability <= 1.0:
        raise ValueError("translation probability must be in [0, 1]")
    batch, _, height, width = images.shape
    offsets = torch.randint(
        -max_pixels, max_pixels + 1, (batch, 2), device=images.device
    )
    if probability < 1.0:
        apply = torch.rand(batch, 1, device=images.device) < probability
        offsets = torch.where(apply, offsets, torch.zeros_like(offsets))
    theta = torch.zeros(batch, 2, 3, dtype=images.dtype, device=images.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * offsets[:, 0].to(images.dtype) / width
    theta[:, 1, 2] = -2.0 * offsets[:, 1].to(images.dtype) / height
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def physics_augment_batch(
    images: torch.Tensor,
    photon_probability: float = 0.0,
    photon_count_min: float = 256.0,
    photon_count_max: float = 2048.0,
    psf_probability: float = 0.0,
    read_noise_std: float = 0.0,
) -> torch.Tensor:
    """Apply train-only, D4-isotropic detector and PSF perturbations."""

    result = images.float()
    batch = result.shape[0]
    if psf_probability > 0.0:
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=result.dtype,
            device=result.device,
        ).view(1, 1, 3, 3) / 16.0
        blurred = F.conv2d(F.pad(result, (1, 1, 1, 1), mode="reflect"), kernel)
        apply = torch.rand(batch, 1, 1, 1, device=result.device) < psf_probability
        result = torch.where(apply, blurred, result)
    if photon_probability > 0.0:
        log_min, log_max = math.log(photon_count_min), math.log(photon_count_max)
        counts = torch.exp(
            torch.empty(batch, 1, 1, 1, device=result.device).uniform_(log_min, log_max)
        )
        noisy = torch.poisson(result.clamp_min(0.0) * counts) / counts
        apply = torch.rand(batch, 1, 1, 1, device=result.device) < photon_probability
        result = torch.where(apply, noisy, result)
    if read_noise_std > 0.0:
        result = result + read_noise_std * torch.randn_like(result)
    return result.clamp_min(0.0).to(images.dtype)


def subtype_mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    probability: float,
    alpha: float,
    num_classes: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """Mix axion/CDM examples only, leaving no-substructure images intact.

    Each selected subtype sample is paired with the opposite subtype.  The
    anchor receives the larger interpolation weight, so the hard target still
    has an unambiguous interpretation for training diagnostics.  The returned
    target distribution is used for the actual supervised loss.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError("subtype MixUp probability must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("subtype MixUp alpha must be positive")
    if targets.ndim != 1 or images.shape[0] != targets.shape[0]:
        raise ValueError("MixUp images and targets have incompatible shapes")
    target_probabilities = F.one_hot(
        targets, num_classes=num_classes
    ).to(dtype=images.dtype)
    if probability == 0.0:
        return images, target_probabilities, 0, 1.0

    selected = (targets < 2) & (
        torch.rand(targets.shape[0], device=targets.device) < probability
    )
    partner = torch.full_like(targets, -1)
    for subtype in (0, 1):
        anchors = torch.where(selected & (targets == subtype))[0]
        candidates = torch.where(targets == 1 - subtype)[0]
        if anchors.numel() and candidates.numel():
            choice = torch.randint(
                candidates.numel(), (anchors.numel(),), device=targets.device
            )
            partner[anchors] = candidates[choice]
    valid = partner >= 0
    count = int(valid.sum())
    if not count:
        return images, target_probabilities, 0, 1.0

    concentration = torch.full(
        (count,), float(alpha), dtype=torch.float32, device=images.device
    )
    anchor_weight = torch.distributions.Beta(
        concentration, concentration
    ).sample()
    anchor_weight = torch.maximum(anchor_weight, 1.0 - anchor_weight)
    image_weight = anchor_weight.to(images.dtype).view(-1, 1, 1, 1)
    valid_indices = torch.where(valid)[0]
    partner_indices = partner[valid]
    mixed_images = images.clone()
    mixed_images[valid] = (
        image_weight * images[valid]
        + (1.0 - image_weight) * images[partner_indices]
    )
    label_weight = anchor_weight.to(target_probabilities.dtype).view(-1, 1)
    target_probabilities[valid] = (
        label_weight * target_probabilities[valid]
        + (1.0 - label_weight) * target_probabilities[partner_indices]
    )
    return (
        mixed_images,
        target_probabilities,
        count,
        float(anchor_weight.mean()),
    )


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross entropy for MixUp distributions with optional uniform smoothing."""

    if logits.shape != target_probabilities.shape:
        raise ValueError("soft targets must have the same shape as logits")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label smoothing must be in [0, 1)")
    targets = target_probabilities.float()
    if label_smoothing:
        targets = (
            (1.0 - label_smoothing) * targets
            + label_smoothing / logits.shape[1]
        )
    return -(targets * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


def hierarchical_model_i_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Zero-parameter hierarchy: substructure/no-substructure and axion/CDM."""

    if logits.shape[1] != 3:
        raise ValueError("Model-I hierarchical loss requires three class logits")
    group_logits = torch.stack(
        (torch.logsumexp(logits[:, :2], dim=1), logits[:, 2]), dim=1
    )
    group_targets = (targets == 2).long()
    group_loss = F.cross_entropy(
        group_logits, group_targets, label_smoothing=label_smoothing
    )
    subtype_mask = targets < 2
    if subtype_mask.any():
        subtype_loss = F.cross_entropy(
            logits[subtype_mask, :2],
            targets[subtype_mask],
            label_smoothing=label_smoothing,
        )
    else:
        subtype_loss = logits.sum() * 0.0
    return 0.25 * group_loss + 0.75 * subtype_loss


def knowledge_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | Iterable[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled forward KL from one or more frozen teachers."""

    if temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    student_log_probabilities = F.log_softmax(
        student_logits.float() / temperature, dim=1
    )
    teacher_outputs = (
        (teacher_logits,) if isinstance(teacher_logits, torch.Tensor) else tuple(teacher_logits)
    )
    if not teacher_outputs:
        raise ValueError("at least one distillation teacher output is required")
    teacher_probabilities = torch.stack(
        [
            F.softmax(output.detach().float() / temperature, dim=1)
            for output in teacher_outputs
        ]
    ).mean(dim=0)
    return (temperature**2) * F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    )


def correctness_gated_oof_distillation_loss(
    student_logits: torch.Tensor,
    morphology_logits: torch.Tensor,
    spatial_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distill only from OOF teachers that classify each clean sample correctly."""

    if temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    if (
        student_logits.ndim != 2
        or student_logits.shape[1] != 3
        or morphology_logits.shape != student_logits.shape
        or spatial_logits.shape != student_logits.shape
        or targets.shape != (student_logits.shape[0],)
    ):
        raise ValueError("OOF distillation tensors have incompatible shapes")
    morphology = morphology_logits.detach().float()
    spatial = spatial_logits.detach().float()
    if not bool(torch.isfinite(morphology).all()) or not bool(
        torch.isfinite(spatial).all()
    ):
        raise RuntimeError("OOF teacher logits must be finite")
    morphology_correct = morphology.argmax(dim=1) == targets
    spatial_correct = spatial.argmax(dim=1) == targets
    valid = morphology_correct | spatial_correct
    morphology_weight = morphology_correct.float()
    spatial_weight = spatial_correct.float()
    denominator = (morphology_weight + spatial_weight).clamp_min(1.0)
    teacher_probabilities = (
        morphology_weight[:, None] * F.softmax(morphology / temperature, dim=1)
        + spatial_weight[:, None] * F.softmax(spatial / temperature, dim=1)
    ) / denominator[:, None]
    student_log_probabilities = F.log_softmax(
        student_logits.float() / temperature, dim=1
    )
    per_sample = (temperature**2) * F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=1)
    per_sample = torch.where(valid, per_sample, torch.zeros_like(per_sample))
    return per_sample, valid, teacher_probabilities


def load_oof_distillation_artifact(
    artifact_path: Path,
    report_path: Path,
    expected_indices: np.ndarray,
    development_labels: np.ndarray,
    development_manifest_sha256: str,
) -> Dict:
    """Load a fail-closed OOF artifact and materialize global-index lookup tables."""

    for path, name in ((artifact_path, "OOF artifact"), (report_path, "OOF report")):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{name} is missing or unsafe: {path}")
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):
        raise RuntimeError("OOF report must contain a JSON object")
    artifact_sha256 = file_sha256(artifact_path)
    expected_membership = index_membership_sha256(expected_indices)
    required_report = {
        "schema_version": 1,
        "protocol": "two-fold-correctness-gated-morphology-spatial-v1",
        "artifact_sha256": artifact_sha256,
        "samples": int(len(expected_indices)),
        "train_membership_sha256": expected_membership,
        "development_manifest_sha256": development_manifest_sha256,
        "canonical_development_validation_samples_used": 0,
        "official_test_samples_used": 0,
        "checkpoint_selection": "fixed final epoch only",
    }
    drift = {
        key: (report.get(key), expected)
        for key, expected in required_report.items()
        if report.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"OOF report contract drifted: {drift}")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required_arrays = {
            "indices",
            "labels",
            "morphology_logits",
            "spatial_logits",
            "source_fold",
            "target_probabilities",
            "gate",
        }
        if set(artifact.files) != required_arrays:
            raise RuntimeError(
                f"OOF artifact arrays drifted: {set(artifact.files)}"
            )
        indices = np.asarray(artifact["indices"], dtype=np.int64)
        labels = np.asarray(artifact["labels"], dtype=np.int64)
        morphology_logits = np.asarray(
            artifact["morphology_logits"], dtype=np.float32
        )
        spatial_logits = np.asarray(
            artifact["spatial_logits"], dtype=np.float32
        )
        source_fold = np.asarray(artifact["source_fold"], dtype=np.int64)
        stored_target_probabilities = np.asarray(
            artifact["target_probabilities"], dtype=np.float32
        )
        stored_gate = np.asarray(artifact["gate"], dtype=np.bool_)
    order = np.argsort(indices)
    indices = indices[order]
    labels = labels[order]
    morphology_logits = morphology_logits[order]
    spatial_logits = spatial_logits[order]
    source_fold = source_fold[order]
    stored_target_probabilities = stored_target_probabilities[order]
    stored_gate = stored_gate[order]
    expected = np.sort(np.asarray(expected_indices, dtype=np.int64))
    if not np.array_equal(indices, expected) or len(np.unique(indices)) != len(indices):
        raise RuntimeError("OOF indices do not exactly match full half-training membership")
    if not np.array_equal(labels, np.asarray(development_labels)[indices]):
        raise RuntimeError("OOF labels differ from the development cache")
    if (
        morphology_logits.shape != (len(indices), 3)
        or spatial_logits.shape != (len(indices), 3)
        or not np.isfinite(morphology_logits).all()
        or not np.isfinite(spatial_logits).all()
    ):
        raise RuntimeError("OOF teacher logits have invalid shape or values")
    if source_fold.shape != (len(indices),) or not set(source_fold.tolist()) <= {0, 1}:
        raise RuntimeError("OOF source-fold assignments are invalid")
    morphology_correct = morphology_logits.argmax(1) == labels
    spatial_correct = spatial_logits.argmax(1) == labels
    routing_counts = {
        "both_correct": int((morphology_correct & spatial_correct).sum()),
        "morphology_only_correct": int((morphology_correct & ~spatial_correct).sum()),
        "spatial_only_correct": int((~morphology_correct & spatial_correct).sum()),
        "neither_correct": int((~morphology_correct & ~spatial_correct).sum()),
    }
    if report.get("routing_counts") != routing_counts:
        raise RuntimeError("OOF routing counts disagree with teacher logits")
    recomputed_gate = morphology_correct | spatial_correct
    temperature = float(report.get("temperature", float("nan")))
    if temperature != 2.0:
        raise RuntimeError("OOF target temperature drifted")
    def numpy_temperature_softmax(values: np.ndarray) -> np.ndarray:
        shifted = values.astype(np.float64) / temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)

    morphology_probability = numpy_temperature_softmax(morphology_logits)
    spatial_probability = numpy_temperature_softmax(spatial_logits)
    denominator = (
        morphology_correct.astype(np.float32)
        + spatial_correct.astype(np.float32)
    ).clip(1.0)
    recomputed_target = (
        morphology_correct[:, None] * morphology_probability
        + spatial_correct[:, None] * spatial_probability
    ) / denominator[:, None]
    recomputed_target[~recomputed_gate] = np.eye(3, dtype=np.float32)[
        labels[~recomputed_gate]
    ]
    if (
        stored_target_probabilities.shape != (len(indices), 3)
        or stored_gate.shape != (len(indices),)
        or not np.array_equal(stored_gate, recomputed_gate)
        or not np.allclose(
            stored_target_probabilities,
            recomputed_target,
            rtol=2e-6,
            atol=2e-6,
        )
        or report.get("gate_sha256")
        != hashlib.sha256(stored_gate.astype(np.uint8).tobytes()).hexdigest()
        or report.get("target_probability_content_sha256")
        != hashlib.sha256(
            np.ascontiguousarray(stored_target_probabilities.astype("<f4")).tobytes()
        ).hexdigest()
    ):
        raise RuntimeError("Stored OOF gates or target probabilities failed replay")
    table_shape = (len(development_labels), 3)
    morphology_table = torch.full(table_shape, float("nan"), dtype=torch.float32)
    spatial_table = torch.full(table_shape, float("nan"), dtype=torch.float32)
    table_indices = torch.tensor(indices.tolist(), dtype=torch.long)
    morphology_table[table_indices] = torch.tensor(
        morphology_logits.tolist(), dtype=torch.float32
    )
    spatial_table[table_indices] = torch.tensor(
        spatial_logits.tolist(), dtype=torch.float32
    )
    return {
        "morphology_logits": morphology_table,
        "spatial_logits": spatial_table,
        "report": report,
        "routing_counts": routing_counts,
        "artifact_sha256": artifact_sha256,
        "report_sha256": file_sha256(report_path),
    }


@torch.no_grad()
def evaluate(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
    class_names,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all, logits_all, indices_all = [], [], []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
        labels_all.append(labels.numpy())
        logits_all.append(logits.float().cpu().numpy())
        indices_all.append(indices.numpy())
    labels_np = np.concatenate(labels_all)
    logits_np = np.concatenate(logits_all)
    indices_np = np.concatenate(indices_all)
    metrics = classification_metrics(labels_np, logits_np, list(class_names))
    return metrics, labels_np, logits_np, indices_np


@torch.no_grad()
def evaluate_parallel_branches(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
    class_names,
) -> Tuple[Dict, Dict]:
    """Evaluate each branch of a jointly trained parallel-core model."""

    model.eval()
    labels_all, branch_a_all, branch_b_all = [], [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, auxiliary = model(images, return_aux=True)
        logits_a, logits_b = auxiliary["branch_logits"]
        labels_all.append(labels.numpy())
        branch_a_all.append(logits_a.float().cpu().numpy())
        branch_b_all.append(logits_b.float().cpu().numpy())
    labels_np = np.concatenate(labels_all)
    return (
        classification_metrics(
            labels_np, np.concatenate(branch_a_all), list(class_names)
        ),
        classification_metrics(
            labels_np, np.concatenate(branch_b_all), list(class_names)
        ),
    )


@torch.no_grad()
def symmetry_audit(
    model: D4OrbitClassifier, loader, device: torch.device, sample_limit: int = 16
) -> Dict:
    model.eval()
    images = next(iter(loader))[0][:sample_limit]
    images = images.to(device).contiguous(memory_format=torch.channels_last)
    base_logits, base_aux = model(images, return_aux=True)
    audit = {}
    all_logit_differences = []
    for element in D4_ELEMENTS:
        logits, aux = model(d4_transform(images, *element), return_aux=True)
        permutation = right_regular_permutation(element).to(device)
        angle_expected = base_aux["angles"].index_select(-1, permutation)
        angle_diff = (aux["angles"] - angle_expected).abs().float().reshape(-1)
        logit_diff = (logits - base_logits).abs().float().reshape(-1)
        all_logit_differences.append(logit_diff)
        record = {
            "angle_regular_max": float(angle_diff.max()),
            "logit_invariant_max": float(logit_diff.max()),
            "logit_invariant_mean": float(logit_diff.mean()),
        }
        if base_aux["equivariant"] is not None:
            for name in ("z", "x"):
                expected = base_aux["equivariant"][name].index_select(-1, permutation)
                difference = (aux["equivariant"][name] - expected).abs().float()
                record[f"circuit_{name}_regular_max"] = float(difference.max())
        audit[f"r{element[0]}s{element[1]}"] = record
    combined = torch.cat(all_logit_differences).cpu().numpy()
    audit["summary"] = {
        "max": float(combined.max()),
        "mean": float(combined.mean()),
        "p99": float(np.quantile(combined, 0.99)),
        "samples": int(len(images)),
        "actions": 8,
    }
    return audit


def count_by_class(indices: np.ndarray, labels: np.ndarray, class_names) -> Dict[str, int]:
    return {name: int((labels[indices] == i).sum()) for i, name in enumerate(class_names)}


def select_model_i_subtype_task(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    labels: np.ndarray,
    class_names,
) -> Tuple[np.ndarray, np.ndarray, list, Dict]:
    """Filter a fixed Model-I split to axion/CDM without reranking samples.

    The parent split and reduced-data membership must be chosen before this
    function is called.  Consequently, the specialist sees exactly the two
    subtype strata already present in the declared three-class half subset,
    rather than receiving a larger binary-specific data allowance.
    """

    names = list(class_names)
    if names[:3] != ["axion", "cdm", "no_sub"]:
        raise ValueError(
            "Model-I subtype specialist requires class order axion, cdm, no_sub"
        )
    train_indices = np.asarray(train_indices)
    val_indices = np.asarray(val_indices)
    parent_train_membership = index_membership_sha256(train_indices)
    parent_val_membership = index_membership_sha256(val_indices)
    subtype_train = train_indices[labels[train_indices] < 2]
    subtype_val = val_indices[labels[val_indices] < 2]
    if len(subtype_train) == 0 or len(subtype_val) == 0:
        raise ValueError("Subtype specialist split is empty")
    if not np.all(np.isin(labels[subtype_train], (0, 1))):
        raise RuntimeError("Subtype training split contains a non-subtype label")
    if not np.all(np.isin(labels[subtype_val], (0, 1))):
        raise RuntimeError("Subtype validation split contains a non-subtype label")
    report = {
        "task": "axion-vs-cdm-specialist",
        "parent_class_names": names,
        "parent_train_size": int(len(train_indices)),
        "parent_validation_size": int(len(val_indices)),
        "parent_train_membership_sha256": parent_train_membership,
        "parent_validation_membership_sha256": parent_val_membership,
        "subtype_train_membership_sha256": index_membership_sha256(subtype_train),
        "subtype_validation_membership_sha256": index_membership_sha256(subtype_val),
    }
    return subtype_train, subtype_val, names[:2], report


def optimizer_parameter_groups(
    model: D4OrbitClassifier,
) -> Tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition every trainable value once; CSSR gates belong to the head."""

    encoder_modules = [model.encoder, model.physics]
    head_modules = [model.orbit_projection, model.head]
    if model.context_projection is not None:
        head_modules.append(model.context_projection)
    if model.haar_subtype_residual is not None:
        head_modules.append(model.haar_subtype_residual)
    encoder_parameters = [
        parameter
        for module in encoder_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for module in head_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if model.dispersion_gates is not None and model.dispersion_gates.requires_grad:
        head_parameters.append(model.dispersion_gates)
    cross_scale_gates = getattr(model, "cross_scale_reupload_gates", None)
    if cross_scale_gates is not None and cross_scale_gates.requires_grad:
        head_parameters.append(cross_scale_gates)
    core_parameters = [
        parameter
        for parameter in model.core.parameters()
        if parameter.requires_grad
    ]
    shared_gates = getattr(model.encoder, "shared_refinement_gates", None)
    if shared_gates is not None and id(shared_gates) not in {
        id(parameter) for parameter in encoder_parameters
    }:
        raise RuntimeError(
            "Shared late-refinement gates are absent from the encoder optimizer group"
        )
    groups = (encoder_parameters, head_parameters, core_parameters)
    grouped_ids = [id(parameter) for group in groups for parameter in group]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("A trainable parameter appears in multiple optimizer groups")
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if set(grouped_ids) != trainable_ids:
        raise RuntimeError(
            "Optimizer groups do not account for every trainable model parameter"
        )
    if cross_scale_gates is not None and cross_scale_gates.requires_grad and id(
        cross_scale_gates
    ) not in {id(parameter) for parameter in head_parameters}:
        raise RuntimeError("CSSR gates are absent from the head optimizer group")
    return groups


def main() -> None:
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    validate_paired_spatial_training_contract(args)
    if not torch.cuda.is_available():
        raise RuntimeError("This research entry point requires a Kubeflow CUDA job")
    validate_meridional_flag_contract(args)
    validate_haar_subtype_freeze_contract(args)
    validate_cross_scale_reupload_contract(args)
    if not 0.0 <= args.hierarchical_loss_weight <= 1.0:
        raise ValueError("hierarchical loss weight must be in [0, 1]")
    if not 0.0 <= args.branch_loss_weight <= 1.0:
        raise ValueError("branch loss weight must be in [0, 1]")
    if args.branch_loss_weight and args.core not in ("hybrid", "classical-fusion"):
        raise ValueError("branch loss requires a hybrid or classical-fusion core")
    if not 0.0 <= args.distillation_weight <= 1.0:
        raise ValueError("distillation weight must be in [0, 1]")
    if args.distillation_temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    if bool(args.oof_distillation_artifact) != bool(args.oof_distillation_report):
        raise ValueError("OOF distillation requires both artifact and report")
    if args.distillation_teacher_checkpoint and args.oof_distillation_artifact:
        raise ValueError("online and OOF distillation teachers are mutually exclusive")
    distillation_source = bool(
        args.distillation_teacher_checkpoint or args.oof_distillation_artifact
    )
    if distillation_source and args.distillation_weight <= 0.0:
        raise ValueError("a distillation source requires positive distillation weight")
    if args.distillation_weight and not distillation_source:
        raise ValueError("positive distillation weight requires a teacher source")
    if args.oof_distillation_artifact and args.subtype_mixup_probability:
        raise ValueError("OOF distillation cannot be combined with subtype MixUp")
    if args.oof_distillation_artifact and args.hierarchical_loss_weight:
        raise ValueError("OOF distillation cannot be combined with hierarchical loss")
    if args.oof_distillation_artifact:
        if (
            not args.fixed_final_validation_only
            or args.evaluate_test
            or not args.deterministic
        ):
            raise ValueError(
                "OOF distillation requires deterministic fixed-final development "
                "validation and no test"
            )
        if (
            args.encoder_variant != "deep-se-haar-morph"
            or args.physics_variant != "base"
            or args.physics_summary != "moments-morphology-haar"
            or args.core != "quantum"
            or args.include_context
            or args.quantum_encoding != "angle"
            or args.observable_readout != "pair"
            or args.heads != 4
            or args.reuploads != 2
            or args.tied_mean_dispersion
            or args.haar_subtype_residual
            or args.haar_subtype_max_envelope
            or args.shared_late_refinement
            or args.r2_entanglers
            or args.equatorial_readout
            or args.meridional_readout
            or args.train_subset_protocol != "hash-v1"
            or args.max_train_per_class != 11667
        ):
            raise ValueError("OOF distillation requires the exact 122595/q88 Haar student")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    for name, probability in (
        ("photon noise", args.photon_noise_probability),
        ("PSF blur", args.psf_blur_probability),
        ("subtype MixUp", args.subtype_mixup_probability),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} probability must be in [0, 1]")
    if args.photon_count_min <= 0 or args.photon_count_max < args.photon_count_min:
        raise ValueError("invalid photon-count range")
    if args.read_noise_std < 0:
        raise ValueError("read-noise standard deviation cannot be negative")
    if args.subtype_mixup_alpha <= 0.0:
        raise ValueError("subtype MixUp alpha must be positive")
    if args.subtype_mixup_probability and args.hierarchical_loss_weight:
        raise ValueError("subtype MixUp and hierarchical loss cannot be combined")
    if args.validation_interval <= 0:
        raise ValueError("validation interval must be positive")
    if args.fixed_final_validation_only and args.patience <= args.epochs:
        raise ValueError("fixed-final validation requires patience greater than epochs")
    if args.subtype_specialist and args.evaluate_test:
        raise ValueError(
            "Subtype-specialist jobs are validation-only and must not open the test set"
        )
    if args.subtype_specialist and args.hierarchical_loss_weight:
        raise ValueError(
            "The three-class hierarchical loss is not defined for a binary specialist"
        )
    if args.freeze_r2_entanglers_at_zero and not args.r2_entanglers:
        raise ValueError(
            "--freeze-r2-entanglers-at-zero requires --r2-entanglers"
        )
    if (
        args.freeze_equatorial_readout_at_zero
        and not args.equatorial_readout
    ):
        raise ValueError(
            "--freeze-equatorial-readout-at-zero requires --equatorial-readout"
        )
    if args.equatorial_readout:
        if (
            args.tied_mean_dispersion
            or args.haar_subtype_residual
            or args.shared_late_refinement
            or args.r2_entanglers
            or args.meridional_readout
            or args.reinitialize_core_after_init
        ):
            raise ValueError(
                "Equatorial readout is mutually exclusive with other "
                "annular-Haar extensions"
            )
        if args.subtype_specialist or args.evaluate_test:
            raise ValueError(
                "Equatorial readout is three-class development-only"
            )
        if not args.init_full_checkpoint:
            raise ValueError(
                "--equatorial-readout requires a full base-Haar checkpoint"
            )
        if args.init_backbone_checkpoint or args.init_compatible_backbone_checkpoint:
            raise ValueError("Equatorial readout forbids other checkpoint modes")
        if (
            args.encoder_variant != "deep-se-haar-morph"
            or args.physics_variant != "base"
            or args.physics_summary != "moments-morphology-haar"
            or args.include_context
            or args.heads != 4
            or args.reuploads != 2
            or args.core != "quantum"
            or args.quantum_encoding != "angle"
            or args.observable_readout != "pair"
        ):
            raise ValueError(
                "Equatorial readout requires the exact four-head/two-reupload "
                "quantum annular-Haar architecture"
            )
        if (
            args.train_subset_protocol != "hash-v1"
            or args.max_train_per_class != 11667
        ):
            raise ValueError(
                "Equatorial readout requires the fixed hash-v1 half-training subset"
            )
    if args.r2_entanglers:
        if (
            args.tied_mean_dispersion
            or args.haar_subtype_residual
            or args.shared_late_refinement
            or args.equatorial_readout
            or args.meridional_readout
            or args.reinitialize_core_after_init
        ):
            raise ValueError(
                "R2 entanglers are mutually exclusive with other annular-Haar extensions"
            )
        if args.subtype_specialist or args.evaluate_test:
            raise ValueError("R2 entanglers are three-class development-only")
        if not args.init_full_checkpoint:
            raise ValueError("--r2-entanglers requires a full base-Haar checkpoint")
        if args.init_backbone_checkpoint or args.init_compatible_backbone_checkpoint:
            raise ValueError("R2 entanglers forbid other checkpoint modes")
        if (
            args.encoder_variant != "deep-se-haar-morph"
            or args.physics_variant != "base"
            or args.physics_summary != "moments-morphology-haar"
            or args.include_context
            or args.heads != 4
            or args.reuploads != 2
            or args.core != "quantum"
            or args.quantum_encoding != "angle"
            or args.observable_readout != "pair"
        ):
            raise ValueError(
                "R2 entanglers require the exact four-head/two-reupload "
                "quantum annular-Haar architecture"
            )
        if (
            args.train_subset_protocol != "hash-v1"
            or args.max_train_per_class != 11667
        ):
            raise ValueError(
                "R2 entanglers require the fixed hash-v1 half-training subset"
            )
    if args.meridional_readout:
        if args.subtype_specialist or args.evaluate_test:
            raise ValueError(
                "Meridional readout is three-class development-only"
            )
        if not args.init_full_checkpoint:
            raise ValueError(
                "--meridional-readout requires a full base-Haar checkpoint"
            )
        if args.init_backbone_checkpoint or args.init_compatible_backbone_checkpoint:
            raise ValueError("Meridional readout forbids other checkpoint modes")
        if (
            args.encoder_variant != "deep-se-haar-morph"
            or args.physics_variant != "base"
            or args.physics_summary != "moments-morphology-haar"
            or args.include_context
            or args.heads != 4
            or args.reuploads != 2
            or args.core != "quantum"
            or args.quantum_encoding != "angle"
            or args.observable_readout != "pair"
        ):
            raise ValueError(
                "Meridional readout requires the exact four-head/two-reupload "
                "quantum annular-Haar architecture"
            )
        if (
            args.train_subset_protocol != "hash-v1"
            or args.max_train_per_class != 11667
        ):
            raise ValueError(
                "Meridional readout requires the fixed hash-v1 half-training subset"
            )
    if args.shared_late_refinement:
        if (
            args.tied_mean_dispersion
            or args.haar_subtype_residual
            or args.reinitialize_core_after_init
        ):
            raise ValueError(
                "--shared-late-refinement, --tied-mean-dispersion, "
                "--haar-subtype-residual, and core reinitialization are "
                "mutually exclusive"
            )
        if args.subtype_specialist or args.evaluate_test:
            raise ValueError(
                "Shared late refinement is three-class development-only"
            )
        if not args.init_full_checkpoint:
            raise ValueError(
                "--shared-late-refinement requires a full base-Haar checkpoint"
            )
        if args.init_backbone_checkpoint or args.init_compatible_backbone_checkpoint:
            raise ValueError(
                "Shared late refinement forbids other checkpoint modes"
            )
        if (
            args.encoder_variant != "deep-se-haar-morph"
            or args.physics_variant != "base"
            or args.physics_summary != "moments-morphology-haar"
            or args.include_context
            or args.heads != 4
            or args.reuploads != 2
            or args.core not in ("quantum", "classical")
            or args.quantum_encoding != "angle"
            or args.observable_readout != "pair"
        ):
            raise ValueError(
                "Shared late refinement requires the exact base-Haar architecture"
            )
        if (
            args.train_subset_protocol != "hash-v1"
            or args.max_train_per_class != 11667
        ):
            raise ValueError(
                "Shared late refinement requires the fixed hash-v1 half-training subset"
            )
    if args.tied_mean_dispersion and (
        args.encoder_variant != "deep-se-haar-morph"
        or args.physics_summary != "moments-morphology-haar"
    ):
        raise ValueError(
            "--tied-mean-dispersion requires the exact annular-Haar candidate"
        )
    if args.tied_mean_dispersion and (
        args.heads != 4
        or args.reuploads != 2
        or args.core not in ("quantum", "classical")
        or args.quantum_encoding != "angle"
        or args.observable_readout != "pair"
    ):
        raise ValueError(
            "--tied-mean-dispersion requires the four-head, two-reupload "
            "angle/pair quantum or classical candidate"
        )
    if args.haar_subtype_residual:
        if args.tied_mean_dispersion:
            raise ValueError(
                "--haar-subtype-residual and --tied-mean-dispersion are mutually exclusive"
            )
        if args.reinitialize_core_after_init:
            raise ValueError(
                "Haar subtype exact replay forbids core reinitialization"
            )
        if args.subtype_specialist or args.evaluate_test:
            raise ValueError(
                "Haar subtype selection is three-class development-only"
            )
        if not args.init_full_checkpoint:
            raise ValueError(
                "--haar-subtype-residual requires a full base-Haar checkpoint"
            )
        if args.init_backbone_checkpoint or args.init_compatible_backbone_checkpoint:
            raise ValueError(
                "Haar subtype residual forbids other checkpoint modes"
            )
        if (
            args.encoder_variant != "deep-se-haar-morph"
            or args.physics_variant != "base"
            or args.physics_summary != "moments-morphology-haar"
            or args.include_context
            or args.heads != 4
            or args.reuploads != 2
            or args.core not in ("quantum", "classical")
            or args.quantum_encoding != "angle"
            or args.observable_readout != "pair"
        ):
            raise ValueError(
                "Haar subtype residual requires the exact base-Haar architecture"
            )
        if (
            args.train_subset_protocol != "hash-v1"
            or args.max_train_per_class != 11667
        ):
            raise ValueError(
                "Haar subtype residual requires the fixed hash-v1 half-training subset"
            )
    if args.haar_subtype_max_envelope and not args.haar_subtype_residual:
        raise ValueError(
            "--haar-subtype-max-envelope requires --haar-subtype-residual"
        )
    if args.freeze_haar_subtype_residual_at_zero and (
        args.haar_subtype_max_envelope
        or args.freeze_base_for_haar_subtype_residual
    ):
        raise ValueError(
            "The zero-residual paired control forbids max-envelope and "
            "frozen-base modes"
        )
    if args.freeze_base_for_haar_subtype_residual and not (
        args.haar_subtype_residual and args.haar_subtype_max_envelope
    ):
        raise ValueError(
            "--freeze-base-for-haar-subtype-residual requires the Haar subtype "
            "residual and max envelope"
        )
    if args.reinitialize_core_after_init:
        if not args.init_full_checkpoint:
            raise ValueError(
                "--reinitialize-core-after-init requires --init-full-checkpoint"
            )
        if args.init_backbone_checkpoint or args.init_compatible_backbone_checkpoint:
            raise ValueError(
                "Core reinitialization is incompatible with other checkpoint modes"
            )
        if (
            args.encoder_variant != "deep-se-haar-morph"
            or args.physics_summary != "moments-morphology-haar"
        ):
            raise ValueError(
                "Core reinitialization is restricted to the exact annular-Haar model"
            )
        if args.core not in ("quantum", "classical"):
            raise ValueError(
                "Core reinitialization requires a quantum or matched classical core"
            )
    if args.oof_teacher_fold_index is not None:
        if args.evaluate_test or args.subtype_specialist:
            raise ValueError("OOF teachers cannot evaluate test or subtype-specialist data")
        if (
            args.train_subset_protocol != "hash-v1"
            or args.max_train_per_class != 11667
            or args.max_val_per_class is not None
            or args.split_seed != 42
            or not math.isclose(args.val_fraction, 0.20, rel_tol=0.0, abs_tol=1e-12)
            or args.image_size != 96
        ):
            raise ValueError(
                "OOF teachers require the exact hash-v1 half-training membership"
            )
        if any(
            (
                args.init_backbone_checkpoint,
                args.init_compatible_backbone_checkpoint,
                args.init_full_checkpoint,
                args.reinitialize_core_after_init,
                args.distillation_teacher_checkpoint,
                args.oof_distillation_artifact,
            )
        ):
            raise ValueError(
                "OOF teachers must start fresh and cannot use teachers or initializers"
            )
        if args.patience <= args.epochs:
            raise ValueError(
                "OOF teachers require patience greater than epochs so the fixed "
                "final checkpoint cannot be selected by held-out labels"
            )
        if not args.save_last_validation_predictions:
            raise ValueError(
                "OOF teachers must save fixed-final validation predictions"
            )
        if not args.deterministic:
            raise ValueError("OOF teachers require deterministic training mode")
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"RUNTIME torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"GPU {torch.cuda.get_device_name(0)} count={torch.cuda.device_count()}", flush=True)
    print(f"CONFIG {json.dumps(vars(args), sort_keys=True)}", flush=True)
    atomic_json(output_dir / "config.json", vars(args))

    cache_root = Path(args.cache_root) / f"model_i_{args.image_size}_v1"
    development_cache = cache_root / "development"
    test_cache = cache_root / "test"
    development_metadata = prepare_cache(
        args.development_root,
        development_cache,
        args.image_size,
        device,
        io_workers=args.io_workers,
    )
    class_names = list(development_metadata["classes"])
    if args.haar_subtype_residual and class_names != ["axion", "cdm", "no_sub"]:
        raise RuntimeError(
            "Haar subtype residual requires canonical axion/cdm/no_sub class order"
        )
    test_metadata = None
    disjoint = None
    if args.evaluate_test:
        test_metadata = prepare_cache(
            args.test_root,
            test_cache,
            args.image_size,
            device,
            io_workers=args.io_workers,
        )
        disjoint = verify_cache_disjoint(development_cache, test_cache)
        if test_metadata["classes"] != class_names:
            raise RuntimeError("Development/test class order differs")

    labels = np.load(development_cache / "labels.npy")
    split_name = f"split_seed{args.split_seed}_val{args.val_fraction:.4f}.npz"
    train_indices, val_indices = fixed_stratified_split(
        labels,
        cache_root / split_name,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )
    development_manifest_sha256 = hashlib.sha256(
        (development_cache / "manifest.csv").read_bytes()
    ).hexdigest()
    if args.train_subset_protocol == "hash-v1":
        train_indices = hash_ranked_subset(
            train_indices,
            labels,
            args.max_train_per_class,
            development_manifest_sha256,
        )
    else:
        train_indices = deterministic_subset(
            train_indices, labels, args.max_train_per_class, seed=args.seed + 1000
        )
    val_indices = deterministic_subset(
        val_indices, labels, args.max_val_per_class, seed=args.split_seed + 2000
    )
    canonical_val_indices = np.asarray(val_indices, dtype=np.int64).copy()
    oof_teacher_report = None
    oof_full_train_indices = None
    if (
        args.haar_subtype_residual
        or args.r2_entanglers
        or args.equatorial_readout
        or args.meridional_readout
    ):
        verify_oof_student_data_contract(
            development_cache,
            development_metadata,
            class_names,
            train_indices,
            canonical_val_indices,
            development_manifest_sha256,
        )
    if args.oof_distillation_artifact:
        verify_oof_student_data_contract(
            development_cache,
            development_metadata,
            class_names,
            train_indices,
            canonical_val_indices,
            development_manifest_sha256,
        )
    if args.oof_teacher_fold_index is not None:
        cache_fingerprints = {
            "images.npy": file_sha256(development_cache / "images.npy"),
            "labels.npy": file_sha256(development_cache / "labels.npy"),
            "metadata.json": file_sha256(development_cache / "metadata.json"),
            "manifest.csv": development_manifest_sha256,
        }
        expected_cache_fingerprints = {
            "images.npy": OOF_DEVELOPMENT_IMAGES_SHA256,
            "labels.npy": OOF_DEVELOPMENT_LABELS_SHA256,
            "metadata.json": OOF_DEVELOPMENT_METADATA_SHA256,
            "manifest.csv": OOF_DEVELOPMENT_MANIFEST_SHA256,
        }
        if cache_fingerprints != expected_cache_fingerprints:
            raise RuntimeError(
                "OOF development cache identity drifted: "
                f"actual={cache_fingerprints} expected={expected_cache_fingerprints}"
            )
        if (
            class_names != ["axion", "cdm", "no_sub"]
            or development_metadata.get("class_counts")
            != {"axion": 28897, "cdm": 29772, "no_sub": 28856}
        ):
            raise RuntimeError("OOF class order or development counts drifted")
        oof_full_train_indices = np.asarray(train_indices, dtype=np.int64).copy()
        if (
            index_membership_sha256(oof_full_train_indices)
            != OOF_FULL_HALF_MEMBERSHIP_SHA256
            or index_membership_sha256(canonical_val_indices)
            != OOF_CANONICAL_VAL_MEMBERSHIP_SHA256
            or np.intersect1d(
                oof_full_train_indices, canonical_val_indices, assume_unique=True
            ).size
        ):
            raise RuntimeError("OOF canonical parent or validation membership drifted")
        folds = stratified_hash_folds(
            oof_full_train_indices,
            labels,
            2,
            development_manifest_sha256,
        )
        train_fold = int(args.oof_teacher_fold_index)
        prediction_fold = 1 - train_fold
        train_indices = folds[train_fold]
        val_indices = folds[prediction_fold]
        oof_teacher_report = {
            "protocol": "stratified-hash-two-fold-v1",
            "fold_count": 2,
            "training_fold": train_fold,
            "prediction_fold": prediction_fold,
            "full_half_train_size": int(len(oof_full_train_indices)),
            "full_half_membership_sha256": index_membership_sha256(
                oof_full_train_indices
            ),
            "training_fold_membership_sha256": index_membership_sha256(
                train_indices
            ),
            "prediction_fold_membership_sha256": index_membership_sha256(
                val_indices
            ),
            "fold_membership_sha256": [
                index_membership_sha256(fold) for fold in folds
            ],
            "training_fold_counts": {
                class_names[label]: int((labels[train_indices] == label).sum())
                for label in range(len(class_names))
            },
            "prediction_fold_counts": {
                class_names[label]: int((labels[val_indices] == label).sum())
                for label in range(len(class_names))
            },
            "canonical_development_validation_size": int(
                len(canonical_val_indices)
            ),
            "canonical_development_validation_samples_used": 0,
            "official_test_samples_used": 0,
            "checkpoint_selection_for_oof": "fixed final epoch only",
            "development_cache_sha256": cache_fingerprints,
        }
    subtype_task_report = None
    if args.subtype_specialist:
        train_indices, val_indices, class_names, subtype_task_report = (
            select_model_i_subtype_task(
                train_indices, val_indices, labels, class_names
            )
        )
    split_payload = {"train": train_indices, "val": val_indices}
    if oof_teacher_report is not None:
        split_payload.update(
            {
                "full_half_train": oof_full_train_indices,
                "canonical_val_unused": canonical_val_indices,
            }
        )
    np.savez(output_dir / "split_indices.npz", **split_payload)
    data_report = {
        "class_names": class_names,
        "development": development_metadata,
        "train_size": int(len(train_indices)),
        "validation_size": int(len(val_indices)),
        "train_counts": count_by_class(train_indices, labels, class_names),
        "validation_counts": count_by_class(val_indices, labels, class_names),
        "train_subset_protocol": args.train_subset_protocol,
        "train_membership_sha256": index_membership_sha256(train_indices),
        "development_manifest_sha256": development_manifest_sha256,
        "official_test_locked_during_selection": not args.evaluate_test,
        "official_test_cache_opened": bool(args.evaluate_test),
    }
    if subtype_task_report is not None:
        data_report["subtype_specialist"] = subtype_task_report
    if oof_teacher_report is not None:
        data_report["oof_teacher"] = oof_teacher_report
    if test_metadata is not None:
        data_report["test"] = test_metadata
        data_report["digest_disjoint"] = disjoint
    atomic_json(output_dir / "data_report.json", data_report)
    print(f"DATA_REPORT {json.dumps(data_report, sort_keys=True)}", flush=True)

    train_loader = make_loader(
        CachedNPYDataset(development_cache, train_indices),
        args.batch_size,
        True,
        args.workers,
        args.seed,
    )
    val_loader = make_loader(
        CachedNPYDataset(development_cache, val_indices),
        args.batch_size,
        False,
        args.workers,
        args.split_seed,
    )

    model = D4OrbitClassifier(
        num_classes=len(class_names),
        heads=args.heads,
        reuploads=args.reuploads,
        core=args.core,
        include_context=args.include_context,
        encoder_variant=args.encoder_variant,
        physics_variant=args.physics_variant,
        physics_summary=args.physics_summary,
        quantum_encoding=args.quantum_encoding,
        observable_readout=args.observable_readout,
        tied_mean_dispersion=args.tied_mean_dispersion,
        haar_subtype_residual=args.haar_subtype_residual,
        haar_subtype_max_envelope=args.haar_subtype_max_envelope,
        shared_late_refinement=args.shared_late_refinement,
        cross_scale_reupload=args.cross_scale_reupload,
        r2_entanglers=args.r2_entanglers,
        equatorial_readout=args.equatorial_readout,
        meridional_readout=args.meridional_readout,
        dropout=args.dropout,
    ).to(device, memory_format=torch.channels_last)
    fresh_core_state = (
        clone_module_state(model.core)
        if args.reinitialize_core_after_init
        else None
    )
    initialization_report = None
    paired_spatial_binding = None
    haar_subtype_source_record = None
    haar_subtype_source_state = None
    haar_subtype_source_config = None
    shared_refinement_source_record = None
    shared_refinement_source_state = None
    shared_refinement_source_config = None
    r2_source_record = None
    r2_source_state = None
    r2_source_config = None
    equatorial_source_record = None
    equatorial_source_state = None
    equatorial_source_config = None
    meridional_source_record = None
    meridional_source_state = None
    meridional_source_config = None
    cross_scale_source_record = None
    cross_scale_source_state = None
    cross_scale_source_config = None
    if sum(
        bool(value)
        for value in (
            args.init_backbone_checkpoint,
            args.init_compatible_backbone_checkpoint,
            args.init_full_checkpoint,
        )
    ) > 1:
        raise ValueError(
            "checkpoint initialization modes are mutually exclusive"
        )
    if args.init_full_checkpoint:
        source_path = Path(args.init_full_checkpoint)
        source_checkpoint = torch.load(source_path, map_location="cpu")
        source_state = dict(source_checkpoint.get("model", source_checkpoint))
        if args.paired_spatial_init_report:
            paired_spatial_binding = validate_paired_spatial_initializer_binding(
                args, source_path, source_checkpoint, source_state
            )
        target_state = model.state_dict()
        initialization_method = "full-model-exact-or-zero-extension"
        if args.cross_scale_reupload:
            source_directory = source_path.parent
            required_artifacts = (
                "config.json",
                "data_report.json",
                "parameter_report.json",
                "summary.json",
                "fixed_summary_normalization.json",
                "haar_normalization.json",
                "morphology_normalization.json",
            )
            missing = [
                name
                for name in required_artifacts
                if not (source_directory / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"CSSR base-Haar source is missing required artifacts: {missing}"
                )
            source_config = json.loads(
                (source_directory / "config.json").read_text()
            )
            source_data = json.loads(
                (source_directory / "data_report.json").read_text()
            )
            source_parameters = json.loads(
                (source_directory / "parameter_report.json").read_text()
            )
            source_summary = json.loads(
                (source_directory / "summary.json").read_text()
            )
            source_contract = validate_cross_scale_source_contract(
                source_config,
                source_parameters,
                source_data,
                source_summary,
                data_report,
                args.core,
            )
            source_state, adapted_tensors = remap_haar_to_cross_scale_reupload(
                source_state, target_state
            )
            initialization_method = "haar-to-zero-cross-scale-reupload-exact-remap"
            cross_scale_source_state = dict(
                source_checkpoint.get("model", source_checkpoint)
            )
            cross_scale_source_config = source_config
            cross_scale_source_record = {
                "checkpoint": str(source_path),
                "checkpoint_sha256": file_sha256(source_path),
                **source_contract,
            }
        elif args.meridional_readout:
            source_directory = source_path.parent
            required_artifacts = (
                "config.json",
                "data_report.json",
                "parameter_report.json",
                "summary.json",
                "fixed_summary_normalization.json",
                "haar_normalization.json",
                "morphology_normalization.json",
            )
            missing = [
                name
                for name in required_artifacts
                if not (source_directory / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "Base-Haar meridional-16 source is missing required "
                    f"artifacts: {missing}"
                )
            source_config = json.loads(
                (source_directory / "config.json").read_text()
            )
            source_data = json.loads(
                (source_directory / "data_report.json").read_text()
            )
            source_parameters = json.loads(
                (source_directory / "parameter_report.json").read_text()
            )
            source_summary = json.loads(
                (source_directory / "summary.json").read_text()
            )
            checkpoint_sha256 = file_sha256(source_path)
            source_contract = validate_meridional_source_contract(
                checkpoint_sha256,
                source_config,
                source_parameters,
                source_data,
                source_summary,
                data_report,
            )
            source_state, adapted_tensors = (
                remap_haar_to_meridional_readout(source_state, target_state)
            )
            initialization_method = (
                "haar-to-zero-meridional-readout-gauge-remap"
            )
            meridional_source_state = dict(
                source_checkpoint.get("model", source_checkpoint)
            )
            meridional_source_config = source_config
            meridional_source_record = {
                "checkpoint": str(source_path),
                **source_contract,
            }
        elif args.equatorial_readout:
            source_directory = source_path.parent
            required_artifacts = (
                "config.json",
                "data_report.json",
                "parameter_report.json",
                "summary.json",
                "fixed_summary_normalization.json",
                "haar_normalization.json",
                "morphology_normalization.json",
            )
            missing = [
                name
                for name in required_artifacts
                if not (source_directory / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "Base-Haar EQR-16 source is missing required artifacts: "
                    f"{missing}"
                )
            source_config = json.loads(
                (source_directory / "config.json").read_text()
            )
            source_data = json.loads(
                (source_directory / "data_report.json").read_text()
            )
            source_parameters = json.loads(
                (source_directory / "parameter_report.json").read_text()
            )
            source_summary = json.loads(
                (source_directory / "summary.json").read_text()
            )
            checkpoint_sha256 = file_sha256(source_path)
            source_contract = validate_equatorial_source_contract(
                checkpoint_sha256,
                source_config,
                source_parameters,
                source_data,
                source_summary,
                data_report,
            )
            source_state, adapted_tensors = (
                remap_haar_to_equatorial_readout(source_state, target_state)
            )
            initialization_method = (
                "haar-to-zero-equatorial-readout-gauge-remap"
            )
            equatorial_source_state = dict(
                source_checkpoint.get("model", source_checkpoint)
            )
            equatorial_source_config = source_config
            equatorial_source_record = {
                "checkpoint": str(source_path),
                **source_contract,
            }
        elif args.r2_entanglers:
            source_directory = source_path.parent
            required_artifacts = (
                "config.json",
                "data_report.json",
                "parameter_report.json",
                "summary.json",
                "fixed_summary_normalization.json",
                "haar_normalization.json",
                "morphology_normalization.json",
            )
            missing = [
                name for name in required_artifacts
                if not (source_directory / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"Base-Haar R2 source is missing required artifacts: {missing}"
                )
            source_config = json.loads(
                (source_directory / "config.json").read_text()
            )
            source_data = json.loads(
                (source_directory / "data_report.json").read_text()
            )
            source_parameters = json.loads(
                (source_directory / "parameter_report.json").read_text()
            )
            source_summary = json.loads(
                (source_directory / "summary.json").read_text()
            )
            expected_config = {
                "encoder_variant": "deep-se-haar-morph",
                "physics_variant": "base",
                "physics_summary": "moments-morphology-haar",
                "heads": 4,
                "reuploads": 2,
                "quantum_encoding": "angle",
                "observable_readout": "pair",
                "include_context": False,
                "core": "quantum",
                "evaluate_test": False,
            }
            drift = {
                key: (source_config.get(key), expected)
                for key, expected in expected_config.items()
                if source_config.get(key) != expected
            }
            if (
                drift
                or bool(source_config.get("tied_mean_dispersion", False))
                or bool(source_config.get("haar_subtype_residual", False))
                or bool(source_config.get("shared_late_refinement", False))
                or bool(source_config.get("r2_entanglers", False))
                or bool(source_config.get("equatorial_readout", False))
                or bool(source_config.get("meridional_readout", False))
            ):
                raise RuntimeError(
                    f"R2 source is not exact base Haar: drift={drift}"
                )
            if (
                int(source_parameters.get("total", -1)) != 122595
                or int(source_parameters.get("quantum", -1)) != 88
                or int(source_parameters.get("haar_subtype_residual_trainable", 0)) != 0
                or int(source_parameters.get("dispersion_gate_trainable", 0)) != 0
                or int(source_parameters.get("shared_late_refinement_gate_trainable", 0)) != 0
            ):
                raise RuntimeError("R2 source parameter report is not base Haar")
            if (
                source_data.get("train_membership_sha256")
                != data_report["train_membership_sha256"]
                or source_data.get("development_manifest_sha256")
                != data_report["development_manifest_sha256"]
                or source_data.get("class_names") != data_report["class_names"]
                or source_data.get("official_test_cache_opened") is not False
                or source_summary.get("official_test_evaluated") is not False
            ):
                raise RuntimeError(
                    "R2 source violates split, manifest, class, or test lock"
                )
            source_state, adapted_tensors = remap_haar_to_r2_entanglers(
                source_state, target_state
            )
            initialization_method = "haar-to-zero-r2-entanglers-gauge-remap"
            r2_source_state = dict(
                source_checkpoint.get("model", source_checkpoint)
            )
            r2_source_config = source_config
            r2_source_record = {
                "checkpoint": str(source_path),
                "checkpoint_sha256": file_sha256(source_path),
                "source_parameters": 122595,
                "source_quantum_parameters": 88,
                "same_training_membership": True,
                "same_development_manifest": True,
                "source_official_test_opened": False,
            }
        elif args.shared_late_refinement:
            source_directory = source_path.parent
            required_artifacts = (
                "config.json",
                "data_report.json",
                "parameter_report.json",
                "summary.json",
                "fixed_summary_normalization.json",
                "haar_normalization.json",
                "morphology_normalization.json",
            )
            missing = [
                name for name in required_artifacts
                if not (source_directory / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"Base-Haar source is missing required artifacts: {missing}"
                )
            source_config = json.loads(
                (source_directory / "config.json").read_text()
            )
            source_data = json.loads(
                (source_directory / "data_report.json").read_text()
            )
            source_parameters = json.loads(
                (source_directory / "parameter_report.json").read_text()
            )
            source_summary = json.loads(
                (source_directory / "summary.json").read_text()
            )
            expected_config = {
                "encoder_variant": "deep-se-haar-morph",
                "physics_variant": "base",
                "physics_summary": "moments-morphology-haar",
                "heads": 4,
                "reuploads": 2,
                "quantum_encoding": "angle",
                "observable_readout": "pair",
                "include_context": False,
                "core": args.core,
                "evaluate_test": False,
            }
            drift = {
                key: (source_config.get(key), expected)
                for key, expected in expected_config.items()
                if source_config.get(key) != expected
            }
            if (
                drift
                or bool(source_config.get("tied_mean_dispersion", False))
                or bool(source_config.get("haar_subtype_residual", False))
                or bool(source_config.get("shared_late_refinement", False))
                or bool(source_config.get("r2_entanglers", False))
                or bool(source_config.get("equatorial_readout", False))
                or bool(source_config.get("meridional_readout", False))
            ):
                raise RuntimeError(
                    "Shared-refinement source is not exact base Haar: "
                    f"drift={drift}"
                )
            if (
                int(source_parameters.get("total", -1)) != 122595
                or int(source_parameters.get("quantum", -1)) != 88
                or int(source_parameters.get("haar_subtype_residual_trainable", 0)) != 0
                or int(source_parameters.get("dispersion_gate_trainable", 0)) != 0
                or int(
                    source_parameters.get(
                        "shared_late_refinement_gate_trainable", 0
                    )
                )
                != 0
            ):
                raise RuntimeError(
                    "Shared-refinement source parameter report is not base Haar"
                )
            if (
                source_data.get("train_membership_sha256")
                != data_report["train_membership_sha256"]
                or source_data.get("development_manifest_sha256")
                != data_report["development_manifest_sha256"]
                or source_data.get("class_names") != data_report["class_names"]
                or source_data.get("official_test_cache_opened") is not False
                or source_summary.get("official_test_evaluated") is not False
            ):
                raise RuntimeError(
                    "Shared-refinement source violates split, manifest, class, "
                    "or test lock"
                )
            source_state, adapted_tensors = (
                remap_haar_to_shared_late_refinement(
                    source_state, target_state
                )
            )
            initialization_method = (
                "haar-to-zero-shared-late-refinement-exact-remap"
            )
            shared_refinement_source_state = dict(
                source_checkpoint.get("model", source_checkpoint)
            )
            shared_refinement_source_config = source_config
            shared_refinement_source_record = {
                "checkpoint": str(source_path),
                "checkpoint_sha256": file_sha256(source_path),
                "source_parameters": 122595,
                "source_core": args.core,
                "same_training_membership": True,
                "same_development_manifest": True,
                "source_official_test_opened": False,
            }
        elif args.haar_subtype_residual:
            source_checkpoint_sha256 = file_sha256(source_path)
            if source_checkpoint_sha256 != ANNULAR_HAAR_BASE_CHECKPOINT_SHA256:
                raise RuntimeError(
                    "Subtype residual source checkpoint identity drifted"
                )
            source_directory = source_path.parent
            required_artifacts = (
                "config.json",
                "data_report.json",
                "parameter_report.json",
                "summary.json",
                "haar_normalization.json",
                "morphology_normalization.json",
            )
            missing = [
                name for name in required_artifacts
                if not (source_directory / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"Base-Haar source is missing required artifacts: {missing}"
                )
            source_config = json.loads(
                (source_directory / "config.json").read_text()
            )
            source_data = json.loads(
                (source_directory / "data_report.json").read_text()
            )
            source_parameters = json.loads(
                (source_directory / "parameter_report.json").read_text()
            )
            source_summary = json.loads(
                (source_directory / "summary.json").read_text()
            )
            expected_config = {
                "image_size": 96,
                "encoder_variant": "deep-se-haar-morph",
                "physics_variant": "base",
                "physics_summary": "moments-morphology-haar",
                "heads": 4,
                "reuploads": 2,
                "quantum_encoding": "angle",
                "observable_readout": "pair",
                "include_context": False,
                "core": args.core,
                "evaluate_test": False,
                "train_subset_protocol": "hash-v1",
                "max_train_per_class": 11667,
            }
            drift = {
                key: (source_config.get(key), expected)
                for key, expected in expected_config.items()
                if source_config.get(key) != expected
            }
            if (
                drift
                or bool(source_config.get("tied_mean_dispersion", False))
                or bool(source_config.get("haar_subtype_residual", False))
                or bool(source_config.get("shared_late_refinement", False))
                or bool(source_config.get("r2_entanglers", False))
                or bool(source_config.get("equatorial_readout", False))
                or bool(source_config.get("meridional_readout", False))
            ):
                raise RuntimeError(
                    f"Subtype residual source is not exact base Haar: drift={drift}"
                )
            if (
                int(source_parameters.get("total", -1)) != 122595
                or int(source_parameters.get("haar_subtype_residual_trainable", 0)) != 0
                or int(source_parameters.get("dispersion_gate_trainable", 0)) != 0
                or int(
                    source_parameters.get(
                        "shared_late_refinement_gate_trainable", 0
                    )
                )
                != 0
            ):
                raise RuntimeError("Subtype residual source parameter report is not base Haar")
            if (
                source_data.get("train_size") != 35001
                or source_data.get("train_membership_sha256")
                != OOF_FULL_HALF_MEMBERSHIP_SHA256
                or source_data.get("development_manifest_sha256")
                != OOF_DEVELOPMENT_MANIFEST_SHA256
                or source_data.get("class_names") != data_report["class_names"]
                or source_data.get("official_test_cache_opened") is not False
                or "test" in source_data
                or source_summary.get("official_test_evaluated") is not False
                or "test" in source_summary
                or data_report.get("train_membership_sha256")
                != OOF_FULL_HALF_MEMBERSHIP_SHA256
                or data_report.get("development_manifest_sha256")
                != OOF_DEVELOPMENT_MANIFEST_SHA256
                or data_report.get("official_test_cache_opened") is not False
                or "test" in data_report
            ):
                raise RuntimeError(
                    "Subtype residual source violates split, manifest, class, or test lock"
                )
            source_state, adapted_tensors = remap_haar_to_subtype_residual(
                source_state, target_state
            )
            initialization_method = "haar-to-zero-subtype-residual-exact-remap"
            haar_subtype_source_state = dict(
                source_checkpoint.get("model", source_checkpoint)
            )
            haar_subtype_source_config = source_config
            haar_subtype_source_record = {
                "checkpoint": str(source_path),
                "checkpoint_sha256": source_checkpoint_sha256,
                "source_parameters": 122595,
                "source_core": args.core,
                "same_training_membership": True,
                "same_development_manifest": True,
                "source_official_test_opened": False,
            }
        elif args.encoder_variant == "deep-se-haar-morph":
            if args.physics_summary != "moments-morphology-haar":
                raise ValueError(
                    "The annular-Haar warm start requires "
                    "moments-morphology-haar"
                )
            source_state, adapted_tensors = (
                remap_morphology_kd_to_haar_candidate(
                    source_state, target_state
                )
            )
            initialization_method = "morphology-kd-to-annular-haar-exact-remap"
        else:
            adapted_tensors = []
            for expandable_key in (
                "encoder.stem.0.weight",
                "orbit_projection.weight",
            ):
                zero_extend_input_weight(
                    source_state,
                    target_state,
                    expandable_key,
                    adapted_tensors,
                    insert_before_tail=(
                        model.physics_summary_dim
                        if expandable_key == "orbit_projection.weight"
                        and args.encoder_variant == "micro-stat"
                        else 0
                    ),
                )
        model.load_state_dict(source_state, strict=True)
        initialization_report = {
            "checkpoint": str(source_path),
            "checkpoint_sha256": file_sha256(source_path),
            "source_epoch": source_checkpoint.get("epoch"),
            "loaded_tensors": len(source_state),
            "loaded_prefixes": ["<full-model>"],
            "missing_target_tensors": 0,
            "adapted_tensors": adapted_tensors,
            "method": initialization_method,
        }
        if paired_spatial_binding is not None:
            initialization_report["paired_spatial_binding"] = (
                paired_spatial_binding
            )
        if args.encoder_variant == "deep-se-haar-morph":
            initialization_report["frozen_morphology_context_indices"] = list(
                HAAR_MORPHOLOGY_CONTEXT_INDICES
            )
        if args.cross_scale_reupload:
            initialization_report["cross_scale_source"] = (
                cross_scale_source_record
            )
            initialization_report["cross_scale_reupload"] = (
                cross_scale_initialization_record(model)
            )
            initialization_report["cross_scale_reupload"][
                "exact_replay_before_normalization_pending"
            ] = True
            initialization_report["cross_scale_reupload"][
                "exact_replay_after_normalization_pending"
            ] = True
        if args.haar_subtype_residual:
            initialization_report["haar_subtype_source"] = (
                haar_subtype_source_record
            )
            initialization_report["haar_subtype_residual"] = {
                "enabled": True,
                "parameters": 15,
                "initialization": "zeros",
                "all_weights_zero_after_remap": bool(
                    torch.equal(
                        model.haar_subtype_residual.weight.detach(),
                        torch.zeros_like(model.haar_subtype_residual.weight),
                    )
                ),
                "logit_path": "[+delta,-delta,0]",
                "exact_base_replay_pending_selection": True,
            }
        if args.shared_late_refinement:
            initialization_report["shared_refinement_source"] = (
                shared_refinement_source_record
            )
            initialization_report["shared_late_refinement"] = (
                shared_late_refinement_initialization_record(model)
            )
            initialization_report["shared_late_refinement"][
                "exact_base_replay_pending_normalization"
            ] = True
        if args.meridional_readout:
            initialization_report["meridional_source"] = (
                meridional_source_record
            )
            initialization_report["meridional_readout"] = (
                meridional_readout_initialization_record(model)
            )
            initialization_report["meridional_readout"][
                "probability_replay_pending_normalization"
            ] = True
        if args.equatorial_readout:
            initialization_report["equatorial_source"] = (
                equatorial_source_record
            )
            initialization_report["equatorial_readout"] = (
                equatorial_readout_initialization_record(model)
            )
            initialization_report["equatorial_readout"][
                "probability_replay_pending_normalization"
            ] = True
        if args.r2_entanglers:
            initialization_report["r2_source"] = r2_source_record
            initialization_report["r2_entanglers"] = (
                r2_entangler_initialization_record(model)
            )
            initialization_report["r2_entanglers"][
                "probability_replay_pending_normalization"
            ] = True
        if args.tied_mean_dispersion:
            initialization_report["tied_mean_dispersion"] = (
                tied_mean_dispersion_initialization_record(model)
            )
        atomic_json(output_dir / "initialization_report.json", initialization_report)
        print(f"INITIALIZATION {json.dumps(initialization_report, sort_keys=True)}", flush=True)
    elif args.init_compatible_backbone_checkpoint:
        source_path = Path(args.init_compatible_backbone_checkpoint)
        source_checkpoint = torch.load(source_path, map_location="cpu")
        source_state = source_checkpoint.get("model", source_checkpoint)
        if args.encoder_variant not in (
            "deep-se",
            "deep-se-morph",
            "deep-se-mscorr",
        ):
            raise ValueError(
                "compatible-prefix initialization is currently defined only for deep-SE variants"
            )
        if args.encoder_variant == "deep-se-morph":
            # The morphology candidate inherits every spatial block from the
            # same-subset deep-SE model.  Its quantum/classical cores remain
            # independently initialized for a clean matched-core control.
            allowed_prefixes = (
                "encoder.stem.",
                "encoder.blocks.0.",
                "encoder.blocks.1.",
                "encoder.blocks.2.",
                "encoder.blocks.3.",
                "encoder.blocks.4.",
                "encoder.blocks.5.",
                "encoder.blocks.6.",
                "encoder.blocks.7.",
                "encoder.final.",
                "orbit_projection.",
            )
        elif args.encoder_variant == "deep-se-mscorr":
            # Unlike the fixed-feature morphology fusion, this candidate keeps
            # the core and classifier semantics unchanged.  Load them exactly,
            # along with every shared deep-SE spatial block.
            allowed_prefixes = (
                "encoder.stem.",
                "encoder.blocks.0.",
                "encoder.blocks.1.",
                "encoder.blocks.2.",
                "encoder.blocks.3.",
                "encoder.blocks.4.",
                "encoder.blocks.5.",
                "encoder.blocks.6.",
                "encoder.blocks.7.",
                "encoder.final.",
                "orbit_projection.",
                "core.",
                "head.",
            )
        else:
            allowed_prefixes = (
                "encoder.stem.",
                "encoder.blocks.0.",
                "encoder.blocks.1.",
                "encoder.blocks.2.",
                "encoder.blocks.3.",
                "encoder.blocks.4.",
                "orbit_projection.",
                "core.",
                "head.",
            )
        compatible_state, skipped_tensors = shape_compatible_prefix_state(
            source_state, model.state_dict(), allowed_prefixes
        )
        adapted_tensors = []
        if args.encoder_variant in ("deep-se-morph", "deep-se-mscorr"):
            adaptation_state = {
                key: source_state[key]
                for key in (
                    "encoder.final.0.weight",
                    "encoder.final.1.weight",
                    "encoder.final.1.bias",
                    "orbit_projection.weight",
                )
                if key in source_state
            }
            for key in (
                "encoder.final.0.weight",
                "encoder.final.1.weight",
                "encoder.final.1.bias",
            ):
                prefix_slice_output_tensor(
                    adaptation_state,
                    model.state_dict(),
                    key,
                    adapted_tensors,
                )
            source_encoder_dim = int(
                source_state["encoder.final.0.weight"].shape[0]
            )
            if args.encoder_variant == "deep-se-morph":
                remap_projection_encoder_and_summary(
                    adaptation_state,
                    model.state_dict(),
                    "orbit_projection.weight",
                    adapted_tensors,
                    source_encoder_dim=source_encoder_dim,
                    target_encoder_dim=model.encoder.output_dim,
                    preserved_summary_dim=2 * model.physics.output_channels,
                )
            else:
                target_final_dim = model.encoder.final[0].out_channels
                remap_projection_to_multiscale_correlation(
                    adaptation_state,
                    model.state_dict(),
                    "orbit_projection.weight",
                    adapted_tensors,
                    source_encoder_dim=source_encoder_dim,
                    target_multiscale_dim=(
                        model.encoder.output_dim - target_final_dim
                    ),
                    target_final_dim=target_final_dim,
                    preserved_summary_dim=2 * model.physics.output_channels,
                )
                required_adaptations = {
                    "encoder.final.0.weight",
                    "encoder.final.1.weight",
                    "encoder.final.1.bias",
                    "orbit_projection.weight",
                }
                actual_adaptations = {
                    item["key"] for item in adapted_tensors
                }
                if not required_adaptations.issubset(actual_adaptations):
                    missing = sorted(required_adaptations - actual_adaptations)
                    raise RuntimeError(
                        "Incompatible deep-SE checkpoint for multiscale "
                        f"initialization; missing adaptations: {missing}"
                    )
            compatible_state.update(adaptation_state)
        else:
            # The four output rows per angle channel and the 16 physics moments
            # keep their semantics; new deep-encoder columns start at zero.
            if "orbit_projection.weight" in source_state:
                compatible_state["orbit_projection.weight"] = source_state[
                    "orbit_projection.weight"
                ]
            zero_extend_input_weight(
                compatible_state,
                model.state_dict(),
                "orbit_projection.weight",
                adapted_tensors,
                insert_before_tail=model.physics_summary_dim,
            )
        if adapted_tensors:
            adapted_keys = {item["key"] for item in adapted_tensors}
            skipped_tensors = [
                item
                for item in skipped_tensors
                if item["key"] not in adapted_keys
            ]
        if not any(key.startswith("encoder.stem.") for key in compatible_state):
            raise RuntimeError(f"No compatible encoder stem found in {source_path}")
        incompatible = model.load_state_dict(compatible_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected compatible initialization keys: {incompatible.unexpected_keys}"
            )
        initialization_report = {
            "checkpoint": str(source_path),
            "checkpoint_sha256": file_sha256(source_path),
            "source_epoch": source_checkpoint.get("epoch"),
            "loaded_tensors": len(compatible_state),
            "loaded_prefixes": list(allowed_prefixes),
            "missing_target_tensors": len(incompatible.missing_keys),
            "adapted_tensors": adapted_tensors,
            "skipped_tensors": skipped_tensors,
            "method": "exact-shape-declared-prefix",
        }
        atomic_json(output_dir / "initialization_report.json", initialization_report)
        print(f"INITIALIZATION {json.dumps(initialization_report, sort_keys=True)}", flush=True)
    elif args.init_backbone_checkpoint:
        source_path = Path(args.init_backbone_checkpoint)
        source_checkpoint = torch.load(source_path, map_location="cpu")
        source_state = source_checkpoint.get("model", source_checkpoint)
        allowed_prefixes = ("physics.", "encoder.", "orbit_projection.")
        backbone_state = {
            key: value for key, value in source_state.items() if key.startswith(allowed_prefixes)
        }
        adapted_tensors = []
        target_state = model.state_dict()
        for expandable_key in (
            "encoder.stem.0.weight",
            "orbit_projection.weight",
        ):
            zero_extend_input_weight(
                backbone_state,
                target_state,
                expandable_key,
                adapted_tensors,
                insert_before_tail=(
                    model.physics_summary_dim
                    if expandable_key == "orbit_projection.weight"
                    and args.encoder_variant == "micro-stat"
                    else 0
                ),
            )
        if not any(key.startswith("encoder.") for key in backbone_state):
            raise RuntimeError(f"No encoder weights found in {source_path}")
        incompatible = model.load_state_dict(backbone_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected initialized keys: {unexpected}")
        initialization_report = {
            "checkpoint": str(source_path),
            "checkpoint_sha256": file_sha256(source_path),
            "source_epoch": source_checkpoint.get("epoch"),
            "loaded_tensors": len(backbone_state),
            "loaded_prefixes": list(allowed_prefixes),
            "missing_target_tensors": len(incompatible.missing_keys),
            "adapted_tensors": adapted_tensors,
        }
        atomic_json(output_dir / "initialization_report.json", initialization_report)
        print(f"INITIALIZATION {json.dumps(initialization_report, sort_keys=True)}", flush=True)
    if args.reinitialize_core_after_init:
        if fresh_core_state is None:
            raise RuntimeError("Fresh core state was not captured before initialization")
        if initialization_report is None or initialization_report.get("method") != (
            "morphology-kd-to-annular-haar-exact-remap"
        ):
            raise RuntimeError(
                "Core reinitialization requires the exact morphology-KD Haar remap"
            )
        core_reinitialization = restore_fresh_haar_core(
            model, fresh_core_state
        )
        initialization_report["core_reinitialized_after_init"] = True
        initialization_report["core_reinitialization"] = core_reinitialization
        atomic_json(
            output_dir / "initialization_report.json", initialization_report
        )
        print(
            "CORE_REINITIALIZATION "
            + json.dumps(core_reinitialization, sort_keys=True),
            flush=True,
        )
    if args.meridional_readout:
        if (
            meridional_source_state is None
            or meridional_source_config is None
            or meridional_source_record is None
            or initialization_report is None
        ):
            raise RuntimeError(
                "Meridional-16 source was not validated during initialization"
            )
        normalization_loader = make_loader(
            CachedNPYDataset(development_cache, train_indices),
            args.batch_size,
            False,
            args.workers,
            args.seed + 3000,
        )
        source_directory = Path(
            meridional_source_record["checkpoint"]
        ).parent
        fixed_normalization = json.loads(
            (source_directory / "fixed_summary_normalization.json").read_text()
        )
        fixed_normalization.update(
            {
                "morphology_preserved_from_initialization": True,
                "haar_preserved_from_initialization": True,
                "preserved_bitwise_for_meridional_readout": True,
                "source_checkpoint_sha256": meridional_source_record[
                    "checkpoint_sha256"
                ],
                "validation_samples_used": 0,
                "official_test_samples_used": 0,
            }
        )
        for name in ("morphology", "haar"):
            source_normalization = json.loads(
                (source_directory / f"{name}_normalization.json").read_text()
            )
            source_normalization[
                "preserved_bitwise_for_meridional_readout"
            ] = True
            source_normalization["preserved_from_checkpoint_sha256"] = (
                meridional_source_record["checkpoint_sha256"]
            )
            atomic_json(
                output_dir / f"{name}_normalization.json",
                source_normalization,
            )
            fixed_normalization[name] = source_normalization
        atomic_json(
            output_dir / "fixed_summary_normalization.json",
            fixed_normalization,
        )

        source_model = D4OrbitClassifier(
            num_classes=len(class_names),
            heads=int(meridional_source_config["heads"]),
            reuploads=int(meridional_source_config["reuploads"]),
            core=str(meridional_source_config["core"]),
            include_context=bool(meridional_source_config["include_context"]),
            encoder_variant=str(
                meridional_source_config["encoder_variant"]
            ),
            physics_variant=str(meridional_source_config["physics_variant"]),
            physics_summary=str(meridional_source_config["physics_summary"]),
            quantum_encoding=str(
                meridional_source_config["quantum_encoding"]
            ),
            observable_readout=str(
                meridional_source_config["observable_readout"]
            ),
            tied_mean_dispersion=False,
            haar_subtype_residual=False,
            shared_late_refinement=False,
            cross_scale_reupload=False,
            r2_entanglers=False,
            equatorial_readout=False,
            meridional_readout=False,
            dropout=float(meridional_source_config["dropout"]),
        ).to(device, memory_format=torch.channels_last)
        source_model.load_state_dict(meridional_source_state, strict=True)
        target_state = model.state_dict()
        protected_equal = all(
            key == "head.classifier.bias"
            or (
                key in target_state
                and torch.equal(
                    target_state[key].detach().cpu(), value.detach().cpu()
                )
            )
            for key, value in meridional_source_state.items()
        )
        source_bias = meridional_source_state[
            "head.classifier.bias"
        ].detach().cpu()
        gauge_equal = torch.equal(
            target_state["head.classifier.bias"].detach().cpu(),
            source_bias[:2] - source_bias[2],
        )
        if not protected_equal or not gauge_equal:
            raise RuntimeError(
                "Meridional-16 remap changed a protected base-Haar tensor"
            )
        replay_images = next(iter(normalization_loader))[0][:16].to(
            device, non_blocking=True
        ).contiguous(memory_format=torch.channels_last)
        replay = meridional_readout_probability_replay(
            source_model, model, replay_images
        )
        replay["protected_source_state_bitwise_equal"] = protected_equal
        replay["classifier_bias_gauge_exact"] = bool(gauge_equal)
        initialization_report["meridional_readout"][
            "probability_replay_pending_normalization"
        ] = False
        initialization_report["meridional_readout"]["exact_replay"] = replay
        atomic_json(
            output_dir / "initialization_report.json", initialization_report
        )
        print(
            "MERIDIONAL_READOUT_REPLAY "
            + json.dumps(replay, sort_keys=True),
            flush=True,
        )
        del source_model
    elif args.equatorial_readout:
        if (
            equatorial_source_state is None
            or equatorial_source_config is None
            or equatorial_source_record is None
            or initialization_report is None
        ):
            raise RuntimeError(
                "EQR-16 source was not validated during initialization"
            )
        normalization_loader = make_loader(
            CachedNPYDataset(development_cache, train_indices),
            args.batch_size,
            False,
            args.workers,
            args.seed + 3000,
        )
        source_directory = Path(
            equatorial_source_record["checkpoint"]
        ).parent
        fixed_normalization = json.loads(
            (source_directory / "fixed_summary_normalization.json").read_text()
        )
        fixed_normalization.update(
            {
                "morphology_preserved_from_initialization": True,
                "haar_preserved_from_initialization": True,
                "preserved_bitwise_for_equatorial_readout": True,
                "source_checkpoint_sha256": equatorial_source_record[
                    "checkpoint_sha256"
                ],
                "validation_samples_used": 0,
                "official_test_samples_used": 0,
            }
        )
        for name in ("morphology", "haar"):
            source_normalization = json.loads(
                (source_directory / f"{name}_normalization.json").read_text()
            )
            source_normalization[
                "preserved_bitwise_for_equatorial_readout"
            ] = True
            source_normalization["preserved_from_checkpoint_sha256"] = (
                equatorial_source_record["checkpoint_sha256"]
            )
            atomic_json(
                output_dir / f"{name}_normalization.json",
                source_normalization,
            )
            fixed_normalization[name] = source_normalization
        atomic_json(
            output_dir / "fixed_summary_normalization.json",
            fixed_normalization,
        )

        source_model = D4OrbitClassifier(
            num_classes=len(class_names),
            heads=int(equatorial_source_config["heads"]),
            reuploads=int(equatorial_source_config["reuploads"]),
            core=str(equatorial_source_config["core"]),
            include_context=bool(equatorial_source_config["include_context"]),
            encoder_variant=str(
                equatorial_source_config["encoder_variant"]
            ),
            physics_variant=str(equatorial_source_config["physics_variant"]),
            physics_summary=str(equatorial_source_config["physics_summary"]),
            quantum_encoding=str(
                equatorial_source_config["quantum_encoding"]
            ),
            observable_readout=str(
                equatorial_source_config["observable_readout"]
            ),
            tied_mean_dispersion=False,
            haar_subtype_residual=False,
            shared_late_refinement=False,
            cross_scale_reupload=False,
            r2_entanglers=False,
            equatorial_readout=False,
            dropout=float(equatorial_source_config["dropout"]),
        ).to(device, memory_format=torch.channels_last)
        source_model.load_state_dict(equatorial_source_state, strict=True)
        target_state = model.state_dict()
        protected_equal = all(
            key == "head.classifier.bias"
            or (
                key in target_state
                and torch.equal(
                    target_state[key].detach().cpu(), value.detach().cpu()
                )
            )
            for key, value in equatorial_source_state.items()
        )
        source_bias = equatorial_source_state[
            "head.classifier.bias"
        ].detach().cpu()
        gauge_equal = torch.equal(
            target_state["head.classifier.bias"].detach().cpu(),
            source_bias[:2] - source_bias[2],
        )
        if not protected_equal or not gauge_equal:
            raise RuntimeError(
                "EQR-16 remap changed a protected base-Haar tensor"
            )
        replay_images = next(iter(normalization_loader))[0][:16].to(
            device, non_blocking=True
        ).contiguous(memory_format=torch.channels_last)
        replay = equatorial_readout_probability_replay(
            source_model, model, replay_images
        )
        replay["protected_source_state_bitwise_equal"] = protected_equal
        replay["classifier_bias_gauge_exact"] = bool(gauge_equal)
        initialization_report["equatorial_readout"][
            "probability_replay_pending_normalization"
        ] = False
        initialization_report["equatorial_readout"]["exact_replay"] = replay
        atomic_json(
            output_dir / "initialization_report.json", initialization_report
        )
        print(
            "EQUATORIAL_READOUT_REPLAY "
            + json.dumps(replay, sort_keys=True),
            flush=True,
        )
        del source_model
    elif args.r2_entanglers:
        if (
            r2_source_state is None
            or r2_source_config is None
            or r2_source_record is None
            or initialization_report is None
        ):
            raise RuntimeError(
                "R2 source was not validated during initialization"
            )
        normalization_loader = make_loader(
            CachedNPYDataset(development_cache, train_indices),
            args.batch_size,
            False,
            args.workers,
            args.seed + 3000,
        )
        source_directory = Path(r2_source_record["checkpoint"]).parent
        fixed_normalization = json.loads(
            (source_directory / "fixed_summary_normalization.json").read_text()
        )
        fixed_normalization.update(
            {
                "morphology_preserved_from_initialization": True,
                "haar_preserved_from_initialization": True,
                "preserved_bitwise_for_r2_entanglers": True,
                "source_checkpoint_sha256": r2_source_record[
                    "checkpoint_sha256"
                ],
                "validation_samples_used": 0,
                "official_test_samples_used": 0,
            }
        )
        for name in ("morphology", "haar"):
            source_normalization = json.loads(
                (source_directory / f"{name}_normalization.json").read_text()
            )
            source_normalization["preserved_bitwise_for_r2_entanglers"] = True
            source_normalization["preserved_from_checkpoint_sha256"] = (
                r2_source_record["checkpoint_sha256"]
            )
            atomic_json(
                output_dir / f"{name}_normalization.json",
                source_normalization,
            )
            fixed_normalization[name] = source_normalization
        atomic_json(
            output_dir / "fixed_summary_normalization.json",
            fixed_normalization,
        )

        source_model = D4OrbitClassifier(
            num_classes=len(class_names),
            heads=int(r2_source_config["heads"]),
            reuploads=int(r2_source_config["reuploads"]),
            core=str(r2_source_config["core"]),
            include_context=bool(r2_source_config["include_context"]),
            encoder_variant=str(r2_source_config["encoder_variant"]),
            physics_variant=str(r2_source_config["physics_variant"]),
            physics_summary=str(r2_source_config["physics_summary"]),
            quantum_encoding=str(r2_source_config["quantum_encoding"]),
            observable_readout=str(r2_source_config["observable_readout"]),
            tied_mean_dispersion=False,
            haar_subtype_residual=False,
            shared_late_refinement=False,
            cross_scale_reupload=False,
            r2_entanglers=False,
            dropout=float(r2_source_config["dropout"]),
        ).to(device, memory_format=torch.channels_last)
        source_model.load_state_dict(r2_source_state, strict=True)
        target_state = model.state_dict()
        protected_equal = all(
            key == "head.classifier.bias"
            or (
                key in target_state
                and torch.equal(
                    target_state[key].detach().cpu(), value.detach().cpu()
                )
            )
            for key, value in r2_source_state.items()
        )
        source_bias = r2_source_state["head.classifier.bias"].detach().cpu()
        gauge_equal = torch.equal(
            target_state["head.classifier.bias"].detach().cpu(),
            source_bias[:2] - source_bias[2],
        )
        if not protected_equal or not gauge_equal:
            raise RuntimeError("R2 remap changed a protected base-Haar tensor")
        replay_images = next(iter(normalization_loader))[0][:16].to(
            device, non_blocking=True
        ).contiguous(memory_format=torch.channels_last)
        replay = r2_entangler_probability_replay(
            source_model, model, replay_images
        )
        replay["protected_source_state_bitwise_equal"] = protected_equal
        replay["classifier_bias_gauge_exact"] = bool(gauge_equal)
        initialization_report["r2_entanglers"][
            "probability_replay_pending_normalization"
        ] = False
        initialization_report["r2_entanglers"]["exact_replay"] = replay
        atomic_json(
            output_dir / "initialization_report.json", initialization_report
        )
        print(
            "R2_ENTANGLER_REPLAY " + json.dumps(replay, sort_keys=True),
            flush=True,
        )
        del source_model
    elif model.shared_late_refinement:
        if (
            shared_refinement_source_state is None
            or shared_refinement_source_config is None
            or shared_refinement_source_record is None
            or initialization_report is None
        ):
            raise RuntimeError(
                "Shared late-refinement source was not validated during initialization"
            )
        normalization_loader = make_loader(
            CachedNPYDataset(development_cache, train_indices),
            args.batch_size,
            False,
            args.workers,
            args.seed + 3000,
        )
        source_directory = Path(
            shared_refinement_source_record["checkpoint"]
        ).parent
        fixed_normalization = json.loads(
            (source_directory / "fixed_summary_normalization.json").read_text()
        )
        fixed_normalization.update(
            {
                "morphology_preserved_from_initialization": True,
                "haar_preserved_from_initialization": True,
                "preserved_bitwise_for_shared_late_refinement": True,
                "source_checkpoint_sha256": shared_refinement_source_record[
                    "checkpoint_sha256"
                ],
                "validation_samples_used": 0,
                "official_test_samples_used": 0,
            }
        )
        for name in ("morphology", "haar"):
            source_normalization = json.loads(
                (source_directory / f"{name}_normalization.json").read_text()
            )
            source_normalization[
                "preserved_bitwise_for_shared_late_refinement"
            ] = True
            source_normalization["preserved_from_checkpoint_sha256"] = (
                shared_refinement_source_record["checkpoint_sha256"]
            )
            atomic_json(
                output_dir / f"{name}_normalization.json",
                source_normalization,
            )
            fixed_normalization[name] = source_normalization
        atomic_json(
            output_dir / "fixed_summary_normalization.json",
            fixed_normalization,
        )

        source_model = D4OrbitClassifier(
            num_classes=len(class_names),
            heads=int(shared_refinement_source_config["heads"]),
            reuploads=int(shared_refinement_source_config["reuploads"]),
            core=str(shared_refinement_source_config["core"]),
            include_context=bool(
                shared_refinement_source_config["include_context"]
            ),
            encoder_variant=str(
                shared_refinement_source_config["encoder_variant"]
            ),
            physics_variant=str(
                shared_refinement_source_config["physics_variant"]
            ),
            physics_summary=str(
                shared_refinement_source_config["physics_summary"]
            ),
            quantum_encoding=str(
                shared_refinement_source_config["quantum_encoding"]
            ),
            observable_readout=str(
                shared_refinement_source_config["observable_readout"]
            ),
            tied_mean_dispersion=False,
            haar_subtype_residual=False,
            shared_late_refinement=False,
            cross_scale_reupload=False,
            dropout=float(shared_refinement_source_config["dropout"]),
        ).to(device, memory_format=torch.channels_last)
        source_model.load_state_dict(
            shared_refinement_source_state, strict=True
        )
        target_state = model.state_dict()
        nonshared_equal = all(
            key in target_state
            and torch.equal(
                target_state[key].detach().cpu(), value.detach().cpu()
            )
            for key, value in shared_refinement_source_state.items()
        )
        if not nonshared_equal:
            raise RuntimeError(
                "Shared refinement changed a base-Haar state tensor"
            )
        replay_images = next(iter(normalization_loader))[0][:16].to(
            device, non_blocking=True
        ).contiguous(memory_format=torch.channels_last)
        replay = shared_late_refinement_exact_replay(
            source_model, model, replay_images
        )
        replay["nonshared_state_bitwise_equal"] = nonshared_equal
        initialization_report["shared_late_refinement"][
            "exact_base_replay_pending_normalization"
        ] = False
        initialization_report["shared_late_refinement"][
            "exact_replay"
        ] = replay
        atomic_json(
            output_dir / "initialization_report.json", initialization_report
        )
        print(
            "SHARED_LATE_REFINEMENT_REPLAY "
            + json.dumps(replay, sort_keys=True),
            flush=True,
        )
        del source_model
    elif model.haar_subtype_residual is not None:
        if (
            haar_subtype_source_state is None
            or haar_subtype_source_config is None
            or haar_subtype_source_record is None
            or initialization_report is None
        ):
            raise RuntimeError("Haar subtype source was not validated during initialization")
        normalization_loader = make_loader(
            CachedNPYDataset(development_cache, train_indices),
            args.batch_size,
            False,
            args.workers,
            args.seed + 3000,
        )
        selection_report = fit_haar_subtype_selection(
            model, normalization_loader, device
        )
        if (
            selection_report["fit_images"] != 35001
            or selection_report["fit_views"] != 280008
            or selection_report["class_counts"]
            != {"axion": 11667, "cdm": 11667}
            or selection_report["selection_samples"] != 23334
        ):
            raise RuntimeError(
                f"Haar subtype selection did not use the fixed half set: {selection_report}"
            )
        selection_report.update(
            {
                "train_membership_sha256": data_report[
                    "train_membership_sha256"
                ],
                "development_manifest_sha256": data_report[
                    "development_manifest_sha256"
                ],
                "class_names": data_report["class_names"],
                "source_checkpoint": haar_subtype_source_record["checkpoint"],
                "source_checkpoint_sha256": haar_subtype_source_record[
                    "checkpoint_sha256"
                ],
            }
        )
        selection_material = dict(selection_report)
        selection_material.pop("selection_spec_sha256", None)
        selection_report["selection_spec_sha256"] = hashlib.sha256(
            json.dumps(
                selection_material,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if (
            selection_report["selection_spec_sha256"]
            != HAAR_SUBTYPE_SELECTION_SPEC_SHA256
        ):
            raise RuntimeError(
                "Haar subtype training-only selection specification drifted: "
                f"actual={selection_report['selection_spec_sha256']} "
                f"expected={HAAR_SUBTYPE_SELECTION_SPEC_SHA256}"
            )
        atomic_json(
            output_dir / "haar_subtype_selection.json", selection_report
        )

        source_directory = Path(haar_subtype_source_record["checkpoint"]).parent
        fixed_normalization = {
            "morphology_preserved_from_initialization": True,
            "haar_preserved_from_initialization": True,
            "source_checkpoint_sha256": haar_subtype_source_record[
                "checkpoint_sha256"
            ],
            "validation_samples_used": 0,
            "official_test_samples_used": 0,
        }
        for name in ("morphology", "haar"):
            source_normalization = json.loads(
                (source_directory / f"{name}_normalization.json").read_text()
            )
            source_normalization["preserved_bitwise_for_subtype_residual"] = True
            source_normalization["preserved_from_checkpoint_sha256"] = (
                haar_subtype_source_record["checkpoint_sha256"]
            )
            atomic_json(
                output_dir / f"{name}_normalization.json",
                source_normalization,
            )
            fixed_normalization[name] = source_normalization
        atomic_json(
            output_dir / "fixed_summary_normalization.json",
            fixed_normalization,
        )

        source_model = D4OrbitClassifier(
            num_classes=len(class_names),
            heads=int(haar_subtype_source_config["heads"]),
            reuploads=int(haar_subtype_source_config["reuploads"]),
            core=str(haar_subtype_source_config["core"]),
            include_context=bool(haar_subtype_source_config["include_context"]),
            encoder_variant=str(haar_subtype_source_config["encoder_variant"]),
            physics_variant=str(haar_subtype_source_config["physics_variant"]),
            physics_summary=str(haar_subtype_source_config["physics_summary"]),
            quantum_encoding=str(haar_subtype_source_config["quantum_encoding"]),
            observable_readout=str(
                haar_subtype_source_config["observable_readout"]
            ),
            tied_mean_dispersion=False,
            haar_subtype_residual=False,
            cross_scale_reupload=False,
            dropout=float(haar_subtype_source_config["dropout"]),
        ).to(device, memory_format=torch.channels_last)
        source_model.load_state_dict(haar_subtype_source_state, strict=True)
        target_state = model.state_dict()
        nonresidual_equal = all(
            key in target_state
            and torch.equal(
                target_state[key].detach().cpu(), value.detach().cpu()
            )
            for key, value in haar_subtype_source_state.items()
        )
        if not nonresidual_equal:
            raise RuntimeError("Subtype selection changed a base-Haar state tensor")
        replay_images = next(iter(normalization_loader))[0][:16].to(
            device, non_blocking=True
        ).contiguous(memory_format=torch.channels_last)
        replay = haar_subtype_exact_replay(source_model, model, replay_images)
        replay["nonresidual_state_bitwise_equal"] = nonresidual_equal
        initialization_report["haar_subtype_residual"][
            "exact_base_replay_pending_selection"
        ] = False
        initialization_report["haar_subtype_residual"]["exact_replay"] = replay
        initialization_report["haar_subtype_residual"][
            "selection_spec_sha256"
        ] = selection_report["selection_spec_sha256"]
        atomic_json(
            output_dir / "initialization_report.json", initialization_report
        )
        print(
            "HAAR_SUBTYPE_SELECTION "
            + json.dumps(selection_report, sort_keys=True),
            flush=True,
        )
        print(
            "HAAR_SUBTYPE_REPLAY " + json.dumps(replay, sort_keys=True),
            flush=True,
        )
        del source_model
    elif model.haar_summary_dim:
        normalization_loader = make_loader(
            CachedNPYDataset(development_cache, train_indices),
            args.batch_size,
            False,
            args.workers,
            args.seed + 3000,
        )
        preserve_morphology = bool(
            initialization_report
            and initialization_report.get("method")
            == "morphology-kd-to-annular-haar-exact-remap"
        )
        fixed_normalization = fit_haar_morphology_normalization(
            model,
            normalization_loader,
            device,
            preserve_morphology=preserve_morphology,
        )
        provenance = {
            "train_membership_sha256": data_report[
                "train_membership_sha256"
            ],
            "development_manifest_sha256": data_report[
                "development_manifest_sha256"
            ],
            "validation_samples_used": 0,
            "official_test_samples_used": 0,
        }
        for name in ("morphology", "haar"):
            fixed_normalization[name].update(provenance)
            atomic_json(
                output_dir / f"{name}_normalization.json",
                fixed_normalization[name],
            )
        fixed_normalization.update(provenance)
        atomic_json(
            output_dir / "fixed_summary_normalization.json",
            fixed_normalization,
        )
        print(
            "FIXED_SUMMARY_NORMALIZATION "
            + json.dumps(fixed_normalization, sort_keys=True),
            flush=True,
        )
    elif model.morphology_feature_dim:
        normalization_loader = make_loader(
            CachedNPYDataset(development_cache, train_indices),
            args.batch_size,
            False,
            args.workers,
            args.seed + 3000,
        )
        morphology_normalization = fit_morphology_normalization(
            model, normalization_loader, device
        )
        morphology_normalization.update(
            {
                "train_membership_sha256": data_report[
                    "train_membership_sha256"
                ],
                "development_manifest_sha256": data_report[
                    "development_manifest_sha256"
                ],
                "validation_samples_used": 0,
                "official_test_samples_used": 0,
            }
        )
        atomic_json(
            output_dir / "morphology_normalization.json",
            morphology_normalization,
        )
        print(
            "MORPHOLOGY_NORMALIZATION "
            + json.dumps(morphology_normalization, sort_keys=True),
            flush=True,
        )
    if args.meridional_readout:
        parameter_report = configure_meridional_optimization_and_report(
            model, args.freeze_meridional_readout_at_zero
        )
    elif args.equatorial_readout:
        parameter_report = configure_equatorial_optimization_and_report(
            model, args.freeze_equatorial_readout_at_zero
        )
    elif args.r2_entanglers:
        parameter_report = configure_r2_optimization_and_report(
            model, args.freeze_r2_entanglers_at_zero
        )
    elif (
        args.haar_subtype_residual
        and args.core == "quantum"
        and not args.haar_subtype_max_envelope
        and not args.freeze_base_for_haar_subtype_residual
    ):
        parameter_report = configure_haar_subtype_optimization_and_report(
            model, args.freeze_haar_subtype_residual_at_zero
        )
    else:
        parameter_report = model.parameter_report()
    if args.shared_late_refinement and (
        int(parameter_report["total"]) != 122599
        or int(parameter_report["shared_late_refinement_gate_trainable"]) != 4
        or int(parameter_report["haar_subtype_residual_trainable"]) != 0
        or int(parameter_report["dispersion_gate_trainable"]) != 0
    ):
        raise RuntimeError(
            f"Invalid shared late-refinement parameter budget: {parameter_report}"
        )
    if args.haar_subtype_residual and (
        int(parameter_report["total"]) != 122610
        or int(parameter_report["haar_subtype_residual_trainable"]) != 15
        or int(parameter_report["dispersion_gate_trainable"]) != 0
        or int(parameter_report["shared_late_refinement_gate_trainable"]) != 0
        or bool(parameter_report["haar_subtype_max_envelope"])
        != bool(args.haar_subtype_max_envelope)
    ):
        raise RuntimeError(
            f"Invalid Haar subtype residual parameter budget: {parameter_report}"
        )
    if args.oof_distillation_artifact and (
        int(parameter_report["total"]) != 122595
        or int(parameter_report["quantum"]) != 88
        or int(parameter_report["haar_subtype_residual_trainable"]) != 0
        or int(parameter_report["dispersion_gate_trainable"]) != 0
        or int(parameter_report["shared_late_refinement_gate_trainable"]) != 0
    ):
        raise RuntimeError(
            f"OOF student parameter contract drifted: {parameter_report}"
        )
    frozen_base_reference = None
    if args.freeze_base_for_haar_subtype_residual:
        residual_weight_key = "haar_subtype_residual.weight"
        frozen_base_reference = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key != residual_weight_key
        }
        model.requires_grad_(False)
        model.haar_subtype_residual.weight.requires_grad_(True)
        optimized_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if optimized_parameters != 15:
            raise RuntimeError(
                "Frozen-base Haar subtype optimization must expose exactly "
                f"15 trainable parameters, got {optimized_parameters}"
            )
        parameter_report.update(
            {
                "inference_total": 122610,
                "optimization_trainable": optimized_parameters,
                "frozen_inference_parameters": 122595,
                "base_parameters_and_buffers_bitwise_frozen": True,
                "optimization_scope": "haar_subtype_residual.weight only",
            }
        )
        initialization_report["frozen_base_optimization"] = {
            "enabled": True,
            "optimized_parameters": optimized_parameters,
            "frozen_inference_parameters": 122595,
            "preserved_state_tensors": len(frozen_base_reference),
            "only_mutable_state_tensor": residual_weight_key,
            "max_preserving_envelope": True,
        }
        atomic_json(
            output_dir / "initialization_report.json", initialization_report
        )
    if args.core in ("quantum", "hybrid"):
        quantum_core = model.core if args.core == "quantum" else model.core.branch_a
        parameter_report.update(
            {f"circuit_{k}": v for k, v in quantum_core.parameter_report().items()}
        )
    atomic_json(output_dir / "parameter_report.json", parameter_report)
    print(f"PARAMETERS {json.dumps(parameter_report, sort_keys=True)}", flush=True)

    distillation_teachers = []
    distillation_report = None
    oof_distillation = None
    if args.oof_distillation_artifact:
        oof_distillation = load_oof_distillation_artifact(
            Path(args.oof_distillation_artifact),
            Path(args.oof_distillation_report),
            train_indices,
            labels,
            development_manifest_sha256,
        )
        distillation_report = {
            "protocol": "two-fold correctness-gated OOF distillation",
            "weight": args.distillation_weight,
            "temperature": args.distillation_temperature,
            "artifact": str(Path(args.oof_distillation_artifact)),
            "artifact_sha256": oof_distillation["artifact_sha256"],
            "report": str(Path(args.oof_distillation_report)),
            "report_sha256": oof_distillation["report_sha256"],
            "routing_counts": oof_distillation["routing_counts"],
            "teacher_provenance": oof_distillation["report"],
            "target_view": "clean cached image OOF logits",
            "student_view": "standard stochastic training augmentation",
            "gate_uses_training_label": True,
            "ungated_rows": "full supervised CE only",
            "train_membership_sha256": data_report["train_membership_sha256"],
            "development_manifest_sha256": development_manifest_sha256,
            "canonical_development_validation_samples_used_for_targets": 0,
            "official_test_opened": False,
        }
        atomic_json(output_dir / "distillation_report.json", distillation_report)
        print(
            f"OOF_DISTILLATION {json.dumps(distillation_report, sort_keys=True)}",
            flush=True,
        )
    elif args.distillation_teacher_checkpoint:
        teacher_records = []
        for teacher_checkpoint_path in args.distillation_teacher_checkpoint:
            teacher_path = Path(teacher_checkpoint_path)
            teacher_directory = teacher_path.parent
            teacher_config = json.loads(
                (teacher_directory / "config.json").read_text()
            )
            teacher_data_report = json.loads(
                (teacher_directory / "data_report.json").read_text()
            )
            teacher_summary = json.loads(
                (teacher_directory / "summary.json").read_text()
            )
            if teacher_config.get("evaluate_test") is not False:
                raise RuntimeError(
                    "Distillation teacher config must keep test evaluation disabled"
                )
            if teacher_data_report.get("official_test_cache_opened") is not False:
                raise RuntimeError("Distillation teacher must not have opened test")
            if "test" in teacher_summary or teacher_summary.get(
                "official_test_evaluated"
            ):
                raise RuntimeError("Distillation teacher summary contains test evaluation")
            if teacher_data_report.get("class_names") != data_report["class_names"]:
                raise RuntimeError("Distillation teacher class order differs")
            if teacher_config.get("image_size") != args.image_size:
                raise RuntimeError("Distillation teacher image size differs")
            if teacher_data_report.get("train_membership_sha256") != data_report[
                "train_membership_sha256"
            ]:
                raise RuntimeError("Distillation teacher used different training images")
            if teacher_data_report.get("development_manifest_sha256") != data_report[
                "development_manifest_sha256"
            ]:
                raise RuntimeError(
                    "Distillation teacher used a different development manifest"
                )
            with np.load(teacher_directory / "split_indices.npz") as teacher_split:
                if not np.array_equal(teacher_split["train"], train_indices):
                    raise RuntimeError("Distillation teacher training split differs")
                if not np.array_equal(teacher_split["val"], val_indices):
                    raise RuntimeError("Distillation teacher validation split differs")
            teacher = D4OrbitClassifier(
                num_classes=len(class_names),
                heads=teacher_config["heads"],
                reuploads=teacher_config["reuploads"],
                core=teacher_config["core"],
                include_context=teacher_config["include_context"],
                encoder_variant=teacher_config["encoder_variant"],
                physics_variant=teacher_config["physics_variant"],
                physics_summary=teacher_config["physics_summary"],
                quantum_encoding=teacher_config["quantum_encoding"],
                observable_readout=teacher_config["observable_readout"],
                tied_mean_dispersion=bool(
                    teacher_config.get("tied_mean_dispersion", False)
                ),
                haar_subtype_residual=bool(
                    teacher_config.get("haar_subtype_residual", False)
                ),
                haar_subtype_max_envelope=bool(
                    teacher_config.get("haar_subtype_max_envelope", False)
                ),
                shared_late_refinement=bool(
                    teacher_config.get("shared_late_refinement", False)
                ),
                cross_scale_reupload=bool(
                    teacher_config.get("cross_scale_reupload", False)
                ),
                r2_entanglers=bool(
                    teacher_config.get("r2_entanglers", False)
                ),
                equatorial_readout=bool(
                    teacher_config.get("equatorial_readout", False)
                ),
                meridional_readout=bool(
                    teacher_config.get("meridional_readout", False)
                ),
                dropout=teacher_config["dropout"],
            ).to(device, memory_format=torch.channels_last)
            teacher_checkpoint = torch.load(teacher_path, map_location="cpu")
            teacher.load_state_dict(
                teacher_checkpoint.get("model", teacher_checkpoint), strict=True
            )
            teacher_parameters = teacher.parameter_report()["total"]
            teacher.eval()
            teacher.requires_grad_(False)
            distillation_teachers.append(teacher)
            teacher_records.append(
                {
                    "checkpoint": str(teacher_path),
                    "checkpoint_sha256": file_sha256(teacher_path),
                    "source_epoch": teacher_checkpoint.get("epoch"),
                    "parameters": teacher_parameters,
                    "core": teacher_config["core"],
                    "encoder_variant": teacher_config["encoder_variant"],
                    "include_context": teacher_config["include_context"],
                    "tied_mean_dispersion": bool(
                        teacher_config.get("tied_mean_dispersion", False)
                    ),
                    "haar_subtype_residual": bool(
                        teacher_config.get("haar_subtype_residual", False)
                    ),
                    "haar_subtype_max_envelope": bool(
                        teacher_config.get("haar_subtype_max_envelope", False)
                    ),
                    "shared_late_refinement": bool(
                        teacher_config.get("shared_late_refinement", False)
                    ),
                    "r2_entanglers": bool(
                        teacher_config.get("r2_entanglers", False)
                    ),
                    "equatorial_readout": bool(
                        teacher_config.get("equatorial_readout", False)
                    ),
                    "meridional_readout": bool(
                        teacher_config.get("meridional_readout", False)
                    ),
                }
            )
        distillation_report = {
            "weight": args.distillation_weight,
            "temperature": args.distillation_temperature,
            "ensemble_method": "mean softened class probabilities",
            "teacher_count": len(distillation_teachers),
            "teacher_parameters_total": sum(
                record["parameters"] for record in teacher_records
            ),
            "teachers": teacher_records,
            "train_membership_sha256": data_report["train_membership_sha256"],
            "development_manifest_sha256": data_report[
                "development_manifest_sha256"
            ],
            "same_training_indices": True,
            "same_validation_indices": True,
            "official_test_opened": False,
        }
        atomic_json(output_dir / "distillation_report.json", distillation_report)
        print(
            f"DISTILLATION {json.dumps(distillation_report, sort_keys=True)}",
            flush=True,
        )

    core_initial = torch.cat([p.detach().float().flatten().cpu() for p in model.core.parameters()])
    haar_subtype_initial = (
        model.haar_subtype_residual.weight.detach().float().cpu().clone()
        if model.haar_subtype_residual is not None
        else None
    )
    shared_refinement_initial = (
        model.encoder.shared_refinement_gates.detach().float().cpu().clone()
        if model.encoder.shared_refinement_gates is not None
        else None
    )
    r2_initial = (
        model.core.r2_params.detach().float().cpu().clone()
        if getattr(model.core, "r2_params", None) is not None
        else None
    )
    equatorial_initial = (
        model.core.readout_phases.detach().float().cpu().clone()
        if getattr(model.core, "readout_phases", None) is not None
        else None
    )
    meridional_initial = (
        model.core.meridional_phases.detach().float().cpu().clone()
        if getattr(model.core, "meridional_phases", None) is not None
        else None
    )
    branch_initial = None
    if args.core in ("hybrid", "classical-fusion"):
        branch_initial = {
            name: torch.cat(
                [
                    parameter.detach().float().flatten().cpu()
                    for parameter in branch.parameters()
                ]
            )
            for name, branch in (
                ("branch_a", model.core.branch_a),
                ("branch_b", model.core.branch_b),
            )
        }
    encoder_parameters, head_parameters, core_parameters = (
        optimizer_parameter_groups(model)
    )
    encoder_lr = (
        args.encoder_learning_rate
        if args.encoder_learning_rate is not None
        else args.learning_rate
    )
    optimizer = torch.optim.AdamW(
        (
            {"params": encoder_parameters, "lr": encoder_lr},
            {"params": head_parameters, "lr": args.learning_rate},
            {"params": core_parameters, "lr": args.core_learning_rate},
        ),
        weight_decay=args.weight_decay,
    )
    optimizer_parameter_count = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    expected_optimizer_parameter_count = int(
        parameter_report.get(
            "optimization_trainable_total",
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )
    )
    if optimizer_parameter_count != expected_optimizer_parameter_count:
        raise RuntimeError(
            "Optimizer parameter groups do not match the declared trainable "
            f"budget: actual={optimizer_parameter_count} "
            f"expected={expected_optimizer_parameter_count}"
        )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, min(3 * len(train_loader), int(0.10 * total_steps)))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history = []
    best_accuracy = -1.0
    best_epoch = -1
    stale_epochs = 0
    global_step = 0
    stochastic_trace = []
    training_rng_report = None
    if args.training_rng_seed is not None:
        # Model construction and the two core implementations consume different
        # numbers of random values.  Reset only after every initialization and
        # optimizer setup step so paired arms enter epoch 1 from the same RNG
        # state without changing the locked subset/loader seed.
        seed_everything(args.training_rng_seed, args.deterministic)
        training_rng_report = {
            "schema_version": 1,
            "training_rng_seed": int(args.training_rng_seed),
            "loader_and_subset_seed": int(args.seed),
            "reset_point": "immediately before epoch 1 after optimizer/scheduler setup",
            "initial_state_sha256": training_rng_state_digests(),
            "stochastic_trace_enabled": bool(args.save_stochastic_trace),
        }
        atomic_json(output_dir / "training_rng_report.json", training_rng_report)
        print(
            "TRAINING_RNG " + json.dumps(training_rng_report, sort_keys=True),
            flush=True,
        )
    run_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_base_for_haar_subtype_residual:
            model.eval()
            model.haar_subtype_residual.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        seen = 0
        core_grad_sum = 0.0
        branch_loss_sum = 0.0
        branch_correct = [0, 0]
        distillation_loss_sum = 0.0
        supervised_loss_sum = 0.0
        teacher_correct = 0
        teacher_student_agreement = 0
        oof_gated_samples = 0
        mixup_samples = 0
        mixup_anchor_weight_sum = 0.0
        epoch_sample_digest = hashlib.sha256()
        epoch_sample_count = 0
        epoch_batch_trace = []
        for batch_index, (images, targets, sample_indices) in enumerate(train_loader):
            if args.save_stochastic_trace:
                sample_array = np.asarray(sample_indices, dtype=np.int64)
                sample_bytes = sample_array.tobytes(order="C")
                epoch_sample_digest.update(sample_bytes)
                epoch_sample_count += int(sample_array.size)
                epoch_batch_trace.append(
                    {
                        "batch_index": int(batch_index),
                        "batch_size": int(sample_array.size),
                        "sample_indices_sha256": _sha256_bytes(sample_bytes),
                        "pre_augmentation_rng_sha256": training_rng_state_digests(),
                    }
                )
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            images = translate_batch(
                images, args.max_translation_pixels, args.translation_probability
            )
            images = physics_augment_batch(
                images,
                photon_probability=args.photon_noise_probability,
                photon_count_min=args.photon_count_min,
                photon_count_max=args.photon_count_max,
                psf_probability=args.psf_blur_probability,
                read_noise_std=args.read_noise_std,
            )
            targets = targets.to(device, non_blocking=True)
            oof_morphology_logits = oof_spatial_logits = None
            if oof_distillation is not None:
                oof_morphology_logits = oof_distillation[
                    "morphology_logits"
                ][sample_indices].to(device, non_blocking=True)
                oof_spatial_logits = oof_distillation["spatial_logits"][
                    sample_indices
                ].to(device, non_blocking=True)
            mixup_target_probabilities = None
            if args.subtype_mixup_probability:
                (
                    images,
                    mixup_target_probabilities,
                    batch_mixup_samples,
                    batch_anchor_weight,
                ) = subtype_mixup_batch(
                    images,
                    targets,
                    probability=args.subtype_mixup_probability,
                    alpha=args.subtype_mixup_alpha,
                    num_classes=len(class_names),
                )
                mixup_samples += batch_mixup_samples
                mixup_anchor_weight_sum += (
                    batch_mixup_samples * batch_anchor_weight
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.branch_loss_weight:
                    logits, training_auxiliary = model(images, return_aux=True)
                else:
                    logits = model(images)
                    training_auxiliary = None
                teacher_logits = None
                if distillation_teachers:
                    with torch.no_grad():
                        teacher_logits = tuple(
                            teacher(images) for teacher in distillation_teachers
                        )
                primary_per_sample = None
                if mixup_target_probabilities is None:
                    if oof_distillation is not None:
                        primary_per_sample = F.cross_entropy(
                            logits,
                            targets,
                            label_smoothing=args.label_smoothing,
                            reduction="none",
                        )
                        primary_loss = primary_per_sample.mean()
                    else:
                        primary_loss = F.cross_entropy(
                            logits, targets, label_smoothing=args.label_smoothing
                        )
                else:
                    primary_loss = soft_target_cross_entropy(
                        logits,
                        mixup_target_probabilities,
                        label_smoothing=args.label_smoothing,
                    )
                if args.hierarchical_loss_weight:
                    auxiliary_loss = hierarchical_model_i_loss(
                        logits, targets, label_smoothing=args.label_smoothing
                    )
                    loss = (
                        (1.0 - args.hierarchical_loss_weight) * primary_loss
                        + args.hierarchical_loss_weight * auxiliary_loss
                    )
                else:
                    loss = primary_loss
                supervised_loss = loss
                distillation_loss = None
                oof_valid = oof_teacher_probabilities = None
                if oof_distillation is not None:
                    (
                        oof_distillation_per_sample,
                        oof_valid,
                        oof_teacher_probabilities,
                    ) = correctness_gated_oof_distillation_loss(
                        logits,
                        oof_morphology_logits,
                        oof_spatial_logits,
                        targets,
                        args.distillation_temperature,
                    )
                    mixed_per_sample = torch.where(
                        oof_valid,
                        (1.0 - args.distillation_weight) * primary_per_sample
                        + args.distillation_weight
                        * oof_distillation_per_sample,
                        primary_per_sample,
                    )
                    loss = mixed_per_sample.mean()
                    distillation_loss = (
                        oof_distillation_per_sample[oof_valid].mean()
                        if bool(oof_valid.any())
                        else oof_distillation_per_sample.sum() * 0.0
                    )
                elif teacher_logits is not None:
                    distillation_loss = knowledge_distillation_loss(
                        logits,
                        teacher_logits,
                        args.distillation_temperature,
                    )
                    loss = (
                        (1.0 - args.distillation_weight) * supervised_loss
                        + args.distillation_weight * distillation_loss
                    )
                branch_loss = None
                if training_auxiliary is not None:
                    branch_logits = training_auxiliary["branch_logits"]
                    branch_loss = 0.5 * sum(
                        (
                            F.cross_entropy(
                                branch_output,
                                targets,
                                label_smoothing=args.label_smoothing,
                            )
                            if mixup_target_probabilities is None
                            else soft_target_cross_entropy(
                                branch_output,
                                mixup_target_probabilities,
                                label_smoothing=args.label_smoothing,
                            )
                        )
                        for branch_output in branch_logits
                    )
                    loss = loss + args.branch_loss_weight * branch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            grad_sq = 0.0
            for parameter in model.core.parameters():
                if parameter.grad is not None:
                    grad_sq += float(parameter.grad.detach().float().square().sum())
            core_grad_sum += math.sqrt(grad_sq)
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch = targets.numel()
            running_loss += float(loss.detach()) * batch
            supervised_loss_sum += float(supervised_loss.detach()) * batch
            correct += int((logits.argmax(1) == targets).sum())
            if distillation_loss is not None:
                if oof_distillation is not None:
                    valid_count = int(oof_valid.sum())
                    oof_gated_samples += valid_count
                    distillation_loss_sum += (
                        float(distillation_loss.detach()) * valid_count
                    )
                    teacher_probabilities = oof_teacher_probabilities
                else:
                    distillation_loss_sum += float(distillation_loss.detach()) * batch
                    teacher_probabilities = torch.stack(
                        [F.softmax(output.float(), dim=1) for output in teacher_logits]
                    ).mean(dim=0)
                teacher_predictions = teacher_probabilities.argmax(1)
                comparison_mask = (
                    oof_valid
                    if oof_distillation is not None
                    else torch.ones_like(targets, dtype=torch.bool)
                )
                teacher_correct += int(
                    ((teacher_predictions == targets) & comparison_mask).sum()
                )
                teacher_student_agreement += int(
                    (
                        (teacher_predictions == logits.argmax(1))
                        & comparison_mask
                    ).sum()
                )
            if branch_loss is not None:
                branch_loss_sum += float(branch_loss.detach()) * batch
                for branch_index, branch_output in enumerate(branch_logits):
                    branch_correct[branch_index] += int(
                        (branch_output.argmax(1) == targets).sum()
                    )
            seen += batch

        validation_due = (
            epoch + 1 == args.epochs
            if (
                args.oof_teacher_fold_index is not None
                or args.fixed_final_validation_only
            )
            else (
                epoch == 0
                or (epoch + 1) % args.validation_interval == 0
                or epoch + 1 == args.epochs
            )
        )
        if validation_due:
            val_metrics, val_labels, val_logits, val_sample_indices = evaluate(
                model, val_loader, device, class_names
            )
            if args.save_last_validation_predictions:
                np.savez_compressed(
                    output_dir / "last_validation_predictions.npz",
                    indices=val_sample_indices,
                    labels=val_labels,
                    logits=val_logits,
                    probabilities=softmax_numpy(val_logits),
                    epoch=np.asarray(epoch + 1, dtype=np.int64),
                )
        else:
            val_metrics = val_labels = val_logits = val_sample_indices = None
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / seen,
            "train_accuracy": correct / seen,
            "validation": val_metrics,
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rate": optimizer.param_groups[1]["lr"],
            "core_learning_rate": optimizer.param_groups[2]["lr"],
            "mean_core_gradient_norm": core_grad_sum / max(len(train_loader), 1),
            "epoch_seconds": time.time() - epoch_start,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        if args.core in ("hybrid", "classical-fusion"):
            record["mixing_weight_branch_a"] = float(
                model.core.mixing_weight.detach()
            )
        if args.branch_loss_weight:
            record["train_branch_loss"] = branch_loss_sum / seen
            record["train_branch_accuracy"] = [
                branch_correct[0] / seen,
                branch_correct[1] / seen,
            ]
        if distillation_teachers or oof_distillation is not None:
            record["train_supervised_loss"] = supervised_loss_sum / seen
            distillation_denominator = (
                oof_gated_samples
                if oof_distillation is not None
                else seen
            )
            record["train_distillation_loss"] = (
                distillation_loss_sum / max(distillation_denominator, 1)
            )
            record["teacher_train_accuracy"] = (
                teacher_correct / max(distillation_denominator, 1)
            )
            record["teacher_student_train_agreement"] = (
                teacher_student_agreement / max(distillation_denominator, 1)
            )
            if oof_distillation is not None:
                record["oof_gated_samples"] = oof_gated_samples
                record["oof_gated_fraction"] = oof_gated_samples / seen
        if args.subtype_mixup_probability:
            record["train_subtype_mixup_samples"] = mixup_samples
            record["train_subtype_mixup_fraction"] = mixup_samples / seen
            record["train_subtype_mixup_mean_anchor_weight"] = (
                mixup_anchor_weight_sum / max(mixup_samples, 1)
            )
        history.append(record)
        atomic_json(output_dir / "history.json", history)
        if args.save_stochastic_trace:
            stochastic_trace.append(
                {
                    "epoch": int(epoch + 1),
                    "ordered_sample_indices_sha256": epoch_sample_digest.hexdigest(),
                    "sample_count": int(epoch_sample_count),
                    "batch_count": int(len(epoch_batch_trace)),
                    "batches": epoch_batch_trace,
                }
            )
            atomic_json(
                output_dir / "stochastic_trace.json", stochastic_trace
            )
        atomic_checkpoint(
            output_dir / "last.pt",
            {"model": model.state_dict(), "epoch": epoch + 1, "record": record},
        )
        print(f"EPOCH {json.dumps(record, sort_keys=True)}", flush=True)

        if validation_due and val_metrics["accuracy"] > best_accuracy + 1e-12:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch + 1
            stale_epochs = 0
            atomic_checkpoint(
                output_dir / "best.pt",
                {"model": model.state_dict(), "epoch": best_epoch, "record": record},
            )
            np.savez_compressed(
                output_dir / "best_validation_predictions.npz",
                indices=val_sample_indices,
                labels=val_labels,
                logits=val_logits,
                probabilities=softmax_numpy(val_logits),
            )
        elif validation_due:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"EARLY_STOP epoch={epoch + 1} best_epoch={best_epoch}", flush=True)
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    frozen_base_audit = None
    if frozen_base_reference is not None:
        final_state = model.state_dict()
        drifted = [
            key
            for key, expected in frozen_base_reference.items()
            if key not in final_state
            or not torch.equal(final_state[key].detach().cpu(), expected)
        ]
        if drifted:
            raise RuntimeError(
                "Frozen-base Haar subtype training changed protected state: "
                f"{drifted[:8]}"
            )
        frozen_base_audit = {
            "bitwise_equal": True,
            "protected_state_tensors": len(frozen_base_reference),
            "drifted_state_tensors": 0,
            "only_optimized_state_tensor": "haar_subtype_residual.weight",
        }
    final_val, val_labels, val_logits, val_sample_indices = evaluate(
        model, val_loader, device, class_names
    )
    symmetry = symmetry_audit(model, val_loader, device)
    atomic_json(output_dir / "symmetry_audit.json", symmetry)
    core_final = torch.cat([p.detach().float().flatten().cpu() for p in model.core.parameters()])
    core_update_norm = float((core_final - core_initial).norm())
    summary = {
        "best_epoch": best_epoch,
        "validation": final_val,
        "parameters": parameter_report,
        "core_update_l2": core_update_norm,
        "symmetry": symmetry["summary"],
        "wall_seconds": time.time() - run_start,
        "official_test_evaluated": bool(args.evaluate_test),
        "initialization": initialization_report,
    }
    if model.haar_subtype_residual is not None:
        subtype_update = haar_subtype_residual_update_record(
            model,
            haar_subtype_initial,
            args.freeze_haar_subtype_residual_at_zero,
        )
        summary["haar_subtype_residual"] = {
            "trainable_parameters": 15,
            "optimization_trainable": subtype_update[
                "optimization_trainable"
            ],
            "frozen_at_zero_control": subtype_update[
                "frozen_at_zero_control"
            ],
            "weights_exact_zero": subtype_update["weights_exact_zero"],
            "weight_update_l2": subtype_update["weight_update_l2"],
            "weight_l2": subtype_update["weight_l2"],
            "weights": subtype_update["weights"],
            "selected_indices": model.haar_subtype_residual.selected_indices.tolist(),
            "selection_spec_sha256": initialization_report[
                "haar_subtype_residual"
            ]["selection_spec_sha256"],
            "logit_path": (
                "max-preserving subtype envelope"
                if args.haar_subtype_max_envelope
                else "[+delta,-delta,0]"
            ),
            "max_preserving_envelope": bool(args.haar_subtype_max_envelope),
            "frozen_base_audit": frozen_base_audit,
        }
    if model.encoder.shared_refinement_gates is not None:
        shared_final = (
            model.encoder.shared_refinement_gates.detach().float().cpu()
        )
        summary["shared_late_refinement"] = {
            "trainable_parameters": 4,
            "gate_update_l2": float(
                (shared_final - shared_refinement_initial).norm()
            ),
            "gate_l2": float(shared_final.norm()),
            "gates": shared_final.tolist(),
            "shared_block_applications": [5, 5, 7, 7],
            "inference_path": "shared encoder refinement -> projected angles -> core only",
        }
    if getattr(model.core, "r2_params", None) is not None:
        r2_final = model.core.r2_params.detach().float().cpu()
        r2_update = float((r2_final - r2_initial).norm())
        if args.freeze_r2_entanglers_at_zero and (
            r2_update != 0.0
            or not torch.equal(r2_final, torch.zeros_like(r2_final))
        ):
            raise RuntimeError("Frozen R2 control changed a zero entangler angle")
        if not args.freeze_r2_entanglers_at_zero and r2_update <= 0.0:
            raise RuntimeError("Trainable R2 entangler angles received no update")
        summary["r2_entanglers"] = {
            "parameters": int(r2_final.numel()),
            "optimization_trainable": int(
                r2_final.numel() if model.core.r2_params.requires_grad else 0
            ),
            "frozen_at_zero_control": bool(args.freeze_r2_entanglers_at_zero),
            "angle_update_l2": r2_update,
            "angle_l2": float(r2_final.norm()),
            "angles": r2_final.tolist(),
            "edge_family": "R2 half-turn complete left-Cayley orbit",
            "pauli_rotations": ["ZZ", "XX"],
        }
    if getattr(model.core, "readout_phases", None) is not None:
        equatorial_final = model.core.readout_phases.detach().float().cpu()
        equatorial_update = float(
            (equatorial_final - equatorial_initial).norm()
        )
        if args.freeze_equatorial_readout_at_zero and (
            equatorial_update != 0.0
            or not torch.equal(
                equatorial_final, torch.zeros_like(equatorial_final)
            )
        ):
            raise RuntimeError(
                "Frozen EQR-16 control changed a zero measurement phase"
            )
        if (
            not args.freeze_equatorial_readout_at_zero
            and equatorial_update <= 0.0
        ):
            raise RuntimeError(
                "Trainable EQR-16 measurement phases received no update"
            )
        summary["equatorial_readout"] = {
            "parameters": int(equatorial_final.numel()),
            "optimization_trainable": int(
                equatorial_final.numel()
                if model.core.readout_phases.requires_grad
                else 0
            ),
            "frozen_at_zero_control": bool(
                args.freeze_equatorial_readout_at_zero
            ),
            "phase_update_l2": equatorial_update,
            "phase_l2": float(equatorial_final.norm()),
            "phases": equatorial_final.tolist(),
            "settings_per_head": ["local", "R", "R2", "S"],
            "observable": "P(phi)=cos(phi)X+sin(phi)Y",
            "state_preparation_parameters": 88,
        }
    if getattr(model.core, "meridional_phases", None) is not None:
        meridional_final = (
            model.core.meridional_phases.detach().float().cpu()
        )
        meridional_update = float(
            (meridional_final - meridional_initial).norm()
        )
        if args.freeze_meridional_readout_at_zero and (
            meridional_update != 0.0
            or not torch.equal(
                meridional_final, torch.zeros_like(meridional_final)
            )
        ):
            raise RuntimeError(
                "Frozen meridional-16 control changed a zero measurement phase"
            )
        if (
            not args.freeze_meridional_readout_at_zero
            and meridional_update <= 0.0
        ):
            raise RuntimeError(
                "Trainable meridional-16 measurement phases received no update"
            )
        summary["meridional_readout"] = {
            "parameters": int(meridional_final.numel()),
            "optimization_trainable": int(
                meridional_final.numel()
                if model.core.meridional_phases.requires_grad
                else 0
            ),
            "frozen_at_zero_control": bool(
                args.freeze_meridional_readout_at_zero
            ),
            "phase_update_l2": meridional_update,
            "phase_l2": float(meridional_final.norm()),
            "phases": meridional_final.tolist(),
            "settings_per_head": ["local", "R", "R2", "S"],
            "observable": "P(phi)=cos(phi)X+sin(phi)Z",
            "mixed_pair_sector": ["XZ", "ZX"],
            "state_preparation_parameters": 88,
        }
    if distillation_report is not None:
        summary["distillation"] = distillation_report
    if args.core in ("hybrid", "classical-fusion"):
        branch_validation = evaluate_parallel_branches(
            model, val_loader, device, class_names
        )
        branch_updates = {}
        for name, branch in (
            ("branch_a", model.core.branch_a),
            ("branch_b", model.core.branch_b),
        ):
            final = torch.cat(
                [
                    parameter.detach().float().flatten().cpu()
                    for parameter in branch.parameters()
                ]
            )
            branch_updates[name] = float((final - branch_initial[name]).norm())
        summary["parallel_core"] = {
            "architecture": model.core.architecture,
            "mixing_weight_branch_a": float(model.core.mixing_weight.detach()),
            "branch_update_l2": branch_updates,
            "branch_validation": {
                "branch_a": branch_validation[0],
                "branch_b": branch_validation[1],
            },
        }

    if args.evaluate_test:
        test_loader = make_loader(
            CachedNPYDataset(test_cache),
            args.batch_size,
            False,
            args.workers,
            args.split_seed,
        )
        test_metrics, test_labels, test_logits, test_indices = evaluate(
            model, test_loader, device, class_names
        )
        summary["test"] = test_metrics
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            indices=test_indices,
            labels=test_labels,
            logits=test_logits,
            probabilities=softmax_numpy(test_logits),
        )

    atomic_json(output_dir / "summary.json", summary)
    if args.oof_teacher_fold_index is not None:
        if best_epoch != args.epochs:
            raise RuntimeError(
                "OOF teacher checkpoint was not selected exclusively at the fixed final epoch"
            )
        last_checkpoint = torch.load(output_dir / "last.pt", map_location="cpu")
        best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
        if (
            int(last_checkpoint.get("epoch", -1)) != args.epochs
            or int(best_checkpoint.get("epoch", -1)) != args.epochs
        ):
            raise RuntimeError("OOF fixed-final checkpoint epoch drifted")
        last_state = last_checkpoint.get("model", last_checkpoint)
        best_state = best_checkpoint.get("model", best_checkpoint)
        if set(last_state) != set(best_state) or not all(
            torch.equal(last_state[key].cpu(), best_state[key].cpu())
            for key in last_state
        ):
            raise RuntimeError("OOF final and best state tensors differ")
        prediction_path = output_dir / "last_validation_predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as predictions:
            if int(predictions["epoch"]) != args.epochs:
                raise RuntimeError("OOF prediction artifact is not from the fixed final epoch")
            if not np.array_equal(
                np.sort(predictions["indices"]), np.sort(val_indices)
            ):
                raise RuntimeError("OOF prediction artifact membership drifted")
        fixed_final_report = {
            "schema_version": 1,
            "protocol": "fixed-final-oof-teacher-v1",
            "epoch": args.epochs,
            "checkpoint_selection": "single held-out evaluation at fixed final epoch",
            "last_checkpoint_sha256": file_sha256(output_dir / "last.pt"),
            "best_checkpoint_sha256": file_sha256(output_dir / "best.pt"),
            "prediction_sha256": file_sha256(prediction_path),
            "training_fold": oof_teacher_report["training_fold"],
            "prediction_fold": oof_teacher_report["prediction_fold"],
            "training_fold_membership_sha256": oof_teacher_report[
                "training_fold_membership_sha256"
            ],
            "prediction_fold_membership_sha256": oof_teacher_report[
                "prediction_fold_membership_sha256"
            ],
            "full_half_membership_sha256": oof_teacher_report[
                "full_half_membership_sha256"
            ],
            "canonical_development_validation_samples_used": 0,
            "official_test_samples_used": 0,
            "final_and_best_state_tensors_bitwise_equal": True,
        }
        atomic_json(output_dir / "fixed_final_oof_report.json", fixed_final_report)
        print(
            "FIXED_FINAL_OOF "
            + json.dumps(fixed_final_report, sort_keys=True),
            flush=True,
        )
    print(f"FINAL_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
