"""Build SHA-sealed two-fold OOF morphology/spatial distillation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .data import index_membership_sha256


SCHEMA_VERSION = 1
PROTOCOL = "two-fold-correctness-gated-morphology-spatial-v1"
CLASSES = ["axion", "cdm", "no_sub"]
EXPECTED_MANIFEST = "c04a3c62afebe3f660ffaad4333b6632471a91c6f5f239f84e68b4b94c330025"
EXPECTED_PARENT = "571d23ced25095cf0cfb57216654f9b7be289b0589a95489a5a815a866aaee71"
EXPECTED_EPOCH = 40
TEMPERATURE = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_json(path: Path) -> Dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def require_run_directory(path: Path) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"OOF run directory is missing or unsafe: {path}")
    path = path.resolve(strict=True)
    required = (
        "config.json",
        "data_report.json",
        "parameter_report.json",
        "split_indices.npz",
        "last.pt",
        "last_validation_predictions.npz",
        "fixed_final_oof_report.json",
    )
    for name in required:
        artifact = path / name
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError(f"OOF run is missing a safe {name}: {path}")
    return path


def validate_teacher_run(
    path: Path,
    architecture: str,
    heldout_fold: int,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    path = require_run_directory(path)
    config = read_json(path / "config.json")
    data = read_json(path / "data_report.json")
    parameters = read_json(path / "parameter_report.json")
    fixed = read_json(path / "fixed_final_oof_report.json")
    expected_architecture = {
        "morphology": {
            "encoder_variant": "deep-se-morph",
            "physics_summary": "moments-morphology",
            "heads": 4,
            "reuploads": 2,
            "parameters": 122573,
        },
        "spatial": {
            "encoder_variant": "micro-stat",
            "physics_summary": "moments",
            "heads": 4,
            "reuploads": 3,
            "parameters": 122573,
        },
    }[architecture]
    expected_config = {
        "core": "quantum",
        "physics_variant": "base",
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "evaluate_test": False,
        "deterministic": True,
        "epochs": EXPECTED_EPOCH,
        "oof_teacher_fold_index": 1 - heldout_fold,
        "save_last_validation_predictions": True,
        "init_backbone_checkpoint": None,
        "init_compatible_backbone_checkpoint": None,
        "init_full_checkpoint": None,
        "distillation_teacher_checkpoint": None,
        "oof_distillation_artifact": None,
        "image_size": 96,
        "split_seed": 42,
        "val_fraction": 0.2,
        "max_train_per_class": 11667,
        "train_subset_protocol": "hash-v1",
    }
    expected_config.update(
        {key: expected_architecture[key] for key in ("encoder_variant", "physics_summary", "heads", "reuploads")}
    )
    drift = {
        key: (config.get(key), expected)
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"{architecture} heldout-{heldout_fold} config drifted: {drift}")
    if int(parameters.get("total", -1)) != expected_architecture["parameters"]:
        raise RuntimeError("OOF teacher parameter count drifted")
    if int(parameters.get("quantum", -1)) <= 0:
        raise RuntimeError("OOF teacher lost its quantum core")
    oof = data.get("oof_teacher")
    if not isinstance(oof, dict):
        raise RuntimeError("OOF teacher data report is missing fold provenance")
    required_oof = {
        "protocol": "stratified-hash-two-fold-v1",
        "fold_count": 2,
        "training_fold": 1 - heldout_fold,
        "prediction_fold": heldout_fold,
        "full_half_train_size": 35001,
        "full_half_membership_sha256": EXPECTED_PARENT,
        "canonical_development_validation_samples_used": 0,
        "official_test_samples_used": 0,
        "checkpoint_selection_for_oof": "fixed final epoch only",
    }
    if any(oof.get(key) != value for key, value in required_oof.items()):
        raise RuntimeError("OOF teacher fold/data contract drifted")
    if (
        data.get("development_manifest_sha256") != EXPECTED_MANIFEST
        or data.get("official_test_cache_opened") is not False
        or data.get("class_names") != CLASSES
    ):
        raise RuntimeError("OOF teacher manifest, test lock, or class order drifted")
    with np.load(path / "split_indices.npz", allow_pickle=False) as split:
        train = np.asarray(split["train"], dtype=np.int64)
        heldout = np.asarray(split["val"], dtype=np.int64)
        parent = np.asarray(split["full_half_train"], dtype=np.int64)
        canonical_val = np.asarray(split["canonical_val_unused"], dtype=np.int64)
    if (
        index_membership_sha256(parent) != EXPECTED_PARENT
        or np.intersect1d(train, heldout).size
        or not np.array_equal(np.sort(np.concatenate((train, heldout))), np.sort(parent))
        or len(canonical_val) != 17504
        or np.intersect1d(parent, canonical_val).size
    ):
        raise RuntimeError("OOF teacher split arrays violate partition or isolation")
    prediction_path = path / "last_validation_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as prediction:
        indices = np.asarray(prediction["indices"], dtype=np.int64)
        labels = np.asarray(prediction["labels"], dtype=np.int64)
        logits = np.asarray(prediction["logits"], dtype=np.float32)
        epoch = int(prediction["epoch"])
    order = np.argsort(indices)
    indices, labels, logits = indices[order], labels[order], logits[order]
    if (
        epoch != EXPECTED_EPOCH
        or not np.array_equal(indices, np.sort(heldout))
        or labels.shape != (len(indices),)
        or logits.shape != (len(indices), 3)
        or not np.isfinite(logits).all()
    ):
        raise RuntimeError("OOF teacher fixed-final predictions are invalid")
    required_fixed = {
        "schema_version": 1,
        "protocol": "fixed-final-oof-teacher-v1",
        "epoch": EXPECTED_EPOCH,
        "training_fold": 1 - heldout_fold,
        "prediction_fold": heldout_fold,
        "full_half_membership_sha256": EXPECTED_PARENT,
        "canonical_development_validation_samples_used": 0,
        "official_test_samples_used": 0,
        "prediction_sha256": sha256_file(prediction_path),
        "last_checkpoint_sha256": sha256_file(path / "last.pt"),
        "final_and_best_state_tensors_bitwise_equal": True,
    }
    if any(fixed.get(key) != value for key, value in required_fixed.items()):
        raise RuntimeError("OOF teacher fixed-final SHA contract drifted")
    record = {
        "architecture": architecture,
        "heldout_fold": heldout_fold,
        "run_dir": str(path),
        "train_size": int(len(train)),
        "heldout_size": int(len(heldout)),
        "train_membership_sha256": index_membership_sha256(train),
        "heldout_membership_sha256": index_membership_sha256(heldout),
        "checkpoint_sha256": fixed["last_checkpoint_sha256"],
        "prediction_sha256": fixed["prediction_sha256"],
        "logit_content_sha256": array_sha256(logits.astype("<f4", copy=False)),
        "fixed_epoch": EXPECTED_EPOCH,
        "correct": int((logits.argmax(1) == labels).sum()),
        "accuracy": float((logits.argmax(1) == labels).mean()),
        "canonical_validation_samples_used": 0,
        "official_test_samples_used": 0,
    }
    return record, {"indices": indices, "labels": labels, "logits": logits}


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits.astype(np.float64) / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.building-{os.getpid()}-{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build(args: argparse.Namespace) -> None:
    output_artifact = Path(args.output_artifact)
    output_report = Path(args.output_report)
    if not output_artifact.is_absolute() or not output_report.is_absolute():
        raise ValueError("OOF outputs must be absolute paths")
    if output_artifact.exists() or output_report.exists():
        raise RuntimeError("Refusing to overwrite OOF target outputs")
    if output_artifact.parent.resolve() != output_report.parent.resolve():
        raise RuntimeError("OOF artifact and report must share one output directory")
    output_artifact.parent.mkdir(parents=True, exist_ok=True)

    records = []
    predictions = {}
    for architecture in ("morphology", "spatial"):
        for heldout_fold in (0, 1):
            path = Path(getattr(args, f"{architecture}_heldout_fold{heldout_fold}"))
            record, prediction = validate_teacher_run(path, architecture, heldout_fold)
            records.append(record)
            predictions[(architecture, heldout_fold)] = prediction

    assembled = {}
    reference_indices = reference_labels = reference_folds = None
    for architecture in ("morphology", "spatial"):
        parts = [predictions[(architecture, fold)] for fold in (0, 1)]
        indices = np.concatenate([part["indices"] for part in parts])
        labels = np.concatenate([part["labels"] for part in parts])
        logits = np.concatenate([part["logits"] for part in parts])
        folds = np.concatenate(
            [np.full(len(part["indices"]), fold, dtype=np.int64) for fold, part in enumerate(parts)]
        )
        order = np.argsort(indices)
        indices, labels, logits, folds = indices[order], labels[order], logits[order], folds[order]
        if len(indices) != 35001 or len(np.unique(indices)) != 35001:
            raise RuntimeError(f"{architecture} OOF coverage is not exactly 35,001 unique samples")
        if index_membership_sha256(indices) != EXPECTED_PARENT:
            raise RuntimeError(f"{architecture} OOF parent membership drifted")
        if reference_indices is None:
            reference_indices, reference_labels, reference_folds = indices, labels, folds
        elif not (
            np.array_equal(indices, reference_indices)
            and np.array_equal(labels, reference_labels)
            and np.array_equal(folds, reference_folds)
        ):
            raise RuntimeError("Morphology/spatial OOF index, label, or fold order differs")
        assembled[architecture] = logits.astype(np.float32, copy=False)

    morphology_correct = assembled["morphology"].argmax(1) == reference_labels
    spatial_correct = assembled["spatial"].argmax(1) == reference_labels
    gate = morphology_correct | spatial_correct
    morphology_probability = softmax_temperature(assembled["morphology"], TEMPERATURE)
    spatial_probability = softmax_temperature(assembled["spatial"], TEMPERATURE)
    denominator = (morphology_correct.astype(np.float64) + spatial_correct.astype(np.float64)).clip(1.0)
    target = (
        morphology_correct[:, None] * morphology_probability
        + spatial_correct[:, None] * spatial_probability
    ) / denominator[:, None]
    target[~gate] = np.eye(3, dtype=np.float64)[reference_labels[~gate]]
    target = target.astype(np.float32)
    if not np.allclose(target.sum(1), 1.0, rtol=1e-6, atol=1e-6):
        raise RuntimeError("OOF target probabilities are not normalized")
    routing_counts = {
        "both_correct": int((morphology_correct & spatial_correct).sum()),
        "morphology_only_correct": int((morphology_correct & ~spatial_correct).sum()),
        "spatial_only_correct": int((~morphology_correct & spatial_correct).sum()),
        "neither_correct": int((~morphology_correct & ~spatial_correct).sum()),
    }
    atomic_npz(
        output_artifact,
        indices=reference_indices.astype(np.int64),
        labels=reference_labels.astype(np.int64),
        morphology_logits=assembled["morphology"],
        spatial_logits=assembled["spatial"],
        source_fold=reference_folds.astype(np.int64),
        target_probabilities=target,
        gate=gate.astype(np.bool_),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "artifact_sha256": sha256_file(output_artifact),
        "samples": 35001,
        "train_membership_sha256": EXPECTED_PARENT,
        "development_manifest_sha256": EXPECTED_MANIFEST,
        "temperature": TEMPERATURE,
        "checkpoint_selection": "fixed final epoch only",
        "canonical_development_validation_samples_used": 0,
        "official_test_samples_used": 0,
        "routing_counts": routing_counts,
        "gated_samples": int(gate.sum()),
        "gated_fraction": float(gate.mean()),
        "index_to_fold_sha256": array_sha256(
            np.column_stack((reference_indices, reference_folds)).astype("<i8")
        ),
        "morphology_correct_mask_sha256": array_sha256(morphology_correct.astype(np.uint8)),
        "spatial_correct_mask_sha256": array_sha256(spatial_correct.astype(np.uint8)),
        "gate_sha256": array_sha256(gate.astype(np.uint8)),
        "target_probability_content_sha256": array_sha256(target.astype("<f4", copy=False)),
        "teachers": records,
    }
    temporary = output_report.with_name(f".{output_report.name}.building-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, output_report)
    print(json.dumps(report, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for architecture in ("morphology", "spatial"):
        for fold in (0, 1):
            value.add_argument(f"--{architecture}-heldout-fold{fold}", required=True)
    value.add_argument("--output-artifact", required=True)
    value.add_argument("--output-report", required=True)
    return value


def main() -> None:
    build(parser().parse_args())


if __name__ == "__main__":
    main()
