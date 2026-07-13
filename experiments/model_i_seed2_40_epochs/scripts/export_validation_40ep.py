#!/usr/bin/env python3
"""Export the 40-epoch q2 model's validation-only CUDA-Q fixture.

This stage runs in the pinned PyTorch image.  It loads the exact source
archive and best checkpoint produced by the 40-epoch job, replays only the
saved development-validation indices, and exports angles, analytic reference
invariants, logits, labels, and the circuit/classifier tensors required by the
subsequent CUDA-Q verification stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from d4_orqb.data import CachedNPYDataset, make_loader
from d4_orqb.metrics import classification_metrics
from d4_orqb.model import D4OrbitClassifier


SOURCE_COMMIT = "f10aad3a92b51dd3af0cc7ecf89288cb53001269"
SOURCE_ARCHIVE_SHA256 = (
    "a687cd1a6c52cde6a8fdfb61d6df195002e40df8a18f76082d9acc0652540dc6"
)
SCHEMA = "deeplense.d4-orqb-40epoch-cudaq-reference.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--development-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_model(config: dict, class_names: list[str]) -> D4OrbitClassifier:
    keys = (
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
        "cross_scale_reupload",
    )
    kwargs = {key: config[key] for key in keys}
    return D4OrbitClassifier(num_classes=len(class_names), **kwargs)


def audit_closed_test_contract(
    config: dict,
    data_report: dict,
    summary: dict,
    provenance: dict,
) -> None:
    violations = []
    if config.get("evaluate_test") is not False:
        violations.append("config.evaluate_test")
    if data_report.get("official_test_cache_opened") is not False:
        violations.append("data_report.official_test_cache_opened")
    if "test" in data_report:
        violations.append("data_report.test")
    if summary.get("official_test_evaluated") is not False:
        violations.append("summary.official_test_evaluated")
    if "test" in summary:
        violations.append("summary.test")
    if provenance.get("official_test_evaluated") is not False:
        violations.append("provenance.official_test_evaluated")
    if violations:
        raise RuntimeError(f"Official-test closure violated: {violations}")


def extract_head_weights(model: D4OrbitClassifier) -> dict[str, np.ndarray]:
    # The selected tiny q2 model has a validation-time deterministic head:
    # LayerNorm(48), Linear(48,32), SiLU, Dropout(disabled in eval), Linear(32,3).
    head = model.head
    if not (
        isinstance(head, nn.Sequential)
        and len(head) == 5
        and isinstance(head[0], nn.LayerNorm)
        and isinstance(head[1], nn.Linear)
        and isinstance(head[2], nn.SiLU)
        and isinstance(head[3], nn.Dropout)
        and isinstance(head[4], nn.Linear)
    ):
        raise RuntimeError(f"Unexpected classifier head: {head}")
    if head.training:
        raise RuntimeError("Classifier head must be in eval mode before export")
    arrays = {
        "layer_norm_weight": head[0].weight.detach().float().cpu().numpy(),
        "layer_norm_bias": head[0].bias.detach().float().cpu().numpy(),
        "layer_norm_eps": np.asarray(head[0].eps, dtype=np.float64),
        "linear1_weight": head[1].weight.detach().float().cpu().numpy(),
        "linear1_bias": head[1].bias.detach().float().cpu().numpy(),
        "linear2_weight": head[4].weight.detach().float().cpu().numpy(),
        "linear2_bias": head[4].bias.detach().float().cpu().numpy(),
        "dropout_probability": np.asarray(head[3].p, dtype=np.float64),
    }
    expected_shapes = {
        "layer_norm_weight": (48,),
        "layer_norm_bias": (48,),
        "linear1_weight": (32, 48),
        "linear1_bias": (32,),
        "linear2_weight": (3, 32),
        "linear2_bias": (3,),
    }
    actual_shapes = {key: tuple(arrays[key].shape) for key in expected_shapes}
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"Classifier shape drift: actual={actual_shapes} expected={expected_shapes}"
        )
    return arrays


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exporter requires exactly one CUDA GPU")
    if sha256_file(args.source_archive) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Pinned source archive SHA-256 mismatch")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    required = (
        "best.pt",
        "best_validation_predictions.npz",
        "config.json",
        "data_report.json",
        "parameter_report.json",
        "run_provenance.json",
        "split_indices.npz",
        "summary.json",
    )
    missing = [name for name in required if not (args.training_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Training output is incomplete: {missing}")

    config = json.loads((args.training_root / "config.json").read_text())
    data_report = json.loads((args.training_root / "data_report.json").read_text())
    parameter_report = json.loads(
        (args.training_root / "parameter_report.json").read_text()
    )
    summary = json.loads((args.training_root / "summary.json").read_text())
    provenance = json.loads((args.training_root / "run_provenance.json").read_text())
    audit_closed_test_contract(config, data_report, summary, provenance)

    if provenance.get("schema") != "deeplense.d4-orqb-40epoch-training.v1":
        raise RuntimeError("Unexpected training provenance schema")
    if provenance.get("status") != "passed":
        raise RuntimeError("Training provenance has not passed")
    if provenance.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("Training source commit mismatch")
    if provenance.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Training source archive mismatch")
    if provenance.get("epochs_completed") != 40 or config.get("epochs") != 40:
        raise RuntimeError("Expected a completed 40-epoch training run")
    if config.get("core") != "quantum" or config.get("heads") != 4:
        raise RuntimeError("Expected the four-head quantum q2 model")
    if config.get("reuploads") != 2 or config.get("quantum_encoding") != "angle":
        raise RuntimeError("Expected two-reupload angle encoding")
    if config.get("observable_readout") != "pair":
        raise RuntimeError("Expected the 12-invariant pair readout")
    if parameter_report.get("total") != 245221:
        raise RuntimeError("Total parameter count drifted")
    if parameter_report.get("quantum") != 88:
        raise RuntimeError("Quantum parameter count drifted")

    checkpoint_path = args.training_root / "best.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if provenance.get("artifact_sha256", {}).get("best.pt") != checkpoint_sha256:
        raise RuntimeError("best.pt does not match training provenance")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != int(provenance.get("best_epoch", -2)):
        raise RuntimeError("Checkpoint epoch does not match training provenance")

    metadata = json.loads((args.development_cache / "metadata.json").read_text())
    class_names = list(metadata["classes"])
    if class_names != list(data_report["class_names"]):
        raise RuntimeError("Development class order drifted")
    development_hashes = {
        name: sha256_file(args.development_cache / name)
        for name in ("images.npy", "labels.npy", "manifest.csv", "metadata.json")
    }
    if development_hashes["manifest.csv"] != data_report.get(
        "development_manifest_sha256"
    ):
        raise RuntimeError("Development manifest drifted")

    with np.load(args.training_root / "split_indices.npz") as split:
        validation_indices = np.asarray(split["val"], dtype=np.int64)
    if len(validation_indices) != int(data_report["validation_size"]):
        raise RuntimeError("Validation split size drifted")
    full_labels = np.load(args.development_cache / "labels.npy", mmap_mode="r")
    if (
        validation_indices.ndim != 1
        or len(np.unique(validation_indices)) != len(validation_indices)
        or validation_indices.min() < 0
        or validation_indices.max() >= len(full_labels)
    ):
        raise RuntimeError("Invalid saved validation indices")

    device = torch.device("cuda")
    model = build_model(config, class_names)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device, memory_format=torch.channels_last).eval()
    if tuple(model.core.params.shape) != (4, 2, 11):
        raise RuntimeError("Circuit parameter shape drifted")
    if sum(parameter.numel() for parameter in model.parameters()) != 245221:
        raise RuntimeError("Loaded model parameter count drifted")
    head_weights = extract_head_weights(model)

    loader = make_loader(
        CachedNPYDataset(args.development_cache, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        seed=int(config["split_seed"]),
    )
    sample_count = len(validation_indices)
    angles_all = np.empty((sample_count, 4, 2, 8), dtype=np.float32)
    invariants_all = np.empty((sample_count, 48), dtype=np.float32)
    logits_bf16_all = np.empty((sample_count, len(class_names)), dtype=np.float32)
    logits_fp32_head_all = np.empty_like(logits_bf16_all)
    labels_all = np.empty(sample_count, dtype=np.int64)
    indices_all = np.empty(sample_count, dtype=np.int64)

    offset = 0
    with torch.no_grad():
        for batch_number, (images, labels, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_bf16, auxiliary = model(images, return_aux=True)
            invariants = auxiliary["invariants"].float()
            with torch.autocast(device_type="cuda", enabled=False):
                logits_fp32_head = model.head(invariants)
            count = len(labels)
            destination = slice(offset, offset + count)
            angles_all[destination] = auxiliary["angles"].float().cpu().numpy()
            invariants_all[destination] = invariants.cpu().numpy()
            logits_bf16_all[destination] = logits_bf16.float().cpu().numpy()
            logits_fp32_head_all[destination] = logits_fp32_head.float().cpu().numpy()
            labels_all[destination] = labels.numpy()
            indices_all[destination] = indices.numpy()
            offset += count
            if batch_number == 1 or offset == sample_count or offset % 2048 < count:
                print(f"EXPORT_PROGRESS {offset}/{sample_count}", flush=True)
    if offset != sample_count:
        raise RuntimeError(f"Exported {offset} of {sample_count} samples")
    if not np.array_equal(indices_all, validation_indices):
        raise RuntimeError("Validation loader order drifted")
    if not np.array_equal(labels_all, np.asarray(full_labels)[validation_indices]):
        raise RuntimeError("Validation labels drifted")
    for name, array in (
        ("angles", angles_all),
        ("expected_invariants", invariants_all),
        ("expected_logits_bf16", logits_bf16_all),
        ("expected_logits_fp32_head", logits_fp32_head_all),
    ):
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite values in {name}")

    with np.load(args.training_root / "best_validation_predictions.npz") as stored:
        stored_labels = np.asarray(stored["labels"], dtype=np.int64)
        stored_logits = np.asarray(stored["logits"], dtype=np.float32)
        stored_indices = np.asarray(stored["indices"], dtype=np.int64)
    if not np.array_equal(stored_labels, labels_all):
        raise RuntimeError("Saved best-validation labels did not replay")
    if not np.array_equal(stored_indices, indices_all):
        raise RuntimeError("Saved best-validation indices did not replay")
    stored_prediction_mismatches = int(
        np.count_nonzero(stored_logits.argmax(1) != logits_bf16_all.argmax(1))
    )
    if stored_prediction_mismatches:
        raise RuntimeError(
            f"Saved best-validation predictions did not replay: {stored_prediction_mismatches}"
        )

    probabilities_bf16 = torch.softmax(
        torch.from_numpy(logits_bf16_all), dim=1
    ).numpy()
    probabilities_fp32_head = torch.softmax(
        torch.from_numpy(logits_fp32_head_all), dim=1
    ).numpy()
    bf16_metrics = classification_metrics(labels_all, logits_bf16_all, class_names)
    fp32_head_metrics = classification_metrics(
        labels_all, logits_fp32_head_all, class_names
    )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output_dir.parent / (
        f".{args.output_dir.name}.building-{socket.gethostname()}-{os.getpid()}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    reference_path = staging / "validation_reference.npz"
    weights_path = staging / "q2_weights.npz"
    np.savez(
        reference_path,
        schema=np.asarray(SCHEMA),
        split=np.asarray("development_validation"),
        angles=angles_all,
        expected_invariants=invariants_all,
        expected_logits_bf16=logits_bf16_all,
        expected_logits_fp32_head=logits_fp32_head_all,
        expected_probabilities_bf16=probabilities_bf16.astype(np.float32),
        expected_probabilities_fp32_head=probabilities_fp32_head.astype(np.float32),
        labels=labels_all,
        sample_indices=indices_all,
        class_names=np.asarray(class_names),
    )
    weight_payload = {
        "schema": np.asarray(SCHEMA),
        "circuit_parameters": model.core.params.detach().float().cpu().numpy(),
        "class_names": np.asarray(class_names),
        **head_weights,
    }
    np.savez(weights_path, **weight_payload)

    report = {
        "schema": SCHEMA,
        "status": "passed",
        "split": "development_validation",
        "official_test_evaluated": False,
        "official_test_cache_opened": False,
        "source_commit": SOURCE_COMMIT,
        "source_archive": str(args.source_archive),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "training_root": str(args.training_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": checkpoint_sha256,
        "training_provenance_sha256": sha256_file(
            args.training_root / "run_provenance.json"
        ),
        "config_sha256": sha256_file(args.training_root / "config.json"),
        "split_indices_sha256": sha256_file(
            args.training_root / "split_indices.npz"
        ),
        "development_cache_sha256": development_hashes,
        "samples": sample_count,
        "angle_shape": list(angles_all.shape),
        "invariant_shape": list(invariants_all.shape),
        "class_names": class_names,
        "class_counts": np.bincount(
            labels_all, minlength=len(class_names)
        ).tolist(),
        "parameters": {"total": 245221, "quantum": 88},
        "circuit_parameter_shape": list(model.core.params.shape),
        "amp": "CUDA bfloat16 encoder/head; float32 analytic PyTorch quantum core",
        "stored_best_prediction_mismatches": stored_prediction_mismatches,
        "stored_best_logit_max_abs_error": float(
            np.max(np.abs(stored_logits - logits_bf16_all))
        ),
        "validation_bf16": bf16_metrics,
        "validation_fp32_head": fp32_head_metrics,
        "reference": str(args.output_dir / reference_path.name),
        "reference_sha256": sha256_file(reference_path),
        "weights": str(args.output_dir / weights_path.name),
        "weights_sha256": sha256_file(weights_path),
        "runtime": {
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    (staging / "export_report.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging, args.output_dir)
    print("EXPORT_COMPLETE " + json.dumps(jsonable(report), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
