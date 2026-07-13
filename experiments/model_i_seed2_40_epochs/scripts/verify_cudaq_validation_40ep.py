#!/usr/bin/env python3
"""Run full validation through the literal notebook CUDA-Q kernel.

The script extracts the one notebook cell tagged ``cudaq-primary-kernel`` to
a real Python source file, imports its decorated ``@cudaq.kernel``, executes
all four heads for every development-validation image on CUDA-Q's NVIDIA
target, checks the resulting invariants against the PyTorch analytic export,
and replays the classifier head.  No official-test artifact is accepted or
opened by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SCHEMA = "deeplense.d4-orqb-40epoch-cudaq-validation.v1"
REFERENCE_SCHEMA = "deeplense.d4-orqb-40epoch-cudaq-reference.v1"
EXPECTED_KERNEL_CELL_SHA256 = (
    "d6ca4829bfaa2d18c80a460dd267f4dae8cd3a6da2e6713092c46c22d2cd1dc3"
)
EXPECTED_CUDAQ_VERSION_FRAGMENT = "0.12.0"
INVARIANT_ATOL = 2.0e-4
INVARIANT_RTOL = 2.0e-4
FP32_LOGIT_ATOL = 2.0e-3
FP32_LOGIT_RTOL = 2.0e-3
BF16_LOGIT_ATOL = 5.0e-2
BF16_PROBABILITY_ATOL = 5.0e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="nvidia")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--parity-samples", type=int, default=16)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def extract_kernel_cell(notebook_path: Path) -> tuple[str, int]:
    notebook = json.loads(notebook_path.read_text())
    matches = []
    for index, cell in enumerate(notebook.get("cells", [])):
        tags = cell.get("metadata", {}).get("tags", [])
        if "cudaq-primary-kernel" in tags:
            matches.append((index, "".join(cell.get("source", []))))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one cudaq-primary-kernel cell, found {len(matches)}"
        )
    cell_index, source = matches[0]
    actual_sha256 = sha256_bytes(source.encode("utf-8"))
    if actual_sha256 != EXPECTED_KERNEL_CELL_SHA256:
        raise RuntimeError(
            "Notebook CUDA-Q kernel cell drifted: "
            f"actual={actual_sha256} expected={EXPECTED_KERNEL_CELL_SHA256}"
        )
    if "@cudaq.kernel" not in source or "def d4_orqb_q2_head" not in source:
        raise RuntimeError("Tagged cell does not contain the required CUDA-Q kernel")
    if "class CudaQD4OrbitBackend" not in source:
        raise RuntimeError("Tagged cell does not contain its audited CUDA-Q backend")
    return source, cell_index


def import_kernel(source: str, destination: Path):
    destination.write_text(source)
    module_name = f"d4_orqb_notebook_cudaq_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(module_name, destination)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not construct a module for the notebook kernel")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    if not module.CUDAQ_AVAILABLE or module.d4_orqb_q2_head is None:
        raise RuntimeError("CUDA-Q was unavailable while importing the notebook cell")
    if (
        module.CUDAQ_HEADS != 4
        or module.CUDAQ_QUBITS != 8
        or module.CUDAQ_REUPLOADS != 2
        or module.CUDAQ_PARAMETERS_PER_LAYER != 11
        or module.CUDAQ_INVARIANTS_PER_HEAD != 12
    ):
        raise RuntimeError("Notebook CUDA-Q architecture constants drifted")
    return module


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(values)
    return (exponential / exponential.sum(axis=1, keepdims=True)).astype(np.float32)


def binary_auc(target: np.ndarray, scores: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = ranks[target].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def classification_metrics(
    labels: np.ndarray, logits: np.ndarray, class_names: list[str]
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    probabilities = softmax(logits)
    predictions = logits.argmax(axis=1)
    classes = len(class_names)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support != 0
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    auc = [
        binary_auc(labels == class_index, probabilities[:, class_index])
        for class_index in range(classes)
    ]
    return {
        "samples": int(len(labels)),
        "correct": int((predictions == labels).sum()),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auc_ovr": float(np.mean(auc)),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "auc_ovr": float(auc[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def numpy_fp32_head(invariants: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(invariants, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True, dtype=np.float32)
    centered = values - mean
    variance = np.square(centered).mean(axis=1, keepdims=True, dtype=np.float32)
    denominator = np.sqrt(
        variance + np.float32(weights["layer_norm_eps"]), dtype=np.float32
    )
    normalized = centered / denominator
    normalized = (
        normalized * weights["layer_norm_weight"][None]
        + weights["layer_norm_bias"][None]
    )
    hidden = (
        normalized @ weights["linear1_weight"].T
        + weights["linear1_bias"][None]
    ).astype(np.float32)
    hidden = (hidden / (np.float32(1.0) + np.exp(-hidden))).astype(np.float32)
    return (
        hidden @ weights["linear2_weight"].T + weights["linear2_bias"][None]
    ).astype(np.float32)


def optional_torch_bf16_head(
    invariants: np.ndarray,
    weights: dict[str, np.ndarray],
    batch_size: int = 256,
) -> tuple[np.ndarray | None, dict]:
    try:
        import torch
        from torch import nn
    except Exception as error:  # CUDA-Q image need not bundle PyTorch.
        return None, {
            "status": "unavailable",
            "reason": f"PyTorch import failed: {type(error).__name__}: {error}",
        }
    if not torch.cuda.is_available():
        return None, {
            "status": "unavailable",
            "reason": "PyTorch is installed but CUDA is unavailable",
            "torch": torch.__version__,
        }
    try:
        head = nn.Sequential(
            nn.LayerNorm(48, eps=float(weights["layer_norm_eps"])),
            nn.Linear(48, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(float(weights["dropout_probability"])),
            nn.Linear(32, 3),
        )
        with torch.no_grad():
            head[0].weight.copy_(torch.from_numpy(weights["layer_norm_weight"]))
            head[0].bias.copy_(torch.from_numpy(weights["layer_norm_bias"]))
            head[1].weight.copy_(torch.from_numpy(weights["linear1_weight"]))
            head[1].bias.copy_(torch.from_numpy(weights["linear1_bias"]))
            head[4].weight.copy_(torch.from_numpy(weights["linear2_weight"]))
            head[4].bias.copy_(torch.from_numpy(weights["linear2_bias"]))
        head = head.cuda().eval()
        output = np.empty((len(invariants), 3), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(invariants), batch_size):
                stop = min(start + batch_size, len(invariants))
                values = torch.from_numpy(invariants[start:stop]).cuda()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = head(values)
                output[start:stop] = logits.float().cpu().numpy()
        return output, {
            "status": "passed",
            "engine": "PyTorch CUDA autocast bfloat16",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    except Exception as error:
        return None, {
            "status": "unavailable",
            "reason": f"BF16 execution failed: {type(error).__name__}: {error}",
            "torch": getattr(torch, "__version__", None),
        }


def cudaq_runtime_report(cudaq, target_name: str) -> dict:
    version = None
    for candidate in (
        getattr(cudaq, "__version__", None),
        getattr(cudaq, "get_version", None),
    ):
        if callable(candidate):
            try:
                candidate = candidate()
            except Exception:
                continue
        if candidate:
            version = str(candidate)
            break
    if version is None:
        completed = subprocess.run(
            ["nvq++", "--version"], text=True, capture_output=True, check=False
        )
        version = (completed.stdout or completed.stderr).strip()
    if EXPECTED_CUDAQ_VERSION_FRAGMENT not in version:
        raise RuntimeError(f"Expected CUDA-Q 0.12.0, got {version}")
    target = cudaq.get_target()
    attributes = {}
    for name in (
        "name",
        "description",
        "simulator",
        "platform",
        "num_qpus",
        "is_remote",
        "is_emulated",
        "get_precision",
    ):
        value = getattr(target, name, None)
        if callable(value):
            try:
                value = value()
            except Exception as error:
                value = f"unavailable: {type(error).__name__}: {error}"
        if value is not None:
            attributes[name] = str(value)
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cudaq_version": version,
        "requested_target": target_name,
        "target": attributes,
        "nvidia_smi_returncode": smi.returncode,
        "nvidia_smi": [line for line in smi.stdout.splitlines() if line],
        "nvidia_smi_stderr": smi.stderr.strip(),
    }


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if args.target != "nvidia":
        raise ValueError("This verification is pinned to target='nvidia'")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    export_report = json.loads(args.export_report.read_text())
    if export_report.get("schema") != REFERENCE_SCHEMA:
        raise RuntimeError("Unexpected export report schema")
    if export_report.get("status") != "passed":
        raise RuntimeError("Reference export has not passed")
    if export_report.get("official_test_evaluated") is not False:
        raise RuntimeError("Reference export did not keep official test closed")
    if export_report.get("official_test_cache_opened") is not False:
        raise RuntimeError("Reference export opened the official test cache")
    reference_sha256 = sha256_file(args.reference)
    weights_sha256 = sha256_file(args.weights)
    if reference_sha256 != export_report.get("reference_sha256"):
        raise RuntimeError("Validation reference SHA-256 mismatch")
    if weights_sha256 != export_report.get("weights_sha256"):
        raise RuntimeError("Weights SHA-256 mismatch")

    with np.load(args.reference, allow_pickle=False) as fixture:
        if str(fixture["schema"].item()) != REFERENCE_SCHEMA:
            raise RuntimeError("Unexpected validation reference schema")
        if str(fixture["split"].item()) != "development_validation":
            raise RuntimeError("Only the development-validation split is permitted")
        angles = np.asarray(fixture["angles"], dtype=np.float32)
        expected_invariants = np.asarray(
            fixture["expected_invariants"], dtype=np.float32
        )
        expected_logits_bf16 = np.asarray(
            fixture["expected_logits_bf16"], dtype=np.float32
        )
        expected_logits_fp32 = np.asarray(
            fixture["expected_logits_fp32_head"], dtype=np.float32
        )
        expected_probabilities_bf16 = np.asarray(
            fixture["expected_probabilities_bf16"], dtype=np.float32
        )
        labels = np.asarray(fixture["labels"], dtype=np.int64)
        sample_indices = np.asarray(fixture["sample_indices"], dtype=np.int64)
        class_names = [str(value) for value in fixture["class_names"].tolist()]
    sample_count = len(labels)
    if sample_count != 17504:
        raise RuntimeError(f"Expected 17,504 validation samples, got {sample_count}")
    expected_shapes = {
        "angles": (sample_count, 4, 2, 8),
        "expected_invariants": (sample_count, 48),
        "expected_logits_bf16": (sample_count, 3),
        "expected_logits_fp32": (sample_count, 3),
        "expected_probabilities_bf16": (sample_count, 3),
        "sample_indices": (sample_count,),
    }
    actual_shapes = {
        "angles": angles.shape,
        "expected_invariants": expected_invariants.shape,
        "expected_logits_bf16": expected_logits_bf16.shape,
        "expected_logits_fp32": expected_logits_fp32.shape,
        "expected_probabilities_bf16": expected_probabilities_bf16.shape,
        "sample_indices": sample_indices.shape,
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"Validation fixture shape drift: actual={actual_shapes} expected={expected_shapes}"
        )
    if class_names != ["axion", "cdm", "no_sub"]:
        raise RuntimeError(f"Unexpected class order: {class_names}")
    if len(np.unique(sample_indices)) != sample_count:
        raise RuntimeError("Validation sample indices are not unique")
    for name, array in (
        ("angles", angles),
        ("expected_invariants", expected_invariants),
        ("expected_logits_bf16", expected_logits_bf16),
        ("expected_logits_fp32", expected_logits_fp32),
    ):
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite values in {name}")

    with np.load(args.weights, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != REFERENCE_SCHEMA:
            raise RuntimeError("Unexpected weight archive schema")
        circuit_parameters = np.asarray(
            archive["circuit_parameters"], dtype=np.float32
        )
        weight_names = (
            "layer_norm_weight",
            "layer_norm_bias",
            "layer_norm_eps",
            "linear1_weight",
            "linear1_bias",
            "linear2_weight",
            "linear2_bias",
            "dropout_probability",
        )
        head_weights = {
            name: np.asarray(archive[name], dtype=np.float32) for name in weight_names
        }
    if circuit_parameters.shape != (4, 2, 11):
        raise RuntimeError("Circuit parameter shape drifted")

    source, notebook_cell_index = extract_kernel_cell(args.notebook)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output_dir.parent / (
        f".{args.output_dir.name}.building-{socket.gethostname()}-{os.getpid()}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    kernel_path = staging / "kernel_source.py"
    module = import_kernel(source, kernel_path)
    backend = module.CudaQD4OrbitBackend(args.target)
    backend.initialize()
    runtime = cudaq_runtime_report(module.cudaq, args.target)

    parity_count = min(args.parity_samples, sample_count)
    parity_started = time.monotonic()
    parity_invariants = backend(
        angles[:parity_count], circuit_parameters
    ).astype(np.float32, copy=False)
    parity_elapsed = time.monotonic() - parity_started
    parity_close = np.isclose(
        parity_invariants,
        expected_invariants[:parity_count],
        rtol=INVARIANT_RTOL,
        atol=INVARIANT_ATOL,
    )
    parity = {
        "status": "passed" if bool(parity_close.all()) else "failed",
        "samples": parity_count,
        "values": int(parity_close.size),
        "mismatches": int(np.count_nonzero(~parity_close)),
        "max_abs_error": float(
            np.max(np.abs(parity_invariants - expected_invariants[:parity_count]))
        ),
        "atol": INVARIANT_ATOL,
        "rtol": INVARIANT_RTOL,
        "elapsed_seconds": parity_elapsed,
        "head_circuits_per_second": 4 * parity_count / parity_elapsed,
    }
    if parity["mismatches"]:
        raise RuntimeError(f"Literal notebook kernel parity failed: {parity}")
    atomic_json(staging / "parity.json", parity)

    actual_invariants = np.empty_like(expected_invariants)
    actual_invariants[:parity_count] = parity_invariants
    full_started = time.monotonic()
    for start in range(parity_count, sample_count, args.chunk_size):
        stop = min(start + args.chunk_size, sample_count)
        actual_invariants[start:stop] = backend(
            angles[start:stop], circuit_parameters
        )
        if stop == sample_count or stop % 512 < args.chunk_size:
            elapsed = time.monotonic() - full_started
            progress = {
                "completed_samples": stop,
                "total_samples": sample_count,
                "elapsed_seconds": elapsed,
                "images_per_second": (stop - parity_count) / max(elapsed, 1e-9),
            }
            atomic_json(staging / "progress.json", progress)
            print(
                f"CUDAQ_PROGRESS {stop}/{sample_count} "
                f"images_per_second={progress['images_per_second']:.3f}",
                flush=True,
            )
    full_elapsed = time.monotonic() - full_started + parity_elapsed
    invariant_close = np.isclose(
        actual_invariants,
        expected_invariants,
        rtol=INVARIANT_RTOL,
        atol=INVARIANT_ATOL,
    )
    invariant_mismatches = int(np.count_nonzero(~invariant_close))
    invariant_max_abs_error = float(
        np.max(np.abs(actual_invariants - expected_invariants))
    )
    if invariant_mismatches:
        raise RuntimeError(
            "Full CUDA-Q invariant parity failed: "
            f"mismatches={invariant_mismatches} max_error={invariant_max_abs_error}"
        )

    logits_fp32 = numpy_fp32_head(actual_invariants, head_weights)
    fp32_logit_close = np.isclose(
        logits_fp32,
        expected_logits_fp32,
        rtol=FP32_LOGIT_RTOL,
        atol=FP32_LOGIT_ATOL,
    )
    fp32_prediction_mismatches = int(
        np.count_nonzero(logits_fp32.argmax(1) != expected_logits_fp32.argmax(1))
    )
    fp32_report = {
        "engine": "NumPy float32 LayerNorm/MLP",
        "status": "passed",
        "logit_atol": FP32_LOGIT_ATOL,
        "logit_rtol": FP32_LOGIT_RTOL,
        "logit_mismatches": int(np.count_nonzero(~fp32_logit_close)),
        "logit_max_abs_error": float(
            np.max(np.abs(logits_fp32 - expected_logits_fp32))
        ),
        "prediction_mismatches": fp32_prediction_mismatches,
        "metrics": classification_metrics(labels, logits_fp32, class_names),
    }
    if fp32_report["logit_mismatches"] or fp32_prediction_mismatches:
        raise RuntimeError(f"FP32 classifier parity failed: {fp32_report}")

    logits_bf16, bf16_runtime = optional_torch_bf16_head(
        actual_invariants, head_weights
    )
    if logits_bf16 is not None:
        probabilities_bf16 = softmax(logits_bf16)
        bf16_prediction_mismatches = int(
            np.count_nonzero(
                logits_bf16.argmax(1) != expected_logits_bf16.argmax(1)
            )
        )
        bf16_report = {
            **bf16_runtime,
            "logit_atol": BF16_LOGIT_ATOL,
            "logit_max_abs_error": float(
                np.max(np.abs(logits_bf16 - expected_logits_bf16))
            ),
            "probability_atol": BF16_PROBABILITY_ATOL,
            "probability_max_abs_error": float(
                np.max(np.abs(probabilities_bf16 - expected_probabilities_bf16))
            ),
            "prediction_mismatches": bf16_prediction_mismatches,
            "metrics": classification_metrics(labels, logits_bf16, class_names),
        }
        if (
            bf16_report["logit_max_abs_error"] > BF16_LOGIT_ATOL
            or bf16_report["probability_max_abs_error"] > BF16_PROBABILITY_ATOL
            or bf16_prediction_mismatches
        ):
            raise RuntimeError(f"BF16 classifier parity failed: {bf16_report}")
        selected_mode = "cudaq_invariants_plus_torch_cuda_bfloat16_head"
        selected_logits = logits_bf16
        selected_probabilities = probabilities_bf16
        selected_metrics = bf16_report["metrics"]
    else:
        bf16_report = bf16_runtime
        selected_mode = "cudaq_invariants_plus_numpy_float32_head"
        selected_logits = logits_fp32
        selected_probabilities = softmax(logits_fp32)
        selected_metrics = fp32_report["metrics"]

    invariants_path = staging / "cudaq_invariants.npy"
    predictions_path = staging / "predictions.npz"
    np.save(invariants_path, actual_invariants)
    np.savez(
        predictions_path,
        schema=np.asarray(SCHEMA),
        split=np.asarray("development_validation"),
        execution_mode=np.asarray(selected_mode),
        logits=selected_logits,
        probabilities=selected_probabilities,
        predictions=selected_logits.argmax(1).astype(np.int64),
        labels=labels,
        sample_indices=sample_indices,
        logits_fp32_head=logits_fp32,
        logits_bf16_head=(
            logits_bf16
            if logits_bf16 is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        class_names=np.asarray(class_names),
    )
    result = {
        "schema": SCHEMA,
        "status": "passed",
        "split": "development_validation",
        "official_test_evaluated": False,
        "official_test_artifacts_opened": 0,
        "samples": sample_count,
        "head_circuits": sample_count * 4,
        "engine": "literal notebook @cudaq.kernel via cudaq.get_state",
        "selected_classifier_mode": selected_mode,
        "metrics": selected_metrics,
        "invariant_values_compared": int(invariant_close.size),
        "invariant_mismatches": invariant_mismatches,
        "invariant_max_abs_error": invariant_max_abs_error,
        "invariant_atol": INVARIANT_ATOL,
        "invariant_rtol": INVARIANT_RTOL,
        "parity": parity,
        "fp32_head": fp32_report,
        "bf16_head": bf16_report,
        "notebook": str(args.notebook),
        "notebook_sha256": sha256_file(args.notebook),
        "notebook_kernel_cell_index": notebook_cell_index,
        "notebook_kernel_cell_sha256": EXPECTED_KERNEL_CELL_SHA256,
        "kernel_source_sha256": sha256_file(kernel_path),
        "runner": str(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "reference": str(args.reference),
        "reference_sha256": reference_sha256,
        "weights": str(args.weights),
        "weights_sha256": weights_sha256,
        "export_report": str(args.export_report),
        "export_report_sha256": sha256_file(args.export_report),
        "invariants": str(args.output_dir / invariants_path.name),
        "invariants_sha256": sha256_file(invariants_path),
        "predictions": str(args.output_dir / predictions_path.name),
        "predictions_sha256": sha256_file(predictions_path),
        "runtime": {
            **runtime,
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "elapsed_seconds": time.monotonic() - started,
            "quantum_elapsed_seconds": full_elapsed,
            "images_per_second": sample_count / full_elapsed,
            "head_circuits_per_second": sample_count * 4 / full_elapsed,
            "chunk_size": args.chunk_size,
        },
    }
    atomic_json(staging / "result.json", result)
    os.replace(staging, args.output_dir)
    print("CUDAQ_VALIDATION_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
