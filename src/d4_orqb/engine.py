"""Training, validation, checkpointing, metrics, and symmetry audits."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import LoaderBundle
from .encoder import ModelIVPhysicsSummary, d4_transform
from .model import D4OrbitClassifier, build_model
from .quantum import D4_ELEMENTS, right_regular_permutation


BACKBONE_PREFIXES = (
    "physics.",
    "physics_summary.",
    "physics_summary_norm.",
    "physics_summary_head.",
    "encoder.",
    "orbit_projection.",
)


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    core: Literal["quantum", "classical"]
    include_context: bool
    epochs: int
    patience: int
    seed: int
    encoder_learning_rate: float
    learning_rate: float
    core_learning_rate: float


def pretrain_spec(config: Config) -> StageSpec:
    return StageSpec(
        name="pretrain_context",
        core="classical",
        include_context=True,
        epochs=config.pretrain_epochs,
        patience=config.pretrain_patience,
        seed=config.pretrain_seed,
        encoder_learning_rate=config.pretrain_learning_rate,
        learning_rate=config.pretrain_learning_rate,
        core_learning_rate=config.pretrain_core_learning_rate,
    )


def quantum_spec(config: Config) -> StageSpec:
    return StageSpec(
        name=f"quantum_seed{config.quantum_seed}_{config.quantum_epochs}ep",
        core="quantum",
        include_context=False,
        epochs=config.quantum_epochs,
        patience=config.quantum_patience,
        seed=config.quantum_seed,
        encoder_learning_rate=config.encoder_learning_rate,
        learning_rate=config.learning_rate,
        core_learning_rate=config.core_learning_rate,
    )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, classes: int
) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    distinct = np.r_[
        np.flatnonzero(np.diff(ordered_scores)), len(ordered_scores) - 1
    ]
    true_positive = np.cumsum(ordered_labels)[distinct]
    false_positive = 1 + distinct - true_positive
    true_positive_rate = np.r_[0.0, true_positive / positives]
    false_positive_rate = np.r_[0.0, false_positive / negatives]
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(true_positive_rate, false_positive_rate))


def _binary_roc_curve(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return false/true-positive rates using the same ties as ``binary_auc``."""

    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if labels.size == 0 or positives == 0 or negatives == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    distinct = np.r_[
        np.flatnonzero(np.diff(ordered_scores)), len(ordered_scores) - 1
    ]
    true_positive = np.cumsum(ordered_labels)[distinct]
    false_positive = 1 + distinct - true_positive
    true_positive_rate = np.r_[0.0, true_positive / positives]
    false_positive_rate = np.r_[0.0, false_positive / negatives]
    return false_positive_rate, true_positive_rate


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    calibration_error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            calibration_error += mask.mean() * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(calibration_error)


def classification_metrics(
    labels: np.ndarray, logits: np.ndarray, class_names: List[str]
) -> Dict:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    classes = len(class_names)
    matrix = confusion_matrix(labels, predictions, classes)
    per_class = {}
    f1_values, recalls, auc_values = [], [], []
    for label, name in enumerate(class_names):
        true_positive = int(matrix[label, label])
        false_positive = int(matrix[:, label].sum() - true_positive)
        false_negative = int(matrix[label, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        auc = binary_auc(labels == label, probabilities[:, label])
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_ovr": auc,
            "support": int(matrix[label].sum()),
        }
        f1_values.append(f1)
        recalls.append(recall)
        auc_values.append(auc)

    clipped = np.clip(
        probabilities[np.arange(len(labels)), labels], 1e-12, 1.0
    )
    one_hot = np.eye(classes, dtype=np.float64)[labels]
    result = {
        "samples": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_auc_ovr": float(np.nanmean(auc_values)),
        "nll": float(-np.log(clipped).mean()),
        "brier": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "ece_15": expected_calibration_error(probabilities, labels, bins=15),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
    return result


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _save_validation_roc_curve(
    path: Path,
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: List[str],
    metrics: Dict,
) -> None:
    """Save development-validation one-vs-rest ROC curves without sklearn."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    try:
        for class_index, class_name in enumerate(class_names):
            false_positive_rate, true_positive_rate = _binary_roc_curve(
                labels == class_index, probabilities[:, class_index]
            )
            if false_positive_rate.size == 0:
                continue
            auc = metrics["per_class"][class_name]["auc_ovr"]
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                linewidth=2,
                label=f"{class_name} (AUC = {auc:.4f})",
            )
        axis.plot(
            (0.0, 1.0),
            (0.0, 1.0),
            color="black",
            linestyle="--",
            linewidth=1,
            label="Chance",
        )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="False positive rate",
            ylabel="True positive rate",
            title="Development-validation one-vs-rest ROC",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
        figure.tight_layout()
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        figure.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, path)
    finally:
        plt.close(figure)


def _write_validation_artifacts(
    output_dir: Path,
    stage: StageSpec,
    best_epoch: int,
    metrics: Dict,
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: List[str],
) -> None:
    """Write concise, explicitly development-only final validation results."""

    _save_validation_roc_curve(
        output_dir / "validation_roc_curve.png",
        labels,
        logits,
        class_names,
        metrics,
    )
    lines = [
        "# Development-validation metrics",
        "",
        f"- Stage: `{stage.name}`",
        f"- Selected epoch: {best_epoch}",
        f"- Validation samples: {metrics['samples']}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Macro one-vs-rest AUC: {metrics['macro_auc_ovr']:.6f}",
        "- Official test evaluated: **No.** The official test set was not opened or evaluated.",
        "",
        "## Per-class one-vs-rest AUC",
        "",
        "| Class | AUC |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {class_name} | "
        f"{metrics['per_class'][class_name]['auc_ovr']:.6f} |"
        for class_name in class_names
    )
    lines.extend(
        (
            "",
            "See `validation_roc_curve.png` for the corresponding ROC curves.",
            "",
        )
    )
    _atomic_text(output_dir / "validation_metrics.md", "\n".join(lines))


def load_backbone_checkpoint(
    model: D4OrbitClassifier, checkpoint_path: str | Path
) -> Dict:
    """Load only the fixed preprocessing, encoder, and orbit projection."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(source, dict):
        raise RuntimeError(f"Invalid checkpoint state in {checkpoint_path}")

    target = model.state_dict()
    expected = {
        key
        for key in target
        if any(key.startswith(prefix) for prefix in BACKBONE_PREFIXES)
    }
    selected = {
        key: value
        for key, value in source.items()
        if key in expected and tuple(value.shape) == tuple(target[key].shape)
    }
    missing = sorted(expected.difference(selected))
    if missing:
        raise RuntimeError(
            "Backbone checkpoint is incomplete; missing compatible tensors: "
            f"{missing[:8]}"
        )
    target.update(selected)
    model.load_state_dict(target, strict=True)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "loaded_prefixes": list(BACKBONE_PREFIXES),
        "loaded_tensors": len(selected),
        "source_epoch": checkpoint.get("epoch"),
        "quantum_core_initialized_fresh": True,
        "classifier_initialized_fresh": True,
    }


def _backbone_modules(model: D4OrbitClassifier) -> Tuple[torch.nn.Module, ...]:
    """Return every module covered by the classical-backbone checkpoint."""

    modules = [model.physics, model.encoder, model.orbit_projection]
    for optional in (
        model.physics_summary,
        model.physics_summary_norm,
        model.physics_summary_head,
    ):
        if optional is not None:
            modules.append(optional)
    return tuple(modules)


def freeze_backbone(model: D4OrbitClassifier) -> Dict[str, int | bool]:
    """Freeze checkpoint-loaded parameters and stateful module behavior."""

    frozen_parameters = 0
    frozen_tensors = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in BACKBONE_PREFIXES):
            parameter.requires_grad_(False)
            frozen_parameters += parameter.numel()
            frozen_tensors += 1
    for module in _backbone_modules(model):
        module.eval()
    if frozen_parameters == 0:
        raise RuntimeError("The selected backbone freeze matched no parameters")
    return {
        "backbone_frozen": True,
        "frozen_parameter_tensors": frozen_tensors,
        "frozen_parameters": frozen_parameters,
        "normalization_buffers_frozen": True,
    }


def keep_frozen_backbone_in_eval(model: D4OrbitClassifier) -> None:
    """Undo the recursive mode change from ``model.train()`` for the backbone."""

    for module in _backbone_modules(model):
        module.eval()


def optimizer_parameter_groups(
    model: D4OrbitClassifier,
) -> Tuple[
    List[torch.nn.Parameter],
    List[torch.nn.Parameter],
    List[torch.nn.Parameter],
]:
    encoder_parameters = [
        parameter
        for module in (model.physics, model.encoder)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    head_modules = [model.orbit_projection, model.head]
    if model.physics_summary_head is not None:
        head_modules.append(model.physics_summary_head)
    if model.context_projection is not None:
        head_modules.append(model.context_projection)
    head_parameters = [
        parameter
        for module in head_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    core_parameters = [
        parameter
        for parameter in model.core.parameters()
        if parameter.requires_grad
    ]
    groups = (encoder_parameters, head_parameters, core_parameters)
    grouped_ids = [id(parameter) for group in groups for parameter in group]
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("A parameter appears in multiple optimizer groups")
    if set(grouped_ids) != trainable_ids:
        raise RuntimeError("Optimizer groups do not cover every trainable parameter")
    return groups


@torch.no_grad()
def evaluate(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_logits, all_indices = [], [], []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        all_labels.append(labels.numpy())
        all_logits.append(logits.float().cpu().numpy())
        all_indices.append(indices.numpy())
    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    indices = np.concatenate(all_indices)
    return (
        classification_metrics(labels, logits, class_names),
        labels,
        logits,
        indices,
    )


@torch.no_grad()
def symmetry_audit(
    model: D4OrbitClassifier,
    loader,
    device: torch.device,
    sample_limit: int = 16,
) -> Dict:
    model.eval()
    images = next(iter(loader))[0][:sample_limit]
    images = images.to(device).contiguous(memory_format=torch.channels_last)
    base_logits, base_auxiliary = model(images, return_aux=True)
    audit: Dict[str, Dict] = {}
    all_logit_differences = []
    for element in D4_ELEMENTS:
        logits, auxiliary = model(
            d4_transform(images, *element), return_aux=True
        )
        permutation = right_regular_permutation(element).to(device)
        expected_angles = base_auxiliary["angles"].index_select(
            -1, permutation
        )
        angle_difference = (
            auxiliary["angles"] - expected_angles
        ).abs().float()
        logit_difference = (logits - base_logits).abs().float().reshape(-1)
        all_logit_differences.append(logit_difference)
        record = {
            "angle_regular_max": float(angle_difference.max()),
            "logit_invariant_max": float(logit_difference.max()),
            "logit_invariant_mean": float(logit_difference.mean()),
        }
        if base_auxiliary["equivariant"] is not None:
            for name in ("z", "x"):
                expected = base_auxiliary["equivariant"][name].index_select(
                    -1, permutation
                )
                difference = (
                    auxiliary["equivariant"][name] - expected
                ).abs().float()
                record[f"circuit_{name}_regular_max"] = float(
                    difference.max()
                )
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


def train(
    config: Config,
    loaders: LoaderBundle,
    stage: StageSpec,
    output_dir: str | Path,
    device: torch.device,
    backbone_checkpoint: str | Path | None = None,
) -> Path:
    """Train one fixed stage and return its selected checkpoint path."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing stage output: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    seed_everything(stage.seed, config.deterministic)
    model = build_model(
        config, core=stage.core, include_context=stage.include_context
    ).to(device=device, memory_format=torch.channels_last)
    expected_parameters = 272_805 if stage.core == "classical" else 245_221
    if config.dataset_id == "model_iv":
        expected_parameters += ModelIVPhysicsSummary.output_dim * 3 + 3
    actual_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"Selected model parameter drift: {actual_parameters} != "
            f"{expected_parameters}"
        )

    initialization = {"mode": "fresh"}
    if backbone_checkpoint:
        if stage.core != "quantum":
            raise ValueError("Backbone initialization is only used by quantum stage")
        initialization = load_backbone_checkpoint(model, backbone_checkpoint)
    if config.freeze_backbone_during_quantum and stage.core == "quantum":
        if not backbone_checkpoint:
            raise ValueError(
                "A checkpoint-initialized quantum stage is required to freeze "
                "the classical backbone"
            )
        initialization.update(freeze_backbone(model))

    class_weights = None

    stage_config = {
        **config.to_dict(),
        "stage_spec": {
            key: getattr(stage, key)
            for key in stage.__dataclass_fields__
        },
        "output_dir": str(output_dir.resolve()),
        "effective_class_weights": (
            class_weights.detach().cpu().tolist()
            if class_weights is not None
            else None
        ),
    }
    _atomic_json(output_dir / "config.json", stage_config)
    _atomic_json(output_dir / "initialization.json", initialization)
    _atomic_json(output_dir / "parameters.json", model.parameter_report())

    encoder_parameters, head_parameters, core_parameters = (
        optimizer_parameter_groups(model)
    )
    optimizer = torch.optim.AdamW(
        (
            {
                "params": encoder_parameters,
                "lr": stage.encoder_learning_rate,
            },
            {"params": head_parameters, "lr": stage.learning_rate},
            {"params": core_parameters, "lr": stage.core_learning_rate},
        ),
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, stage.epochs * len(loaders.train))
    warmup_steps = max(
        1, min(3 * len(loaders.train), int(0.10 * total_steps))
    )

    def learning_rate_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        return 0.01 + 0.99 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_factor
    )
    history = []
    best_selection_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    stale_epochs = 0
    run_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(stage.epochs):
        model.train()
        if config.freeze_backbone_during_quantum and stage.core == "quantum":
            keep_frozen_backbone_in_eval(model)
        epoch_start = time.time()
        loss_sum = 0.0
        correct = 0
        seen = 0
        core_gradient_sum = 0.0
        for images, targets, _ in loaders.train:
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    targets,
                    weight=class_weights,
                    label_smoothing=config.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            gradient_squared = 0.0
            for parameter in model.core.parameters():
                if parameter.grad is not None:
                    gradient_squared += float(
                        parameter.grad.detach().float().square().sum()
                    )
            core_gradient_sum += math.sqrt(gradient_squared)
            optimizer.step()
            scheduler.step()

            batch_size = targets.numel()
            seen += batch_size
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum())

        metrics, labels, logits, indices = evaluate(
            model,
            loaders.validation,
            device,
            loaders.class_names,
        )
        selection_key = (
            metrics["balanced_accuracy"],
            metrics["macro_auc_ovr"],
            -metrics["nll"],
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "validation": metrics,
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rate": optimizer.param_groups[1]["lr"],
            "core_learning_rate": optimizer.param_groups[2]["lr"],
            "mean_core_gradient_norm": core_gradient_sum
            / max(len(loaders.train), 1),
            "epoch_seconds": time.time() - epoch_start,
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "selection_key": list(selection_key),
        }
        history.append(record)
        _atomic_json(output_dir / "history.json", history)
        _atomic_checkpoint(
            output_dir / "last.pt",
            {"model": model.state_dict(), "epoch": epoch + 1, "record": record},
        )
        print(f"EPOCH {json.dumps(record, sort_keys=True)}", flush=True)

        if selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch + 1
            stale_epochs = 0
            _atomic_checkpoint(
                output_dir / "best.pt",
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "record": record,
                },
            )
            np.savez_compressed(
                output_dir / "best_validation_predictions.npz",
                indices=indices,
                labels=labels,
                logits=logits,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= stage.patience:
                print(
                    f"EARLY_STOP epoch={epoch + 1} best_epoch={best_epoch}",
                    flush=True,
                )
                break

    best_checkpoint = output_dir / "best.pt"
    checkpoint = torch.load(
        best_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    final_metrics, final_labels, final_logits, _ = evaluate(
        model, loaders.validation, device, loaders.class_names
    )
    symmetry = symmetry_audit(model, loaders.validation, device)
    _atomic_json(output_dir / "symmetry_audit.json", symmetry)
    summary = {
        "stage": stage.name,
        "best_epoch": best_epoch,
        "validation": final_metrics,
        "parameters": model.parameter_report(),
        "symmetry": symmetry["summary"],
        "initialization": initialization,
        "wall_seconds": time.time() - run_start,
        "official_test_evaluated": False,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_validation_artifacts(
        output_dir,
        stage,
        best_epoch,
        final_metrics,
        final_labels,
        final_logits,
        loaders.class_names,
    )
    print(f"SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    return best_checkpoint
