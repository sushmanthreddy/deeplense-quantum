"""Command-line entry point for the cleaned two-stage D4-ORQB run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import torch

from .config import DATASET_IDS, Config
from .data import (
    INCONCLUSIVE_NO_SIGNAL_DETECTED,
    PASS_SIGNAL_DETECTED,
    build_loaders,
    run_model_iv_audit,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected D4-ORQB pipeline. Dataset paths are "
            "intentionally blank by default."
        )
    )
    parser.add_argument(
        "--dataset-id",
        choices=DATASET_IDS,
        default=defaults.dataset_id,
        help="Dataset family used for cache isolation and run provenance.",
    )
    parser.add_argument("--development-root", default="")
    parser.add_argument(
        "--validation-root",
        default="",
        help=(
            "Required Model-IV validation directory containing the three "
            "class folders; unsupported for other datasets. This is a "
            "development-validation split, not an official test root."
        ),
    )
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--stage",
        choices=("all", "pretrain", "quantum", "audit"),
        default=defaults.stage,
        help=(
            "Run both training stages, one training stage, or the CPU-only "
            "Model-IV signal audit. Model-IV training runs the audit "
            "automatically before CUDA is initialized."
        ),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        default="",
        help=(
            "Optional pretraining checkpoint for --stage quantum. --stage all "
            "creates and uses its own checkpoint."
        ),
    )
    parser.add_argument(
        "--freeze-backbone-during-quantum",
        action="store_true",
        help=(
            "Freeze the checkpoint-initialized deterministic physics bank, "
            "image encoder, and orbit projection during quantum training. "
            "Their evaluation-mode buffers are frozen as well."
        ),
    )
    parser.add_argument(
        "--allow-inconclusive-model-iv-audit",
        action="store_true",
        help=(
            "Explicit research-only override for an inconclusive Model-IV "
            "signal audit. Integrity failure and preprocessing signal loss "
            "cannot be overridden, and the report remains inconclusive."
        ),
    )
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--io-workers", type=int, default=defaults.io_workers)
    parser.add_argument(
        "--val-fraction", type=float, default=defaults.val_fraction
    )
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)

    parser.add_argument(
        "--pretrain-epochs", type=int, default=defaults.pretrain_epochs
    )
    parser.add_argument(
        "--pretrain-patience", type=int, default=defaults.pretrain_patience
    )
    parser.add_argument(
        "--pretrain-seed", type=int, default=defaults.pretrain_seed
    )
    parser.add_argument(
        "--pretrain-learning-rate",
        type=float,
        default=defaults.pretrain_learning_rate,
    )
    parser.add_argument(
        "--pretrain-core-learning-rate",
        type=float,
        default=defaults.pretrain_core_learning_rate,
    )

    parser.add_argument(
        "--quantum-epochs", type=int, default=defaults.quantum_epochs
    )
    parser.add_argument(
        "--quantum-patience", type=int, default=defaults.quantum_patience
    )
    parser.add_argument(
        "--quantum-seed", type=int, default=defaults.quantum_seed
    )
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=defaults.encoder_learning_rate,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate
    )
    parser.add_argument(
        "--core-learning-rate",
        type=float,
        default=defaults.core_learning_rate,
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=defaults.label_smoothing
    )
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args(argv)


def build_config(arguments: argparse.Namespace) -> Config:
    return Config(**vars(arguments))


def main(argv: Sequence[str] | None = None) -> None:
    config = build_config(parse_args(argv))
    config.validate()
    if config.stage == "all" and config.backbone_checkpoint:
        raise ValueError(
            "--stage all creates its own pretraining checkpoint; do not also "
            "set --backbone-checkpoint"
        )
    if config.stage == "pretrain" and config.backbone_checkpoint:
        raise ValueError("--backbone-checkpoint is not used by pretraining")
    if config.stage == "audit" and config.backbone_checkpoint:
        raise ValueError("--backbone-checkpoint is not used by the audit")

    output_root = config.output_path
    output_created = False
    if config.dataset_id == "model_iv":
        if output_root.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing run directory: "
                f"{output_root}"
            )
        output_root.mkdir(parents=True)
        output_created = True
        audit = run_model_iv_audit(config, output_root)
        print(
            f"MODEL_IV_AUDIT_STATUS {audit.status} "
            f"report={audit.report_path}",
            flush=True,
        )
        override = bool(
            audit.status == INCONCLUSIVE_NO_SIGNAL_DETECTED
            and config.allow_inconclusive_model_iv_audit
            and config.stage != "audit"
        )
        if audit.status != PASS_SIGNAL_DETECTED and not override:
            raise RuntimeError(
                "Model-IV training gate did not pass; CUDA training was not "
                f"started. See {audit.report_path}"
            )
        if override:
            print(
                "MODEL_IV_AUDIT_OVERRIDE research_only=true "
                "status_remains_inconclusive=true",
                flush=True,
            )
        if config.stage == "audit":
            return

    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Keep the audit path independent of the training graph and TorchQuantum.
    # This import is intentionally below the Model-IV gate.
    from .engine import pretrain_spec, quantum_spec, train

    quantum_requested = config.stage in ("all", "quantum")
    if quantum_requested:
        from .quantum import require_torchquantum, smoke_test_torchquantum

        require_torchquantum()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA-capable GPU")

    device = torch.device("cuda")
    if quantum_requested:
        print(f"QUANTUM_BACKEND {smoke_test_torchquantum(device)}", flush=True)
        torch.cuda.empty_cache()

    if output_root.exists() and not output_created:
        raise FileExistsError(
            f"Refusing to overwrite an existing run directory: {output_root}"
        )
    if not output_created:
        output_root.mkdir(parents=True)

    backbone_checkpoint: Path | None = None
    if config.stage in ("all", "pretrain"):
        specification = pretrain_spec(config)
        loaders = build_loaders(config, specification.seed, device)
        backbone_checkpoint = train(
            config,
            loaders,
            specification,
            output_root / specification.name,
            device,
        )

    if config.stage in ("all", "quantum"):
        specification = quantum_spec(config)
        loaders = build_loaders(config, specification.seed, device)
        if config.stage == "quantum" and config.backbone_checkpoint:
            backbone_checkpoint = Path(config.backbone_checkpoint).expanduser()
        train(
            config,
            loaders,
            specification,
            output_root / specification.name,
            device,
            backbone_checkpoint=backbone_checkpoint,
        )


if __name__ == "__main__":
    main()
