"""CLI entrypoint: build data + model, train, then evaluate.

Examples
--------
Run from the ``src`` directory (so the package is importable)::

    cd src
    python -m steerable_qvf.main --encoding amplitude --dataset-id model_1

Override anything via flags or the same ``DEEPLENSE_*`` env vars as the notebook::

    DEEPLENSE_DATA_ROOT=/path/to/Model_I python -m steerable_qvf.main --epochs 5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import VALID_DATASET_IDS, VALID_ENCODINGS, Config
from .data import build_loaders
from .engine import evaluate, train
from .model import build_model, parameter_summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steerable-QVF Equivariant-QCNN for DeepLense")
    p.add_argument("--dataset-id", choices=sorted(VALID_DATASET_IDS), default=None)
    p.add_argument("--encoding", choices=VALID_ENCODINGS, default=None,
                   help="amplitude | angle | reupload (default: amplitude)")
    p.add_argument("--reupload-layers", type=int, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--base-width", type=int, default=None)
    p.add_argument("--group-n", type=int, default=None)
    p.add_argument("--reflections", action="store_true", default=None)
    p.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Start from env-driven defaults, then apply any explicit CLI overrides."""
    overrides = {}
    mapping = {
        "dataset_id": args.dataset_id,
        "encoding": args.encoding,
        "reupload_layers": args.reupload_layers,
        "data_root": args.data_root,
        "test_dir": args.test_dir,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_width": args.base_width,
        "group_n": args.group_n,
        "use_reflections": args.reflections,
        "augment": args.augment,
        "seed": args.seed,
    }
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value
    return Config.from_env(**overrides)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    import torchquantum as tq
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchQuantum version: {tq.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.ensure_dirs()
    print(f"DATA_ROOT: {cfg.data_root}")
    print(f"TEST_DIR:  {cfg.test_dir}")
    print(f"Group: {cfg.group_name} | qubits: {cfg.n_qubits} | "
          f"amplitudes: {cfg.amp_dim} | encoding: {cfg.encoding}")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = build_loaders(cfg)
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(cfg, num_classes=num_classes, device=device)
    summary = parameter_summary(model)
    enc_desc = cfg.encoding + (f" (L={cfg.reupload_layers})" if cfg.encoding == "reupload" else "")
    print("Steerable-QVF quantum model")
    print(f"  encoding:                   {enc_desc}")
    print(f"  steerable encoder params:   {summary['encoder']}")
    print(f"  quantum (conv/pool) params: {summary['quantum']} ({model.qcnn.n_layers} x 33)")
    print(f"  quantum readout dim:        {model.qcnn.readout_dim}")
    print(f"  residual proj params:       {summary['residual']}")
    print(f"  linear head params:         {summary['head']}")
    print(f"  TOTAL trainable params:     {summary['total']}")

    # Sanity check on one batch.
    _xb, _yb = next(iter(train_loader))
    with torch.no_grad():
        _out = model(_xb.to(device))
    print(f"Sanity: input {tuple(_xb.shape)} -> logits {tuple(_out.shape)}")

    history = train(model, train_loader, val_loader, class_names, cfg, device)
    evaluate(model, test_loader, class_names, cfg, device,
             history=history, param_summary=summary)


if __name__ == "__main__":
    main()
