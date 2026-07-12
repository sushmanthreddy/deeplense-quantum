"""Train LensPINN-small on the fixed Model-I protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    CachedNPYDataset,
    fixed_stratified_split,
    make_loader,
    prepare_cache,
    verify_cache_disjoint,
)
from .lenspinn import LensPINNSmall, lenspinn_distortion
from .metrics import classification_metrics
from .train import atomic_checkpoint, atomic_json, seed_everything, softmax_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--faithful-softmax", action="store_true")
    parser.add_argument(
        "--reconstruction",
        choices=("archived-hard", "differentiable"),
        default="differentiable",
    )
    parser.add_argument("--retain-archived-unused-block", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    labels_all, logits_all, indices_all = [], [], []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        distortion = lenspinn_distortion(images)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images, distortion)
        labels_all.append(labels.numpy())
        logits_all.append(logits.float().cpu().numpy())
        indices_all.append(indices.numpy())
    labels_np = np.concatenate(labels_all)
    logits_np = np.concatenate(logits_all)
    indices_np = np.concatenate(indices_all)
    return (
        classification_metrics(labels_np, logits_np, list(class_names)),
        labels_np,
        logits_np,
        indices_np,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LensPINN training requires a Kubeflow CUDA job")
    seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", vars(args))
    print(f"RUNTIME torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CONFIG {json.dumps(vars(args), sort_keys=True)}", flush=True)

    cache_root = Path(args.cache_root) / f"model_i_{args.image_size}_v1"
    development_cache, test_cache = cache_root / "development", cache_root / "test"
    development_meta = prepare_cache(
        args.development_root, development_cache, args.image_size, device, args.io_workers
    )
    class_names = development_meta["classes"]
    test_meta = None
    disjoint = None
    if args.evaluate_test:
        test_meta = prepare_cache(
            args.test_root, test_cache, args.image_size, device, args.io_workers
        )
        disjoint = verify_cache_disjoint(development_cache, test_cache)
        if test_meta["classes"] != class_names:
            raise RuntimeError("Development/test class order differs")
    labels = np.load(development_cache / "labels.npy")
    split_path = cache_root / f"split_seed{args.split_seed}_val{args.val_fraction:.4f}.npz"
    train_indices, val_indices = fixed_stratified_split(
        labels, split_path, args.val_fraction, args.split_seed
    )
    np.savez(output_dir / "split_indices.npz", train=train_indices, val=val_indices)
    data_report = {
        "development": development_meta,
        "train_size": int(len(train_indices)),
        "validation_size": int(len(val_indices)),
        "official_test_locked_during_selection": not args.evaluate_test,
        "official_test_cache_opened": bool(args.evaluate_test),
    }
    if test_meta is not None:
        data_report["test"] = test_meta
        data_report["digest_disjoint"] = disjoint
    atomic_json(output_dir / "data_report.json", data_report)

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
    model = LensPINNSmall(
        image_size=args.image_size,
        patch_size=args.patch_size,
        pretrained=not args.no_pretrained,
        logits_fix=not args.faithful_softmax,
        reconstruction=args.reconstruction,
        retain_archived_unused_block=args.retain_archived_unused_block,
    ).to(device, memory_format=torch.channels_last)
    count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
    unused_parameters = (
        count(model.archived_unused_transformer)
        if model.archived_unused_transformer is not None
        else 0
    )
    parameter_report = {
        "total": count(model),
        "active_forward_graph": count(model) - unused_parameters,
        "archived_unused_registered": unused_parameters,
        "tokenizer_and_inversion": count(model.tokenizer) + count(model.inversion),
        "observed_source_decoder": count(model.decoder_observed_source),
        "distortion_decoder": count(model.decoder_distortion),
        "classifier": count(model.classifier),
        "pretrained": not args.no_pretrained,
        "cross_entropy_logits_fix": not args.faithful_softmax,
        "reconstruction": args.reconstruction,
    }
    atomic_json(output_dir / "parameter_report.json", parameter_report)
    print(f"PARAMETERS {json.dumps(parameter_report, sort_keys=True)}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_accuracy, best_epoch, stale = -1.0, -1, 0
    run_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        seen = 0
        inversion_grad_sum = torch.zeros((), device=device)
        tokenizer_grad_sum = torch.zeros((), device=device)
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            distortion = lenspinn_distortion(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images, distortion)
                loss_input = logits.softmax(dim=1) if args.faithful_softmax else logits
                loss = F.cross_entropy(loss_input, targets)
            loss.backward()
            grad_sq = torch.zeros((), device=device)
            for parameter in model.inversion.parameters():
                if parameter.grad is not None:
                    grad_sq = grad_sq + parameter.grad.detach().float().square().sum()
            inversion_grad_sum = inversion_grad_sum + grad_sq.sqrt()
            tokenizer_grad_sq = torch.zeros((), device=device)
            for parameter in model.tokenizer.parameters():
                if parameter.grad is not None:
                    tokenizer_grad_sq = (
                        tokenizer_grad_sq
                        + parameter.grad.detach().float().square().sum()
                    )
            tokenizer_grad_sum = tokenizer_grad_sum + tokenizer_grad_sq.sqrt()
            optimizer.step()
            batch = targets.numel()
            running_loss = running_loss + loss.detach() * batch
            correct = correct + (logits.argmax(1) == targets).sum()
            seen += batch

        metrics, val_labels, val_logits, val_indices_out = evaluate(
            model, val_loader, device, class_names
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(running_loss / seen),
            "train_accuracy": float(correct / seen),
            "validation": metrics,
            "mean_inversion_gradient_norm": float(
                inversion_grad_sum / max(len(train_loader), 1)
            ),
            "mean_tokenizer_gradient_norm": float(
                tokenizer_grad_sum / max(len(train_loader), 1)
            ),
            "epoch_seconds": time.time() - epoch_start,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        history.append(record)
        if args.reconstruction == "differentiable" and epoch == 0:
            if record["mean_inversion_gradient_norm"] <= 0.0:
                raise RuntimeError("Differentiable LensPINN inversion received zero gradient")
            if record["mean_tokenizer_gradient_norm"] <= 0.0:
                raise RuntimeError("Differentiable LensPINN tokenizer received zero gradient")
        atomic_json(output_dir / "history.json", history)
        atomic_checkpoint(
            output_dir / "last.pt", {"model": model.state_dict(), "epoch": epoch + 1, "record": record}
        )
        print(f"EPOCH {json.dumps(record, sort_keys=True)}", flush=True)
        if metrics["accuracy"] > best_accuracy + 1e-12:
            best_accuracy, best_epoch, stale = metrics["accuracy"], epoch + 1, 0
            atomic_checkpoint(
                output_dir / "best.pt",
                {"model": model.state_dict(), "epoch": best_epoch, "record": record},
            )
            np.savez_compressed(
                output_dir / "best_validation_predictions.npz",
                indices=val_indices_out,
                labels=val_labels,
                logits=val_logits,
                probabilities=softmax_numpy(val_logits),
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"EARLY_STOP epoch={epoch + 1} best_epoch={best_epoch}", flush=True)
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    final_validation, _, _, _ = evaluate(model, val_loader, device, class_names)
    summary = {
        "best_epoch": best_epoch,
        "validation": final_validation,
        "parameters": parameter_report,
        "official_test_evaluated": bool(args.evaluate_test),
        "wall_seconds": time.time() - run_start,
    }
    if args.evaluate_test:
        test_loader = make_loader(
            CachedNPYDataset(test_cache), args.batch_size, False, args.workers, args.split_seed
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
    print(f"FINAL_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
