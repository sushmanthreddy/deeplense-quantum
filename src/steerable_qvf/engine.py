"""Training and evaluation loops (cross-entropy, Adam, ReduceLROnPlateau, early stop)."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from .config import Config


def make_amp(device: torch.device):
    """Return (use_amp, scaler). AMP is off by default (e2cnn is finicky under fp16)."""
    use_amp = torch.cuda.is_available() and os.environ.get("DEEPLENSE_AMP", "0") != "0"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (AMP): {'ON' if use_amp else 'OFF'}")
    return use_amp, scaler


def run_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, train_mode):
    model.train(train_mode)
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train_mode)
    desc = "train" if train_mode else "val"
    for xb, yb in tqdm(loader, desc=desc, leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


def train(model, train_loader, val_loader, class_names, cfg: Config, device: torch.device):
    """Full training loop with checkpointing + early stopping. Returns the history dict."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    use_amp, scaler = make_amp(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, epochs_no_improve = 0.0, 0
    cfg.ensure_dirs()

    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, True
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, use_amp, False
        )
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:02d}/{cfg.num_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"lr {lr_now:.2e} | {time.time() - t0:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "val_acc": best_val_acc,
                        "epoch": epoch,
                        "classes": class_names}, cfg.checkpoint_path)
            print(f"  -> saved best (val_acc={best_val_acc:.4f}) to {cfg.checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(no val improvement for {cfg.patience} epochs).")
                break

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    history["best_val_acc"] = best_val_acc
    return history


@torch.no_grad()
def evaluate(model, test_loader, class_names, cfg: Config, device: torch.device,
             history=None, param_summary=None):
    """Load the best checkpoint, run the test set, print + save metrics. Returns a dict."""
    num_classes = len(class_names)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded best checkpoint (val_acc={ckpt['val_acc']:.4f}, epoch={ckpt['epoch'] + 1}).")

    all_logits, all_labels = [], []
    for xb, yb in tqdm(test_loader, desc="test", leave=False):
        all_logits.append(model(xb.to(device)).cpu())
        all_labels.append(yb)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(1)

    test_acc = float((preds == labels).mean())
    y_onehot = label_binarize(labels, classes=list(range(num_classes)))
    try:
        macro_auc = roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr")
    except ValueError:
        macro_auc = float("nan")
    macro_f1 = f1_score(labels, preds, average="macro")

    print(f"Test accuracy:      {test_acc:.4f}")
    print(f"Test macro ROC-AUC: {macro_auc:.4f}")
    print(f"Test macro F1:      {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    metrics = {
        "test_acc": test_acc, "macro_auc": float(macro_auc), "macro_f1": float(macro_f1),
        "group": cfg.group_name, "classes": class_names,
    }
    if history is not None:
        metrics["best_val_acc"] = history.get("best_val_acc")
    if param_summary is not None:
        metrics["total_params"] = param_summary["total"]
        metrics["encoder_params"] = param_summary["encoder"]
        metrics["quantum_params"] = param_summary["quantum"]
        metrics["head_params"] = param_summary["head"]

    cfg.ensure_dirs()
    _save_plots(cm_labels=labels, preds=preds, class_names=class_names,
                history=history, test_acc=test_acc, cfg=cfg)
    with open(cfg.results_dir / "steerable_qvf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics + figure to {cfg.results_dir}")
    return metrics


def _save_plots(cm_labels, preds, class_names, history, test_acc, cfg: Config):
    import matplotlib.pyplot as plt
    num_classes = len(class_names)
    cm = confusion_matrix(cm_labels, preds)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_title(f"Confusion matrix (steerable-QVF, {test_acc * 100:.2f}%)")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_xticks(range(num_classes))
    ax[0].set_xticklabels(class_names, rotation=45)
    ax[0].set_yticks(range(num_classes))
    ax[0].set_yticklabels(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            ax[0].text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    if history is not None:
        ax[1].plot(history["train_acc"], label="train acc")
        ax[1].plot(history["val_acc"], label="val acc")
        ax[1].axhline(1.0 / num_classes, ls="--", c="red",
                      label=f"chance ({1 / num_classes:.2f})")
        ax[1].set_title("Accuracy curve")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.results_dir / "steerable_qvf_test_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
