"""Aggregate paired D4-ORQB runs and compute reproducible uncertainty tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .metrics import classification_metrics


CLASS_NAMES = ["axion", "cdm", "no_sub"]


def probability_ensemble_logits(logits: Iterable[np.ndarray]) -> np.ndarray:
    """Return log probabilities for a uniform, predeclared probability ensemble."""

    probabilities = []
    for value in logits:
        shifted = value - value.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities.append(exponentiated / exponentiated.sum(axis=1, keepdims=True))
    mean_probability = np.mean(probabilities, axis=0)
    return np.log(np.clip(mean_probability, 1e-12, 1.0))


def load_predictions(run_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(Path(run_dir) / "best_validation_predictions.npz")
    order = np.argsort(data["indices"])
    return data["indices"][order], data["labels"][order], data["logits"][order]


def paired_bootstrap_accuracy(
    labels: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    samples: int = 10_000,
    seed: int = 20260711,
) -> Dict[str, float]:
    delta = (
        (logits_a.argmax(1) == labels).astype(np.float64)
        - (logits_b.argmax(1) == labels).astype(np.float64)
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk = 100
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        estimates[start:stop] = delta[indices].mean(axis=1)
    return {
        "difference": float(delta.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": samples,
    }


def mcnemar_exact(labels: np.ndarray, logits_a: np.ndarray, logits_b: np.ndarray) -> Dict:
    correct_a = logits_a.argmax(1) == labels
    correct_b = logits_b.argmax(1) == labels
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_only, b_only)
        numerator = sum(math.comb(discordant, k) for k in range(lower + 1))
        p_value = min(1.0, 2.0 * float(numerator / (1 << discordant)))
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def scalar_metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    metrics = classification_metrics(labels, logits, CLASS_NAMES)
    return {
        key: metrics[key]
        for key in ("accuracy", "macro_f1", "macro_auc_ovr", "nll", "brier", "ece_15")
    }


def mean_std(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return result


def analyze_seed(seed: int, root: Path, bootstrap_samples: int) -> Tuple[Dict, Dict[str, np.ndarray]]:
    loaded = {}
    labels_ref = indices_ref = None
    for name in ("quantum", "classical", "pretrain-context"):
        indices, labels, logits = load_predictions(root / name)
        if labels_ref is None:
            indices_ref, labels_ref = indices, labels
        elif not np.array_equal(indices, indices_ref) or not np.array_equal(labels, labels_ref):
            raise ValueError(f"Prediction manifest mismatch for seed {seed}, model {name}")
        loaded[name] = logits
    q, c, context = loaded["quantum"], loaded["classical"], loaded["pretrain-context"]
    report = {
        "seed": seed,
        "quantum": scalar_metrics(labels_ref, q),
        "classical": scalar_metrics(labels_ref, c),
        "context": scalar_metrics(labels_ref, context),
        "equal_quantum_classical_ensemble": scalar_metrics(
            labels_ref, probability_ensemble_logits((q, c))
        ),
        "quantum_minus_classical": paired_bootstrap_accuracy(
            labels_ref, q, c, samples=bootstrap_samples, seed=20260711 + seed
        ),
        "mcnemar_quantum_vs_classical": mcnemar_exact(labels_ref, q, c),
    }
    loaded["labels"] = labels_ref
    loaded["indices"] = indices_ref
    return report, loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-run",
        action="append",
        required=True,
        help="SEED=/absolute/path containing quantum, classical, pretrain-context",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()

    seed_roots = []
    for item in args.seed_run:
        seed_text, path_text = item.split("=", 1)
        seed_roots.append((int(seed_text), Path(path_text)))
    seed_roots.sort()

    reports, arrays = [], []
    for seed, root in seed_roots:
        report, loaded = analyze_seed(seed, root, args.bootstrap_samples)
        reports.append(report)
        arrays.append(loaded)

    aggregate = {
        "seeds": [seed for seed, _ in seed_roots],
        "per_seed": reports,
        "quantum": mean_std([report["quantum"] for report in reports]),
        "classical": mean_std([report["classical"] for report in reports]),
        "equal_quantum_classical_ensemble": mean_std(
            [report["equal_quantum_classical_ensemble"] for report in reports]
        ),
    }
    reference_indices = arrays[0]["indices"]
    reference_labels = arrays[0]["labels"]
    for loaded in arrays[1:]:
        if not np.array_equal(loaded["indices"], reference_indices):
            raise ValueError("Seed prediction indices differ")
        if not np.array_equal(loaded["labels"], reference_labels):
            raise ValueError("Seed labels differ")
    quantum_seed_ensemble = probability_ensemble_logits(
        loaded["quantum"] for loaded in arrays
    )
    classical_seed_ensemble = probability_ensemble_logits(
        loaded["classical"] for loaded in arrays
    )
    aggregate["quantum_seed_ensemble"] = scalar_metrics(reference_labels, quantum_seed_ensemble)
    aggregate["classical_seed_ensemble"] = scalar_metrics(reference_labels, classical_seed_ensemble)
    aggregate["all_paired_models_ensemble"] = scalar_metrics(
        reference_labels,
        probability_ensemble_logits((quantum_seed_ensemble, classical_seed_ensemble)),
    )
    aggregate["ensemble_quantum_minus_classical"] = paired_bootstrap_accuracy(
        reference_labels,
        quantum_seed_ensemble,
        classical_seed_ensemble,
        samples=args.bootstrap_samples,
    )
    aggregate["ensemble_mcnemar"] = mcnemar_exact(
        reference_labels, quantum_seed_ensemble, classical_seed_ensemble
    )

    rendered = json.dumps(aggregate, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
