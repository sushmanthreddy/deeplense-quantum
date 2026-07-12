from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


REPO = Path("/home/jovyan/susmered-datavol-1/repos/deeplense-quantum")
SOURCE = REPO / "_best_model_notebook_source.py"
OUTPUT = REPO / "Best_D4_ORQB_Quantum_Model.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


model_source = SOURCE.read_text()
model_source = model_source.split('\nif __name__ == "__main__":', 1)[0]
model_source = model_source.replace(
    "/home/jovyan/susmered-datavol-1/outputs/deeplense-quantum/",
    "/home/jovyan/data/outputs/deeplense-quantum/",
)

cells = [
    md(
        r"""
        # Best D4 Orbit-Reuploading Quantum Bottleneck for Model I

        **Executed, self-contained architecture and paper report**

        This notebook contains only the selected seed-2 hybrid quantum model used for three-class gravitational-lensing classification. It defines the exact model, strictly loads the frozen checkpoint, audits all parameters, performs a live CPU inference and D4-invariance check, recomputes the sealed official-test metrics, and generates paper-ready figures.

        **Selected result:** 98.4746% development-validation accuracy and **98.3667% official-test accuracy** on 15,000 held-out images. The official-test prediction bundle was generated in the prospectively sealed v3 pass; this notebook analyzes that immutable bundle and does not tune on the test set.

        The quantum circuit is evaluated with the exact differentiable PyTorch statevector implementation used for training. No optional quantum package is needed to reproduce the selected checkpoint.
        """
    ),
    md(
        r"""
        ## Reproducibility contract

        - Classes: `axion`, `cdm`, `no_sub`.
        - Development split: 70,021 training and 17,504 validation images, stratified with seed 42.
        - Official test: 15,000 balanced images, exactly 5,000 per class.
        - Model selection used development validation only.
        - Selected checkpoint: seed 2, best epoch 19.
        - Checkpoint, predictions, and result JSON are checked against fixed SHA-256 digests before analysis.
        - Representative images are selected deterministically from the development-validation split. Official-test pixels are not opened for figure selection.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import hashlib
        import json
        import math
        import os
        import platform

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        import numpy as np
        import torch
        from IPython.display import display, Markdown

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.manual_seed(2)
        np.random.seed(2)

        DATA_ROOT = Path(os.environ.get("DEEPLENSE_DATA_ROOT", "/home/jovyan/data"))
        SEALED_MEMBER = DATA_ROOT / "outputs/deeplense-quantum/d4-orqb/model-i/locked/d4-orqb-model-i-v3-20260711-seal/members/q2"
        OFFICIAL = DATA_ROOT / "outputs/deeplense-quantum/d4-orqb/model-i/locked/d4-orqb-model-i-v3-20260711-official-test"
        CACHE = DATA_ROOT / "cache/d4-orqb/model_i_96_v1"
        CHECKPOINT = SEALED_MEMBER / "best.pt"
        VALIDATION_SUMMARY = SEALED_MEMBER / "summary.json"
        HISTORY = SEALED_MEMBER / "history.json"
        TEST_PREDICTIONS = OFFICIAL / "predictions/q2.npz"
        TEST_RESULT = OFFICIAL / "result.json"
        FIGURE_DIR = DATA_ROOT / "outputs/deeplense-quantum-paper-best-q2/figures"
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        EXPECTED_SHA256 = {
            CHECKPOINT: "9b6bb5bab038b086c7e6dd3b0448de9fa204e6138e28da8bb13deb9374f87007",
            TEST_PREDICTIONS: "1479ffe28b950320166f913ed8dd53b0d9dceb4626faf79444ea17f9e96ae691",
            TEST_RESULT: "86b0b1df9cf33a772719799dd0d7632e81312e837c0de39e0003fc4e8ca83340",
        }

        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        missing = [str(path) for path in EXPECTED_SHA256 if not path.is_file()]
        assert not missing, f"Missing required artifacts: {missing}"
        observed_hashes = {str(path): sha256(path) for path in EXPECTED_SHA256}
        for path, expected in EXPECTED_SHA256.items():
            assert observed_hashes[str(path)] == expected, f"SHA-256 mismatch: {path}"

        CLASS_NAMES = ("Axion", "CDM", "No substructure")
        COLORS = ("#6A5ACD", "#E45756", "#2A9D8F")
        mpl.rcParams.update({
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        print({
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "artifact_hashes_verified": len(EXPECTED_SHA256),
            "paper_figure_directory": str(FIGURE_DIR),
        })
        """
    ),
    md(
        r"""
        ## Exact selected architecture

        1. Each 96×96 intensity image is converted into eight fixed physics-informed maps: normalized intensity, log intensity, gradient magnitude, Laplacian magnitude, difference of local averages, radial gradient, tangential gradient, and a stabilized mixed derivative.
        2. All eight elements of the dihedral group D4 are applied to the image. One shared MobileNet-style MBConv feature extractor produces a 128-dimensional descriptor for every orbit view.
        3. A shared linear orbit projection maps each descriptor to two angle fields for each of four quantum heads, giving a tensor of shape `batch × 4 × 2 × 8`.
        4. Each head is an eight-qubit circuit with two data reuploads. Every reupload performs dense RY/RZ data encoding, tied local RX/RY/RZ rotations, and symmetry-tied XX/ZZ interactions over complete D4 Cayley-edge orbits.
        5. Local X/Z and pair XX/ZZ expectations are orbit-reduced into 12 invariant values per head, producing a 48-dimensional D4-invariant quantum bottleneck.
        6. A compact prediction head maps the 48 invariants to three class logits.

        The cell below is the complete minimal model definition. It contains only this selected path and preserves every checkpoint key exactly.
        """
    ),
    code(model_source),
    code(
        r"""
        model, checkpoint_payload = BestD4ORQB.from_checkpoint(CHECKPOINT)
        CLASS_NAMES = ("Axion", "CDM", "No substructure")
        assert checkpoint_payload["epoch"] == 19
        assert model.parameter_count == 245_221

        parameter_breakdown = {
            "Physics feature bank": sum(p.numel() for p in model.physics.parameters()),
            "Shared feature extractor": sum(p.numel() for p in model.encoder.parameters()),
            "Orbit-to-angle projection": sum(p.numel() for p in model.orbit_projection.parameters()),
            "Quantum circuit": sum(p.numel() for p in model.core.parameters()),
            "Prediction head": sum(p.numel() for p in model.head.parameters()),
        }
        assert sum(parameter_breakdown.values()) == 245_221
        assert parameter_breakdown["Quantum circuit"] == 88

        parameter_rows = [
            (name, value, 100 * value / model.parameter_count)
            for name, value in parameter_breakdown.items()
        ]
        parameter_markdown = ["| Component | Trainable parameters | Share |", "|---|---:|---:|"]
        parameter_markdown.extend(
            f"| {name} | {value:,} | {share:.4f}% |"
            for name, value, share in parameter_rows
        )
        display(Markdown("\n".join(parameter_markdown)))
        print("Strict checkpoint load: PASSED")
        print("Circuit tensor:", tuple(model.core.params.shape), "= 4 heads × 2 reuploads × 11 tied parameters")
        """
    ),
    code(
        r"""
        def save_paper_figure(fig, stem: str):
            fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
            display(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 3.8))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 4)
        ax.axis("off")
        blocks = [
            (0.15, "96×96 lens\nimage", "#E8F1FA"),
            (2.0, "8 D4 orbit\nviews", "#DDEBF7"),
            (3.85, "8-map physics\nfeature bank", "#CCE5FF"),
            (5.85, "Shared MBConv\n128/view", "#B8D8F0"),
            (7.85, "Angle projection\n4×2×8", "#A8DADC"),
            (9.75, "4 quantum heads\n8 qubits/head\n2 reuploads", "#CDB4DB"),
            (11.75, "48 D4-invariant\nexpectations", "#BDE0FE"),
            (13.15, "3-class\noutput", "#B7E4C7"),
        ]
        widths = [1.35, 1.35, 1.55, 1.55, 1.45, 1.65, 1.25, 0.75]
        for i, ((x, label, color), width) in enumerate(zip(blocks, widths)):
            box = FancyBboxPatch((x, 1.25), width, 1.45, boxstyle="round,pad=0.08",
                                 linewidth=1.4, edgecolor="#264653", facecolor=color)
            ax.add_patch(box)
            ax.text(x + width/2, 1.98, label, ha="center", va="center", fontsize=8.3, weight="bold")
            if i < len(blocks) - 1:
                nx = blocks[i+1][0]
                ax.add_patch(FancyArrowPatch((x + width + .04, 1.98), (nx - .06, 1.98),
                                             arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
                                             color="#264653"))
        ax.text(7.0, 3.35, "D4 Orbit-Reuploading Quantum Bottleneck (D4-ORQB)",
                ha="center", fontsize=15, weight="bold", color="#1D3557")
        ax.text(7.0, .55, "245,221 trainable parameters • 88 circuit parameters • 4 × 8 qubits • 2 reuploads • 48 invariant observables",
                ha="center", fontsize=10.5, color="#334155")
        save_paper_figure(fig, "fig01_d4_orqb_architecture")
        """
    ),
    code(
        r"""
        fig, ax = plt.subplots(figsize=(9, 4.8))
        names = list(parameter_breakdown)
        values = np.array(list(parameter_breakdown.values()))
        y = np.arange(len(names))
        bars = ax.barh(y, values, color=["#CBD5E1", "#457B9D", "#76B5C5", "#8E6BBE", "#2A9D8F"])
        ax.set_xscale("symlog", linthresh=100)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("Trainable parameters (symlog scale)")
        ax.set_title("Exact parameter allocation of the selected model")
        for bar, value in zip(bars, values):
            ax.text(max(value, 1) * 1.12, bar.get_y() + bar.get_height()/2,
                    f"{value:,}", va="center", fontsize=9, weight="bold")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        save_paper_figure(fig, "fig02_parameter_allocation")
        """
    ),
    md(
        r"""
        ## Live checkpoint inference and exact D4-invariance check

        The next cells use deterministic examples from the development-validation split. This verifies that the notebook's standalone implementation loads the selected weights, produces finite outputs, and remains invariant under all eight D4 transformations.
        """
    ),
    code(
        r"""
        development_images = np.load(CACHE / "development/images.npy", mmap_mode="r")
        development_labels = np.load(CACHE / "development/labels.npy", mmap_mode="r")
        split = np.load(CACHE / "split_seed42_val0.2000.npz")
        validation_indices = split["val"]

        selected_indices = {
            label: validation_indices[development_labels[validation_indices] == label][:3]
            for label in range(3)
        }

        def display_transform(image):
            image = np.asarray(image, dtype=np.float32)
            return np.log1p(30 * np.clip(image, 0, None)) / np.log(31)

        fig, axes = plt.subplots(3, 3, figsize=(8.3, 8.2))
        for label, row in selected_indices.items():
            for col, index in enumerate(row):
                axes[label, col].imshow(display_transform(development_images[index]), cmap="magma", vmin=0, vmax=1)
                axes[label, col].set_title(f"{CLASS_NAMES[label]} • val index {int(index)}", fontsize=9)
                axes[label, col].axis("off")
        fig.suptitle("Deterministic development-validation examples", fontsize=14, weight="bold")
        fig.tight_layout()
        save_paper_figure(fig, "fig03_development_lens_examples")
        """
    ),
    code(
        r"""
        reference_index = int(selected_indices[0][0])
        reference_np = np.asarray(development_images[reference_index], dtype=np.float32)
        reference = torch.tensor(reference_np.tolist(), dtype=torch.float32)[None, None]
        orbit_batch = torch.cat([
            d4_transform(reference, rotation, reflected)
            for reflected in (0, 1) for rotation in range(4)
        ], dim=0)

        with torch.inference_mode():
            logits, auxiliary = model(reference, return_aux=True)
            orbit_logits = model(orbit_batch)
            orbit_probabilities = torch.softmax(orbit_logits, dim=1)

        assert tuple(auxiliary["angles"].shape) == (1, 4, 2, 8)
        assert tuple(auxiliary["invariants"].shape) == (1, 48)
        max_logit_difference = float((orbit_logits - orbit_logits[:1]).abs().max())
        assert max_logit_difference < 2e-5

        fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
        for i, ax in enumerate(axes.flat):
            orbit_image = np.asarray(orbit_batch[i, 0].tolist(), dtype=np.float32)
            ax.imshow(display_transform(orbit_image), cmap="magma", vmin=0, vmax=1)
            transform_name = f"r^{i % 4}" + ("s" if i >= 4 else "")
            ax.set_title(transform_name)
            ax.axis("off")
        fig.suptitle(f"D4 orbit of one validation image • max logit difference = {max_logit_difference:.2e}",
                     fontsize=12, weight="bold")
        fig.tight_layout()
        save_paper_figure(fig, "fig04_d4_orbit_invariance")

        print({
            "live_prediction": CLASS_NAMES[int(logits.argmax(1))],
            "live_probabilities": dict(zip(CLASS_NAMES, torch.softmax(logits, 1)[0].tolist())),
            "angles_shape": tuple(auxiliary["angles"].shape),
            "invariants_shape": tuple(auxiliary["invariants"].shape),
            "maximum_D4_logit_difference": max_logit_difference,
        })
        """
    ),
    code(
        r"""
        quantum_parameter_names = [
            "data RY scale", "data RY bias", "data RZ scale", "data RZ bias",
            "local RX", "local RY", "local RZ", "R-edge ZZ", "S-edge ZZ",
            "R-edge XX", "S-edge XX",
        ]
        learned_circuit = np.asarray(model.core.params.detach().cpu().tolist(), dtype=np.float64).reshape(8, 11)
        row_names = [f"head {head+1}, upload {upload+1}" for head in range(4) for upload in range(2)]
        limit = max(abs(learned_circuit.min()), abs(learned_circuit.max()))
        fig, ax = plt.subplots(figsize=(11, 5.2))
        image = ax.imshow(learned_circuit, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_xticks(np.arange(11), quantum_parameter_names, rotation=42, ha="right")
        ax.set_yticks(np.arange(8), row_names)
        ax.set_title("Learned 88-parameter quantum circuit")
        fig.colorbar(image, ax=ax, label="Learned angle / scale")
        fig.tight_layout()
        save_paper_figure(fig, "fig05_learned_quantum_parameters")
        """
    ),
    md(
        r"""
        ## Sealed official-test evaluation

        The following analysis reads only the immutable q2 prediction arrays and recomputes all central metrics. It neither performs model selection nor changes thresholds. The stored test result was produced on one NVIDIA H200 using bfloat16 inference, batch size 128, loader seed 42.
        """
    ),
    code(
        r"""
        with np.load(TEST_PREDICTIONS) as bundle:
            test_labels = bundle["labels"].astype(np.int64)
            test_probabilities = bundle["probabilities"].astype(np.float64)
            test_predictions = bundle["predictions"].astype(np.int64)

        official_result = json.loads(TEST_RESULT.read_text())
        sealed_metrics = official_result["metrics"]["q2"]
        validation_metrics = json.loads(VALIDATION_SUMMARY.read_text())["validation"]
        training_history = json.loads(HISTORY.read_text())

        assert len(test_labels) == 15_000
        assert np.array_equal(np.bincount(test_labels, minlength=3), [5_000, 5_000, 5_000])
        assert np.allclose(test_probabilities.sum(axis=1), 1.0, atol=1e-12)
        assert np.array_equal(test_predictions, test_probabilities.argmax(axis=1))

        def confusion_matrix_np(labels, predictions, classes=3):
            matrix = np.zeros((classes, classes), dtype=np.int64)
            np.add.at(matrix, (labels, predictions), 1)
            return matrix

        def binary_roc(labels_binary, scores):
            order = np.argsort(-scores, kind="mergesort")
            y = labels_binary[order].astype(np.int64)
            s = scores[order]
            distinct = np.r_[np.where(np.diff(s))[0], y.size - 1]
            tp = np.cumsum(y)[distinct]
            fp = 1 + distinct - tp
            positives = y.sum()
            negatives = y.size - positives
            tpr = np.r_[0.0, tp / positives]
            fpr = np.r_[0.0, fp / negatives]
            return fpr, tpr, float(np.trapz(tpr, fpr))

        def binary_pr(labels_binary, scores):
            order = np.argsort(-scores, kind="mergesort")
            y = labels_binary[order].astype(np.int64)
            s = scores[order]
            distinct = np.r_[np.where(np.diff(s))[0], y.size - 1]
            tp = np.cumsum(y)[distinct]
            fp = 1 + distinct - tp
            precision = tp / np.maximum(tp + fp, 1)
            recall = tp / max(y.sum(), 1)
            return np.r_[0.0, recall], np.r_[1.0, precision]

        confusion = confusion_matrix_np(test_labels, test_predictions)
        accuracy = float(np.mean(test_labels == test_predictions))
        recalls = np.diag(confusion) / confusion.sum(axis=1)
        precisions = np.diag(confusion) / confusion.sum(axis=0)
        f1s = 2 * precisions * recalls / (precisions + recalls)
        balanced_accuracy = float(recalls.mean())
        macro_f1 = float(f1s.mean())
        nll = float(-np.log(np.clip(test_probabilities[np.arange(len(test_labels)), test_labels], 1e-15, 1)).mean())
        one_hot = np.eye(3)[test_labels]
        brier = float(np.square(test_probabilities - one_hot).sum(axis=1).mean())
        roc_data = [binary_roc(test_labels == cls, test_probabilities[:, cls]) for cls in range(3)]
        macro_auc = float(np.mean([item[2] for item in roc_data]))

        assert np.array_equal(confusion, np.asarray(sealed_metrics["confusion_matrix"]))
        checks = {
            "accuracy": (accuracy, sealed_metrics["accuracy"]),
            "balanced_accuracy": (balanced_accuracy, sealed_metrics["balanced_accuracy"]),
            "macro_f1": (macro_f1, sealed_metrics["macro_f1"]),
            "macro_auc_ovr": (macro_auc, sealed_metrics["macro_auc_ovr"]),
            "nll": (nll, sealed_metrics["nll"]),
            "brier": (brier, sealed_metrics["brier"]),
        }
        for name, (recomputed, sealed) in checks.items():
            assert math.isclose(recomputed, sealed, rel_tol=0, abs_tol=2e-12), (name, recomputed, sealed)

        overall_rows = [
            ("Training (best epoch)", 70_021, checkpoint_payload["record"]["train_accuracy"], None, None),
            ("Development validation", validation_metrics["samples"], validation_metrics["accuracy"],
             validation_metrics["macro_auc_ovr"], validation_metrics["macro_f1"]),
            ("Official test", sealed_metrics["samples"], accuracy, macro_auc, macro_f1),
        ]
        overall_markdown = [
            "| Split | Images | Accuracy | Macro AUC | Macro F1 |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {name} | {images:,} | {acc:.4%} | "
                f"{('—' if auc is None else f'{auc:.6f}')} | "
                f"{('—' if f1 is None else f'{f1:.4%}')} |"
                for name, images, acc, auc, f1 in overall_rows
            ],
        ]
        display(Markdown("\n".join(overall_markdown)))
        print("Sealed q2 metric replay: PASSED")
        print(f"Official test: {confusion.trace():,}/{confusion.sum():,} correct")
        print("Wilson 95% accuracy interval:", sealed_metrics["accuracy_wilson95"])
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        normalized = confusion / confusion.sum(axis=1, keepdims=True)
        for ax, values, title, formatter in [
            (axes[0], confusion, "Official-test confusion matrix (counts)", lambda x: f"{int(x):,}"),
            (axes[1], normalized, "Official-test confusion matrix (row %)", lambda x: f"{100*x:.2f}%"),
        ]:
            im = ax.imshow(values, cmap="Blues", vmin=0)
            ax.set_xticks(range(3), CLASS_NAMES, rotation=25, ha="right")
            ax.set_yticks(range(3), CLASS_NAMES)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(title)
            threshold = values.max() * .55
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, formatter(values[i, j]), ha="center", va="center",
                            color="white" if values[i, j] > threshold else "#172554", weight="bold")
        fig.suptitle(f"Seed-2 D4-ORQB • test accuracy {accuracy:.4%}", fontsize=14, weight="bold")
        fig.tight_layout()
        save_paper_figure(fig, "fig06_official_test_confusion")
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for cls, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
            fpr, tpr, auc = roc_data[cls]
            axes[0].plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC {auc:.6f})")
            recall, precision = binary_pr(test_labels == cls, test_probabilities[:, cls])
            axes[1].plot(recall, precision, color=color, lw=2.2, label=name)
        axes[0].plot([0, 1], [0, 1], "--", color="#94A3B8", lw=1)
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC curves", xlim=(0, .08), ylim=(.90, 1.002))
        axes[1].set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision–recall curves", xlim=(.90, 1.002), ylim=(.90, 1.002))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(loc="lower left", fontsize=8.5)
        fig.suptitle(f"Official test • macro AUC {macro_auc:.6f}", fontsize=14, weight="bold")
        fig.tight_layout()
        save_paper_figure(fig, "fig07_official_test_roc_pr")
        """
    ),
    code(
        r"""
        class_auc = np.array([item[2] for item in roc_data])
        class_support = confusion.sum(axis=1)
        class_markdown = [
            "| Class | Precision | Recall | F1 | AUC | Support |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {CLASS_NAMES[i]} | {precisions[i]:.4%} | {recalls[i]:.4%} | "
                f"{f1s[i]:.4%} | {class_auc[i]:.6f} | {class_support[i]:,} |"
                for i in range(3)
            ],
        ]
        display(Markdown("\n".join(class_markdown)))

        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        metric_names = ["Precision", "Recall", "F1", "AUC"]
        x = np.arange(3)
        width = .19
        for offset, metric, color in zip(np.arange(4) - 1.5, metric_names, ["#457B9D", "#E76F51", "#2A9D8F", "#8E6BBE"]):
            values = {"Precision": precisions, "Recall": recalls, "F1": f1s, "AUC": class_auc}[metric]
            ax.bar(x + offset*width, values, width, label=metric, color=color)
        ax.set_xticks(x, CLASS_NAMES)
        ax.set_ylim(.94, 1.002)
        ax.set_ylabel("Score")
        ax.set_title("Official-test per-class performance")
        ax.grid(axis="y", alpha=.2)
        ax.legend(ncol=4, loc="lower right")
        fig.tight_layout()
        save_paper_figure(fig, "fig08_official_test_per_class")
        """
    ),
    code(
        r"""
        confidence = test_probabilities.max(axis=1)
        correct = test_predictions == test_labels
        edges = np.linspace(0, 1, 16)
        bin_ids = np.minimum(np.digitize(confidence, edges[1:-1], right=True), 14)
        bin_confidence, bin_accuracy, bin_count = [], [], []
        for bin_id in range(15):
            mask = bin_ids == bin_id
            bin_count.append(int(mask.sum()))
            bin_confidence.append(float(confidence[mask].mean()) if mask.any() else np.nan)
            bin_accuracy.append(float(correct[mask].mean()) if mask.any() else np.nan)
        ece = sum((count / len(correct)) * abs(acc - conf)
                  for count, acc, conf in zip(bin_count, bin_accuracy, bin_confidence) if count)
        assert math.isclose(ece, sealed_metrics["ece_15"], abs_tol=2e-12)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        valid = np.asarray(bin_count) > 0
        axes[0].plot([0, 1], [0, 1], "--", color="#94A3B8", label="Perfect calibration")
        axes[0].plot(np.asarray(bin_confidence)[valid], np.asarray(bin_accuracy)[valid], "o-", color="#6A5ACD", lw=2, label=f"ECE-15 = {ece:.4f}")
        axes[0].set(xlabel="Mean confidence", ylabel="Empirical accuracy", title="Reliability diagram", xlim=(.5, 1.005), ylim=(.5, 1.005))
        axes[0].legend(loc="lower right")
        bins = np.linspace(.45, 1, 35)
        axes[1].hist(confidence[correct], bins=bins, alpha=.72, color="#2A9D8F", label=f"Correct ({correct.sum():,})")
        axes[1].hist(confidence[~correct], bins=bins, alpha=.78, color="#E45756", label=f"Errors ({(~correct).sum():,})")
        axes[1].set_yscale("log")
        axes[1].set(xlabel="Maximum predicted probability", ylabel="Images (log scale)", title="Prediction confidence")
        axes[1].legend()
        fig.suptitle(f"Official-test calibration • NLL {nll:.5f} • Brier {brier:.5f}", fontsize=13, weight="bold")
        fig.tight_layout()
        save_paper_figure(fig, "fig09_official_test_calibration")
        """
    ),
    code(
        r"""
        epochs = np.array([row["epoch"] for row in training_history])
        train_accuracy = np.array([row["train_accuracy"] for row in training_history])
        val_accuracy = np.array([row["validation"]["accuracy"] for row in training_history])
        train_loss = np.array([row["train_loss"] for row in training_history])
        val_nll = np.array([row["validation"]["nll"] for row in training_history])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].plot(epochs, 100*train_accuracy, color="#457B9D", lw=2, label="Training")
        axes[0].plot(epochs, 100*val_accuracy, color="#6A5ACD", lw=2, label="Validation")
        axes[0].axvline(19, color="#334155", ls="--", lw=1, label="Selected epoch 19")
        axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Optimization trajectory")
        axes[0].legend()
        axes[1].plot(epochs, train_loss, color="#E76F51", lw=2, label="Training loss")
        axes[1].plot(epochs, val_nll, color="#2A9D8F", lw=2, label="Validation NLL")
        axes[1].axvline(19, color="#334155", ls="--", lw=1)
        axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss trajectory")
        axes[1].legend()
        for ax in axes: ax.grid(alpha=.2)
        fig.tight_layout()
        save_paper_figure(fig, "fig10_training_history")
        """
    ),
    code(
        r"""
        validation_per_class = validation_metrics["per_class"]
        test_per_class = sealed_metrics["per_class"]
        validation_recalls = np.array([validation_per_class[key]["recall"] for key in ("axion", "cdm", "no_sub")])
        test_recalls = np.array([test_per_class[key]["recall"] for key in ("axion", "cdm", "no_sub")])

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        aggregate_names = ["Accuracy", "Macro F1", "Macro AUC"]
        validation_aggregate = [validation_metrics["accuracy"], validation_metrics["macro_f1"], validation_metrics["macro_auc_ovr"]]
        test_aggregate = [sealed_metrics["accuracy"], sealed_metrics["macro_f1"], sealed_metrics["macro_auc_ovr"]]
        x = np.arange(3)
        axes[0].bar(x-.18, validation_aggregate, .36, label="Validation", color="#457B9D")
        axes[0].bar(x+.18, test_aggregate, .36, label="Official test", color="#8E6BBE")
        axes[0].set_xticks(x, aggregate_names)
        axes[0].set_ylim(.96, 1.002)
        axes[0].set_title("Aggregate generalization")
        axes[0].legend()
        axes[1].bar(x-.18, validation_recalls, .36, label="Validation", color="#457B9D")
        axes[1].bar(x+.18, test_recalls, .36, label="Official test", color="#8E6BBE")
        axes[1].set_xticks(x, CLASS_NAMES)
        axes[1].set_ylim(.94, 1.002)
        axes[1].set_title("Per-class recall generalization")
        axes[1].legend()
        for ax in axes:
            ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        save_paper_figure(fig, "fig11_validation_test_generalization")
        """
    ),
    md(
        r"""
        ## Paper-ready result summary

        The selected model contains **245,221 trainable parameters**, including an **88-parameter quantum circuit** organized as four reusable eight-qubit heads with two data reuploads. Its learned orbit construction and tied circuit operations produce 48 D4-invariant quantum expectation features.

        At the selected checkpoint (epoch 19), training accuracy was **99.9586%** and development-validation accuracy was **98.4746%**. The prospectively sealed official-test pass classified **14,755 of 15,000 images correctly**, corresponding to **98.3667% accuracy** with a Wilson 95% confidence interval of **98.1511–98.5575%**. Official-test macro AUC was **0.998369**, macro F1 was **0.983650**, NLL was **0.057747**, Brier score was **0.025378**, and ECE-15 was **0.008445**.

        The remaining errors are concentrated in the physically difficult axion/CDM distinction. No-substructure recall is **99.98%**, while axion and CDM recall are **97.94%** and **97.18%**, respectively.

        ### Statements appropriate for the paper

        - The architecture is structurally invariant to D4 rotations and reflections through orbit lifting, shared processing, symmetry-tied circuit operations, and invariant observable reduction.
        - The quantum bottleneck is trained: its saved update norm is nonzero and the learned 88 circuit parameters are displayed above.
        - The official-test result comes from one frozen checkpoint and one sealed test pass; test data were not used for checkpoint selection.
        - The circuit result is based on ideal statevector simulation with analytic expectation values. It is not a claim of execution on noisy quantum hardware.
        - Seed 2 is a fixed-backbone core replication initialized from the selected development backbone; it should not be described as a fully independent end-to-end training replication.

        All PNG and vector PDF figures produced above are saved under the printed paper-figure directory. The same figures remain embedded in this executed notebook.
        """
    ),
    code(
        r"""
        expected_stems = [f"fig{i:02d}" for i in range(1, 12)]
        generated_png = sorted(FIGURE_DIR.glob("fig*.png"))
        generated_pdf = sorted(FIGURE_DIR.glob("fig*.pdf"))
        assert len(generated_png) == 11 and len(generated_pdf) == 11
        final_audit = {
            "model": "D4-ORQB seed 2",
            "checkpoint_epoch": checkpoint_payload["epoch"],
            "trainable_parameters": model.parameter_count,
            "quantum_parameters": parameter_breakdown["Quantum circuit"],
            "validation_accuracy": validation_metrics["accuracy"],
            "official_test_accuracy": accuracy,
            "official_test_macro_auc": macro_auc,
            "official_test_macro_f1": macro_f1,
            "official_test_samples": len(test_labels),
            "artifact_hashes_verified": True,
            "live_inference_passed": True,
            "D4_invariance_max_logit_difference": max_logit_difference,
            "paper_png_figures": len(generated_png),
            "paper_pdf_figures": len(generated_pdf),
            "status": "PASS",
        }
        print(json.dumps(final_audit, indent=2))
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "title": "Best D4-ORQB Quantum Model",
    },
)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
