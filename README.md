# D4 Orbit-Reuploading Quantum Bottleneck

This repository contains a clean, result-free implementation of the D4 Orbit-Reuploading Quantum Bottleneck (D4-ORQB) for three-class DeepLense Models I–V. The code is organized like the `main` branch: one small Python package under `src/` and one complete, standalone implementation notebook per dataset under `notebooks/`. The quantum stage runs through TorchQuantum's differentiable batched statevector device.

No dataset, cache, generated figure, metric report, run manifest, or checkpoint is committed. Supply fresh paths at runtime and train on the GPU machine.

## Layout

```text
.
├── AGENTS.md
├── README.md
├── notebooks/
│   └── d4_orqb/
│       ├── train_model_i.ipynb
│       ├── train_model_ii.ipynb
│       ├── train_model_iii.ipynb
│       ├── train_model_iv.ipynb
│       └── train_model_v.ipynb
└── src/
    ├── README.md
    ├── requirements.txt
    └── d4_orqb/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── encoder.py
        ├── engine.py
        ├── main.py
        ├── model.py
        └── quantum.py
```

`assets/` and `weights/` are reserved for deliberately selected, documented artifacts; they are empty in this result-free version. Fresh run output, curated validation results, and generated checkpoints remain ignored by Git.

## Why LensPINN is not here

LensPINN was an independent comparison baseline in the research workspace. It is not part of the selected D4-ORQB model, is not called by its training path, and has therefore been removed.

The selected encoder still receives eight deterministic morphology channels. The eighth channel is a parameter-free mixed spatial derivative. It is ordinary feature engineering—not a PINN, not a learned PDE solver, and not the LensPINN baseline.

The equivariance in this repository comes from explicit D4 orbit lifting: every image is transformed into all eight rotations/reflections, the same MBConv encoder processes every view, and tied TorchQuantum orbit operations plus invariant reductions preserve the D4 structure. This differs from the `main` implementation, which uses `e2cnn` steerable layers inside the encoder. Both target the same symmetry, but they encode it in different ways.

## Quantum runtime

The selected four-head, two-reupload circuit is implemented with TorchQuantum. It retains 88 trainable circuit parameters, eight D4-indexed qubits, tied R/S-edge interactions, and 48 invariant outputs. TorchQuantum remains a classical statevector simulator when it runs on a CUDA GPU; it makes the gate model and autograd integration explicit, but it is not quantum hardware.

The historical CUDA-Q component was a separate post-training parity backend. It crossed through NumPy, had no gradient path back to the encoder, and did not implement the selected two-reupload training circuit. It is therefore not used for end-to-end training here.

## Environment

Use Python 3.10 or newer. Install a CUDA-enabled PyTorch build appropriate for the GPU machine, then install the remaining dependencies:

```bash
python -m pip install -r src/requirements.txt
```

TorchQuantum is pinned to the tested official Git commit recorded in `src/requirements.txt`; do not replace it with the outdated unpinned PyPI release.

Training requires a CUDA-capable GPU. A local full training run is not expected during repository verification.

## Dataset

Keep dataset paths empty in committed code and notebooks. Models I–III use separate development and official-test directories with the same class-folder layout:

```text
<development-root>/
├── axion/*.npy
├── cdm/*.npy
└── no_sub/*.npy

<official-test-root>/
├── axion/*.npy
├── cdm/*.npy
└── no_sub/*.npy
```

Their notebooks carve a fixed, stratified 20% development-validation split from the development root. The official test root is not opened by training or checkpoint selection; its final evaluation requires the explicit notebook confirmation flag.

Model IV uses its supplied validation split and has no official test directory:

```text
<model-iv-root>/
├── train/{axion,cdm,no_sub}/*.npy
└── val/{axion,cdm,no_sub}/*.npy
```

Pass `train/` as the development root and `val/` as the validation root. Its notebook fixes a stratified 15% test holdout from `train/`, trains on the remaining 85%, and keeps all of supplied `val/` for development validation. The Model-IV signal audit receives a training-only symlink view, so the carved test samples are excluded from the audit as well as training and selection.

Model V has one combined class-folder root and no supplied validation or test directory. Its notebook performs one fixed, stratified 65/20/15 train/validation/test split. Test is selected first and validation second from each class's original population; the partitions are persisted and checked for index and model-visible-content overlap.

The carved Model-IV/V partitions are file-level because the supplied layouts expose no source/pair grouping manifest. If such provenance becomes available, replace the file-level carve with a source- or pair-grouped split before treating the holdout as benchmark evidence.

Each `.npy` file must contain a two-dimensional lens image in a format accepted by the loader. The package CLI remains development-validation only; the supplied/carved test policies above are explicit notebook extensions for final reporting.

### Model IV preflight gate

Model IV is not treated as equivalent to Model II. The public generators use different source and dark-matter models, and the selected public Model-II renderer accidentally saves a doubled noiseless model instead of its computed noisy observation. The supplied Model-IV archive also cannot be reproduced by any public generator revision: its shapes, object payload, variable axion mass, counts, and split differ from the public code.

Before any Model-IV CUDA or TorchQuantum work, the CLI now performs a CPU-only full integrity scan and a frozen, pixel-only D4-invariant physics probe on both raw and exact model-visible images. It writes `dataset_audit.md` in the fresh output directory and blocks training unless held-out development-validation signal passes the predeclared accuracy, AUC, bootstrap, per-class, and max-T permutation criteria. Run the same gate without a GPU using:

```bash
PYTHONPATH=src python -m d4_orqb.main \
  --dataset-id model_iv \
  --development-root "<model-iv-train>" \
  --validation-root "<model-iv-validation>" \
  --output-dir "<fresh-audit-output>" \
  --stage audit
```

`INCONCLUSIVE_NO_SIGNAL_DETECTED` means this fixed audit did not demonstrate learnable signal; it does not assert that the Bayes-optimal signal is zero. An explicit `--allow-inconclusive-model-iv-audit` flag exists for research-only training, but it never changes the recorded status and cannot bypass integrity failure or detected preprocessing loss. See [the Model IV data-generation contract](MODEL_IV_DATA_CONTRACT.md) for the paired counterfactual repair and proof-of-signal ladder. The official test remains closed.

## Fresh two-stage run

The selected workflow trains from scratch in two stages:

1. Use the fixed morphology bank while pretraining the shared D4-view MBConv encoder and orbit projection with the classical context head.
2. Initialize those shared components in the TorchQuantum model and run the selected quantum stage.

The package default remains the documented 40-epoch Model-I workflow. The five dataset notebooks explicitly request 50 quantum epochs for the requested Models I–V run family:

- [Model I](notebooks/d4_orqb/train_model_i.ipynb)
- [Model II](notebooks/d4_orqb/train_model_ii.ipynb)
- [Model III](notebooks/d4_orqb/train_model_iii.ipynb)
- [Model IV](notebooks/d4_orqb/train_model_iv.ipynb)
- [Model V](notebooks/d4_orqb/train_model_v.ipynb)

Each notebook embeds the complete configuration, data pipeline, Model-IV audit, encoder, TorchQuantum circuit, classifier, metrics, checkpointing, symmetry checks, and training engine. It calls those definitions directly rather than launching the package CLI. Development, validation, test, cache, and output paths remain blank; reused output directories are rejected. Model IV automatically fixes its split and runs the training-only preflight gate before TorchQuantum or CUDA initialization.

The equivalent package-level Model-I development-validation command starts as follows:

```bash
export DEVELOPMENT_ROOT=""
export CACHE_ROOT=""
export OUTPUT_DIR=""

PYTHONPATH=src python -m d4_orqb.main \
  --dataset-id model_i \
  --development-root "$DEVELOPMENT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stage all \
  --quantum-epochs 50
```

Fill runtime paths only on the machine that owns the data and storage. The command refuses an empty required path and refuses to reuse an existing output directory. The `all` workflow owns the handoff from the pretraining checkpoint to the quantum stage; no committed weight file is required.

After a successful notebook run, the validation-selected checkpoint candidate is:

```text
<output-dir>/quantum_seed2_50ep/best.pt
```

It does not copy that checkpoint into `weights/`. Review each completed run and its validation evidence first, then explicitly select and document any checkpoint that should be retained under `weights/`.

The quantum stage writes development-validation artifacts beside that checkpoint:

```text
<output-dir>/quantum_seed2_50ep/validation_metrics.md
<output-dir>/quantum_seed2_50ep/validation_roc_curve.png
```

These are development-validation results, not test results. Final held-out evaluation is skipped by default. After choices are frozen, set `CONFIRM_FINAL_TEST_EVALUATION = True` and run the final cell once; it reloads `best.pt` and writes metrics, predictions, ROC, and confusion-matrix artifacts under `<output-dir>/final_test/`. After a kernel restart, set `FINAL_TEST_ONLY = True` as well to reopen the completed output and skip training. Models I–III label this as official test evaluation. Models IV–V label it as a carved development holdout, never as official test evidence.

## Future GPU runner

`AGENTS.md` instructs future coding agents to create an ignored root `runner.yaml` only when a run is requested. It also defines the safety checks for a one-shot Kubeflow `PyTorchJob`. YAML/YML, JSON, checkpoints, caches, logs, and generated outputs remain ignored and must not be committed.
