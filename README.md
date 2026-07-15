# D4 Orbit-Reuploading Quantum Bottleneck

This repository contains a clean, result-free implementation of the D4 Orbit-Reuploading Quantum Bottleneck (D4-ORQB) for three-class DeepLense Models I–V. The code is organized like the `main` branch: one small Python package under `src/` and one thin runnable notebook per dataset under `notebooks/`. The quantum stage runs through TorchQuantum's differentiable batched statevector device.

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

Keep dataset paths empty in committed code and notebooks. Models I–III and V use a development directory with this layout:

```text
<development-root>/
├── axion/*.npy
├── cdm/*.npy
└── no_sub/*.npy
```

Each `.npy` file must contain a two-dimensional lens image in a format accepted by the loader.

Model IV uses its supplied split without carving another holdout:

```text
<model-iv-root>/
├── train/{axion,cdm,no_sub}/*.npy
└── val/{axion,cdm,no_sub}/*.npy
```

Pass `train/` as the development root and `val/` as the validation root. The notebooks do not accept or inspect an official test path.

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

Each notebook leaves the development, optional Model-IV validation, cache, run, and results roots blank. It rejects reused run and result directories, invokes `--stage all --dataset-id ... --quantum-epochs 50` with explicit learning rates, and never opens an official test set. Model IV automatically runs the preflight gate first. The equivalent Model-I command starts as follows:

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

After a successful run, the notebook validates and prints the generated checkpoint candidate at:

```text
<run-root>/<dataset-id>/<run-name>/quantum_seed2_50ep/best.pt
```

It does not copy that checkpoint into `weights/`. Review each completed run and its validation evidence first, then explicitly select and document any checkpoint that should be retained under `weights/`.

The notebook automatically copies only the development-validation presentation artifacts into fresh paths shaped as:

```text
results/<dataset-id>/<run-name>/metrics.md
results/<dataset-id>/<run-name>/roc_curve.png
```

These metrics and curves are development-validation results, not official-test results.

## Future GPU runner

`AGENTS.md` instructs future coding agents to create an ignored root `runner.yaml` only when a run is requested. It also defines the safety checks for a one-shot Kubeflow `PyTorchJob`. YAML/YML, JSON, checkpoints, caches, logs, and generated outputs remain ignored and must not be committed.
