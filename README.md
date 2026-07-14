# D4 Orbit-Reuploading Quantum Bottleneck

This repository contains a clean, result-free implementation of the D4 Orbit-Reuploading Quantum Bottleneck (D4-ORQB) for three-class DeepLense Model I classification. The code is organized like the `main` branch: one small Python package under `src/` and one runnable notebook under `notebooks/`. The quantum stage runs through TorchQuantum's differentiable batched statevector device.

No dataset, cache, generated figure, metric report, run manifest, or checkpoint is committed. Supply fresh paths at runtime and train on the GPU machine.

## Layout

```text
.
├── AGENTS.md
├── README.md
├── notebooks/
│   └── d4_orqb/
│       └── train_model_i.ipynb
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

`assets/` and `weights/` are reserved for deliberately selected, documented artifacts; they are empty in this result-free version. Training writes to a user-supplied output directory, which stays outside version control.

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

Keep dataset paths empty in committed code and notebooks. At runtime, provide a development directory with this layout:

```text
<development-root>/
├── axion/*.npy
├── cdm/*.npy
└── no_sub/*.npy
```

Each `.npy` file must contain a two-dimensional lens image in a format accepted by the loader.

## Fresh two-stage run

The selected workflow trains from scratch in two stages:

1. Use the fixed morphology bank while pretraining the shared D4-view MBConv encoder and orbit projection with the classical context head.
2. Initialize those shared components in the TorchQuantum model and run the 40-epoch Model I training stage.

Use [train_model_i.ipynb](notebooks/d4_orqb/train_model_i.ipynb), or run the equivalent command from the repository root:

```bash
export DEVELOPMENT_ROOT=""
export CACHE_ROOT=""
export OUTPUT_DIR=""

PYTHONPATH=src python -m d4_orqb.main \
  --development-root "$DEVELOPMENT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stage all
```

Fill all three variables only on the machine that owns the data and storage. The command refuses an empty required path and refuses to reuse an existing output directory. The default `all` workflow owns the handoff from the pretraining checkpoint to the quantum stage; no committed weight file is required.

This is a development-training workflow. Do not open an official test set during model selection.

## Future GPU runner

`AGENTS.md` instructs future coding agents to create an ignored root `runner.yaml` only when a run is requested. It also defines the safety checks for a one-shot Kubeflow `PyTorchJob`. YAML/YML, JSON, checkpoints, caches, logs, and generated outputs remain ignored and must not be committed.
