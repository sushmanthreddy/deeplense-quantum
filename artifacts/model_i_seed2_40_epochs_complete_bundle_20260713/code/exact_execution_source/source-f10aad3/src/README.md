# Steerable-QVF Equivariant-QCNN (project package)

A packaged refactor of the notebook-style script
`../steerable_qvf_quantum_lensing.py` into an importable, runnable Python
project. The original script is intentionally left untouched (use it for
notebook / cell-by-cell exploration; use this `src/` package for code-type
project work).

## Pipeline

```
lensing image (B,1,H,W)
  -> C8 / D8 steerable CNN (e2cnn)            -> rotation-equivariant feature maps
  -> GroupPooling + global avg pool           -> rotation-INVARIANT vector
  -> encoding head:
       amplitude -> Linear -> softmax -> sqrt -> 256 real amplitudes (||.||2 = 1)
       angle     -> Linear -> tanh*pi/2       -> 8 RY angles
       reupload  -> L x 8 RY angles
  -> p4m EQUIVARIANT QCNN (U2_equiv + Pooling_ansatz_equiv, 33 params) [EQNN_for_HEP]
  -> <Z>,<X>,<Y> per wire (+ optional <Z_iZ_j>) -> shared MLP head -> logits
```

## Layout

| Module        | Responsibility                                              |
|---------------|-------------------------------------------------------------|
| `config.py`   | `Config` dataclass, dataset paths, env-var loading          |
| `data.py`     | `.npy` dataset, transforms, augmentation, dataloaders       |
| `encoder.py`  | Steerable C8/D8 CNN + QVF amplitude/angle/reupload head      |
| `quantum.py`  | p4m equivariant QCNN ported to TorchQuantum                  |
| `model.py`    | Full encoder -> QCNN -> head model + parameter summary       |
| `engine.py`   | Train / evaluate loops, checkpointing, metrics, plots       |
| `main.py`     | CLI entrypoint                                               |

The classifier head is **shared across all three encodings**
(`Linear -> ReLU -> Dropout(0.2) -> Linear`).

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
cd src
python -m steerable_qvf.main --encoding amplitude --dataset-id model_1 --epochs 50
```

Dataset paths default to the EQNN-for-HEP lensing layout and can be overridden
with flags (`--data-root`, `--test-dir`) or the same env vars as the notebook
(`DEEPLENSE_DATA_ROOT`, `DEEPLENSE_TEST_DIR`, `DEEPLENSE_DATASET_ID`, ...).

## Use as a library

```python
import torch
from steerable_qvf import Config, build_model, parameter_summary

cfg = Config(encoding="angle", num_epochs=5)
model = build_model(cfg, num_classes=3, device=torch.device("cpu"))
print(parameter_summary(model))
```
