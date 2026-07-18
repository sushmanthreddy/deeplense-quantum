# Python package

`d4_orqb` is the importable D4 Orbit-Reuploading Quantum Bottleneck implementation. It intentionally mirrors the compact module layout on the `main` branch.

| Module | Responsibility |
| --- | --- |
| `config.py` | Validated dataset routing and defaults for the pretraining and 40-epoch quantum stages |
| `data.py` | Model I–V `.npy` discovery, image extraction, caching, validation splits/loaders, and the CPU-only Model-IV integrity/signal gate |
| `encoder.py` | Deterministic morphology channels, eight-view D4 lifting, and the shared MBConv encoder |
| `quantum.py` | TorchQuantum D4 circuit, tied orbit interactions, and invariant observables |
| `model.py` | Classical pretraining head, quantum bottleneck integration, and final classifier |
| `engine.py` | Training, validation metrics and ROC output, checkpoint handoff, and run-output handling |
| `main.py` | Dataset-aware command-line entry point, pre-CUDA Model-IV gate, and two-stage orchestration |
| `__init__.py` | Small public package interface |

LensPINN was a separate experimental baseline and is deliberately absent. The mixed-derivative morphology channel in `encoder.py` is deterministic feature engineering only. D4 equivariance is implemented by lifting each input to all eight group views, applying one shared encoder, and using tied TorchQuantum orbit operations and invariant reductions. CUDA-Q is not the training backend; the removed CUDA-Q path was post-training validation without PyTorch autograd.

From the repository root:

```bash
python -m pip install -r src/requirements.txt
PYTHONPATH=src python -m d4_orqb.main --help
```

For a fresh end-to-end run, use `--stage all --dataset-id model_i` (or `model_ii` through `model_v`) and provide nonempty development, cache, and output paths. Model IV additionally requires its supplied `val/` directory through `--validation-root` and automatically runs the CPU-only integrity/signal audit before TorchQuantum or CUDA. Use `--stage audit` with a fresh output directory to run that gate alone; it needs no cache or GPU. Dataset paths are intentionally absent from the package, and no official test path is accepted.

The gate reports `PASS_SIGNAL_DETECTED`, `PREPROCESSING_SIGNAL_LOSS`, `INCONCLUSIVE_NO_SIGNAL_DETECTED`, or `INTEGRITY_FAILED` in `dataset_audit.md`. Only a model-visible pass proceeds by default. The research-only `--allow-inconclusive-model-iv-audit` override keeps the inconclusive status in the report and never bypasses an integrity or preprocessing failure. This fixed low-capacity probe is an operational GPU safeguard, not a proof that an inconclusive dataset contains no possible label signal.

The CLI keeps the selected 40-epoch default and remains development-validation only. The five notebooks under `notebooks/d4_orqb/` are standalone, source-mirroring implementations that explicitly request 50 quantum epochs. They add transparent final-test routing without changing the package model: Models I–III use separate official test roots, Model IV carves 15% from its supplied `train/` while retaining supplied `val/`, and Model V uses one 65/20/15 split. Final test evaluation is explicitly gated, reloads the validation-selected `best.pt`, and never affects early stopping. Checkpoints are selected for `weights/` only after review; the notebooks never promote them automatically.
