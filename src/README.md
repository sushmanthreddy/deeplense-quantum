# Python package

`d4_orqb` is the importable D4 Orbit-Reuploading Quantum Bottleneck implementation. It intentionally mirrors the compact module layout on the `main` branch.

| Module | Responsibility |
| --- | --- |
| `config.py` | Validated defaults for the pretraining and 40-epoch quantum stages |
| `data.py` | DeepLense `.npy` discovery, image extraction, caching, deterministic splits, and loaders |
| `encoder.py` | Deterministic morphology channels, eight-view D4 lifting, and the shared MBConv encoder |
| `quantum.py` | TorchQuantum D4 circuit, tied orbit interactions, and invariant observables |
| `model.py` | Classical pretraining head, quantum bottleneck integration, and final classifier |
| `engine.py` | Training, validation, metrics, checkpoint handoff, and run-output handling |
| `main.py` | Command-line entry point and two-stage orchestration |
| `__init__.py` | Small public package interface |

LensPINN was a separate experimental baseline and is deliberately absent. The mixed-derivative morphology channel in `encoder.py` is deterministic feature engineering only. D4 equivariance is implemented by lifting each input to all eight group views, applying one shared encoder, and using tied TorchQuantum orbit operations and invariant reductions. CUDA-Q is not the training backend; the removed CUDA-Q path was post-training validation without PyTorch autograd.

From the repository root:

```bash
python -m pip install -r src/requirements.txt
PYTHONPATH=src python -m d4_orqb.main --help
```

For a fresh end-to-end run, use `--stage all` and provide nonempty development, cache, and output paths. Dataset paths are intentionally absent from the package.
