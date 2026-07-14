# Repository instructions

## Scope and layout

- Keep the importable package limited to these eight files under `src/d4_orqb/`: `__init__.py`, `config.py`, `data.py`, `encoder.py`, `engine.py`, `main.py`, `model.py`, and `quantum.py`.
- Keep notebook implementations under `notebooks/`; `notebooks/d4_orqb/train_model_i.ipynb` is the canonical notebook.
- Keep deliberately selected figures and static visual assets under `assets/` only.
- Keep only explicitly selected and documented model checkpoints under `weights/`. Generated checkpoints stay in ignored run output and are never promoted automatically.
- Keep repository documentation in Markdown.
- Do not commit datasets, caches, logs, generated figures or outputs, archives, temporary scripts, YAML/YML, JSON, NumPy arrays, or generated checkpoints.
- Do not copy the local nested `DeepLense/` or `QMLHEP/` repositories into this repository.
- Dataset paths must remain empty in committed code and notebooks. Accept them at runtime through CLI arguments or environment variables.
- Never commit machine-specific paths, cluster PVC paths, namespaces, credentials, tokens, or secrets.
- Use a fresh output directory for every run. Never overwrite or resume into a completed run directory.

The canonical training entry point is:

```bash
PYTHONPATH=src python -m d4_orqb.main --stage all
```

Inspect `python -m d4_orqb.main --help` before adding flags. The default `all` workflow performs classical-context backbone pretraining followed by the documented 40-epoch Model I quantum stage. It must pass the newly selected pretraining checkpoint to the quantum stage; it must not depend on a checkpoint committed to the repository.

LensPINN is not part of this package. It was a separate comparison baseline and must not be reintroduced into the D4-ORQB training path. The retained mixed-derivative morphology channel is deterministic feature engineering, not a PINN. The model's equivariance comes from explicit eight-view D4 lifting, a shared MBConv encoder, tied TorchQuantum orbit operations, and invariant reductions.

The selected quantum training backend is the exact TorchQuantum commit pinned in `src/requirements.txt`. Preserve the four heads, two reuploads, 88 trainable circuit parameters, D4 edge tying, and 48 invariant outputs. Do not substitute the different pooling QCNN from `main`. Do not reintroduce the historical CUDA-Q parity backend as if it were differentiable training; a future CUDA-Q training implementation requires explicit encoder-input and circuit-parameter gradient parity.

## Required runner YAML workflow

Do not add a runner manifest preemptively. When the user asks to prepare, schedule, or execute a training/evaluation run:

1. Create `runner.yaml` at the repository root. It is intentionally ignored and must never be force-added to Git.
2. Use a fresh Kubeflow `PyTorchJob` for a one-shot GPU run unless the user names another platform. A `PyTorchJob` does not contain a cron schedule.
3. Leave dataset, namespace, PVC, and other installation-specific values as empty strings until the user or runtime environment supplies them. Add fail-fast shell checks so empty required values cannot start training.
4. Do not reuse historical job names, namespaces, PVC names, paths, source hashes, or credentials. Generate a unique job name and output directory.
5. Pin the container image by digest where possible. Declare CPU, memory, GPU requests/limits, an active deadline, and `restartPolicy: Never`.
6. Ensure the image contains the pinned TorchQuantum dependency and fail fast with `python -c 'import torchquantum'` before starting the two-stage run.
7. Preserve basic hardening: disable the service-account token unless required, disallow privilege escalation, drop Linux capabilities, prefer non-root execution, and mount writable temporary storage when using a read-only root filesystem.
8. Set `PYTHONPATH` to the mounted repository's `src/` directory and invoke `python -m d4_orqb.main --stage all` with quoted environment variables.
9. Keep the official test closed unless the user explicitly requests evaluation and the current CLI supports both a nonempty test root and an explicit test-evaluation flag.
10. Validate locally with `kubectl apply --dry-run=client -f runner.yaml` when `kubectl` is available. If it is unavailable, perform a schema/tooling check appropriate to the target platform and report that limitation.
11. Submit only when the user explicitly asks to run it and after verifying the active cluster context and namespace. For Kubeflow, submit with `kubectl apply -f runner.yaml`, then monitor the job and preserve logs/results outside Git.
12. If recurring calendar scheduling is requested, use the scheduler supported by the target platform. Do not invent a cron field inside a `PyTorchJob`.
13. Never commit `runner.yaml`, generated JSON reports, output checkpoints, or other run artifacts.

At minimum, a generated runner must expose empty runtime values equivalent to:

```text
DEVELOPMENT_ROOT=""
TEST_ROOT=""
CACHE_ROOT=""
OUTPUT_DIR=""
```

`TEST_ROOT` may remain empty for development-validation training. `DEVELOPMENT_ROOT`, `CACHE_ROOT`, and `OUTPUT_DIR` must be checked before submission. The command must also reject an already existing `OUTPUT_DIR`.

## Verification

For source changes, run syntax/import checks, verify the pinned TorchQuantum import, inspect `python -m d4_orqb.main --help`, and run the smallest relevant model smoke test. Quantum backend changes require forward, input-gradient, parameter-gradient, and D4-symmetry parity checks. A full training run is not a local verification step unless a suitable CUDA GPU and dataset were explicitly provided. Document whether any metrics come from development validation or the official test set.
