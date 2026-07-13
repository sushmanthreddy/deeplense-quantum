# Model I seed-2 hybrid quantum experiment: 40 epochs

This folder preserves the reproducible code, executed Kubeflow manifests, and
small audit reports for the fresh 40-epoch follow-up of the selected Model I
seed-2 D4 orbit-reuploading quantum bottleneck (D4-ORQB).

## Scientific scope

- Dataset: Model I development split only for training and checkpoint selection
- Split: 70,021 training images and 17,504 validation images, split seed 42
- Training seed: 2
- Best checkpoint: epoch 40
- Validation accuracy: 98.6003199% (17,259 / 17,504)
- Macro ROC-AUC: 99.8025465%
- Macro F1: 98.6057699%
- Official test evaluated: no

This was a fresh 40-epoch run. It was not represented as an exact continuation
of the earlier 20-epoch run because the earlier checkpoints did not contain the
optimizer, learning-rate scheduler, or RNG state required for an exact resume.

The earlier validation accuracy was 98.4746344%. The new run improved it by 22
correct validation predictions (+0.1256856 percentage points), but the paired
two-sided McNemar p-value is 0.0903. The difference should therefore not be
described as statistically significant at the conventional 0.05 threshold.

The root notebook remains the record of the earlier prospectively sealed
20-epoch checkpoint and its official-test evaluation; it has not been relabeled
as a 40-epoch notebook. For this follow-up, the verifier extracts only the
audited circuit definition from that notebook and supplies the separately
exported 40-epoch weights and validation tensors whose hashes are recorded here.

## Architecture and optimization

The model has 245,221 parameters:

- Tiny classical image encoder: 242,338
- D4 orbit projection: 1,032
- Classification head: 1,763
- Quantum circuit: 88, shaped `(4 heads, 2 reuploads, 11 angles)`

The pretrained encoder was **not frozen**. It was initialized from the declared
epoch-18 backbone checkpoint and fine-tuned jointly with the projection, head,
and quantum circuit for all 40 epochs. The peak learning rates were 5e-4 for the
encoder, 3e-3 for the projection/head, and 5e-3 for the quantum circuit. The
quantum-angle update had L2 norm 2.4957418.

Training used the differentiable PyTorch statevector implementation of the
circuit. The selected weights were subsequently replayed through the literal
`@cudaq.kernel` in the root notebook using CUDA-Q 0.12 on an NVIDIA H200. The
full validation replay covered 17,504 images, 70,016 head circuits, and 840,192
invariant values. It produced zero invariant or prediction mismatches; the
maximum invariant absolute error was 5.96e-6.

CUDA-Q used the NVIDIA cuStateVec simulator (`nvidia` / `cusvsim_fp32`). This is
GPU-accelerated quantum simulation, not execution on a physical QPU, and it was
used for inference verification rather than gradient-based training.

## Folder contents

- `run_training.sh`: portable form of the exact 40-epoch trainer arguments
- `jobs/`: the three immutable Kubeflow manifests that were executed
- `scripts/export_validation_40ep.py`: exports a validation-only CUDA-Q fixture
- `scripts/verify_cudaq_validation_40ep.py`: extracts and executes the tagged
  notebook CUDA-Q kernel over the complete validation split
- `reports/training_summary.json`: selected training summary
- `reports/run_provenance.json`: source, checkpoint, and artifact hashes
- `reports/cudaq_validation_result.json`: full CUDA-Q parity report

The executed trainer source is retained in repository history at commit
`f10aad3a92b51dd3af0cc7ecf89288cb53001269` (tag
`pre-notebook-consolidation-20260712`). Checkpoints, datasets, NumPy arrays,
cached tensors, and the generated source tarball are deliberately not committed.

## Running the training command

Create a detached worktree for the exact historical trainer and expose its
`src` directory:

```bash
git worktree add --detach /tmp/d4-orqb-trainer \
  f10aad3a92b51dd3af0cc7ecf89288cb53001269
export PYTHONPATH=/tmp/d4-orqb-trainer/src
```

Then set the five required paths and run:

```bash
export DEVELOPMENT_ROOT=/path/to/Model_I
export TEST_ROOT=/path/to/Model_I_test
export CACHE_ROOT=/path/to/d4-orqb-cache
export OUTPUT_DIR=/new/output/directory
export BACKBONE_CHECKPOINT=/path/to/pretrained/best.pt
./experiments/model_i_seed2_40_epochs/run_training.sh
```

`TEST_ROOT` is required by the trainer interface, but this command does not pass
`--evaluate-test`; the official test remains closed. Use a new output directory
because the runner and orchestration layer intentionally refuse overwrites. The
manifests in `jobs/` are execution records with PVC-specific paths and completed
job names; change names and output paths before a new submission.

## Runtime and hashes

- Training image: `docker.io/pytorch/pytorch` digest
  `sha256:971fbeae82c0a5a7a970a264a8b8ce1c3426aa79df7111004ad2bc2640f7d89c`
- CUDA-Q image: `nvcr.io/nvidia/quantum/cuda-quantum` digest
  `sha256:84901e99f1d83e0abdb7b02f9870cef1fb1122f889e53af5a0af2c6b7fe3596e`
- Backbone SHA-256:
  `d46c458444474f50262e9e00be0edcca72a8081e55039cc722074d10a16b7dcb`
- Best checkpoint SHA-256:
  `5a7156ecc483ce215843b980c3f356ddc10ce61bc619ca0b01537619d48e4edf`
- Tagged notebook kernel-cell SHA-256:
  `d6ca4829bfaa2d18c80a460dd267f4dae8cd3a6da2e6713092c46c22d2cd1dc3`
