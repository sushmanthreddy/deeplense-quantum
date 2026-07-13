# Model I seed-2 40-epoch complete artifact bundle

This bundle collects the code, weights, notebook, execution records, and full
saved results for the fresh Model I seed-2 D4-ORQB 40-epoch experiment run on
2026-07-13.

## Scientific scope

- Training protocol: fresh seed-2 run, 40 epochs, development split only
- Best checkpoint: epoch 40
- Validation samples: 17,504
- Validation accuracy: 98.6003199%
- Macro ROC-AUC: 99.8025465%
- Official test evaluated for this 40-epoch follow-up: no
- Training source commit: `f10aad3a92b51dd3af0cc7ecf89288cb53001269`

The root notebook is included with all of its saved outputs, but it remains the
earlier prospectively sealed 20-epoch/official-test record. It is not relabeled
as a 40-epoch notebook. The 40-epoch CUDA-Q verification extracted only the
tagged kernel definition from that notebook and supplied the separately saved
40-epoch weights and validation fixture.

## Contents

- `weights/`
  - `trained_40_epoch_best.pt`: selected final model, best at epoch 40
  - `trained_40_epoch_last.pt`: final training-state checkpoint
  - `initial_backbone_epoch18_best.pt`: initialization checkpoint actually used
  - `initial_backbone_epoch18_last.pt`: optional archival companion checkpoint
  - `cudaq_q2_weights.npz`: portable exported quantum/head weights
- `code/exact_execution_source/source-f10aad3/`: extracted Python modules used
  by the training job, plus the historical dependency file and source README
- `code/experiment_record/`: portable runner, three executed manifests, export
  and CUDA-Q verification scripts, documentation, and small audit reports
- `execution/`: immutable job workspace files and the original verified source
  tarball
- `notebook/Best_D4_ORQB_Quantum_Model.ipynb`: hash-matched notebook with 25
  saved outputs
- `results/training_run/`: complete raw 40-epoch output (checkpoints, log,
  configuration, split indices, predictions, history, metrics, and provenance)
- `results/backbone_pretrain_context/`: complete initialization-run context
- `results/cudaq_validation/`: exported reference fixture and full CUDA-Q replay
  results, including the standalone extracted `kernel_source.py`
- `FILE_MANIFEST.tsv`: byte size and path for every bundled file
- `SHA256SUMS`: SHA-256 checksum for every bundled file except itself

## Key verified hashes

```text
5a7156ecc483ce215843b980c3f356ddc10ce61bc619ca0b01537619d48e4edf  trained_40_epoch_best.pt
1268a442acf35b00f6bd5522b074f90c89c351f5ea5cb08d34ce881944c1a400  trained_40_epoch_last.pt
d46c458444474f50262e9e00be0edcca72a8081e55039cc722074d10a16b7dcb  initial_backbone_epoch18_best.pt
2a4f99e880570fa747d604e5e5154f04e6551fe8282bf682d681dd2743d31e38  initial_backbone_epoch18_last.pt
55877e52c5e5413768707f4d316796296d175d2b9ef34a7f806f8adcfe3b58e3  cudaq_q2_weights.npz
a687cd1a6c52cde6a8fdfb61d6df195002e40df8a18f76082d9acc0652540dc6  source-f10aad3.tar
c520b0370bb74fecd9f09616953e7933fc5a76e74f2c43ff36e9cdc0f93afc79  Best_D4_ORQB_Quantum_Model.ipynb
```

Run `sha256sum -c SHA256SUMS` from this directory to verify the full extracted
bundle. A separate `.zip.sha256` file verifies the ZIP archive itself.

## Runtime and reproduction notes

The authoritative runtime records are the manifests under
`code/experiment_record/jobs/`:

- Training image digest:
  `sha256:971fbeae82c0a5a7a970a264a8b8ce1c3426aa79df7111004ad2bc2640f7d89c`
- CUDA-Q image digest:
  `sha256:84901e99f1d83e0abdb7b02f9870cef1fb1122f889e53af5a0af2c6b7fe3596e`
- CUDA-Q version used for replay: 0.12
- Simulator: NVIDIA cuStateVec (`nvidia` / `cusvsim_fp32`), not a physical QPU

Raw Model I datasets, processed image caches, virtual environments, `.git`, and
`__pycache__` are intentionally excluded. They are large or rebuildable and are
not needed to inspect the model, view saved results, or replay CUDA-Q using the
included validation fixture. From-scratch retraining still requires the Model I
development dataset and preprocessing cache at paths supplied by the user.
