# Selected Evaluation Figures

The selected figure set is organized by model and evaluation stage:

```text
model_N/
├── pretraining/validation_roc_curve.png
├── quantum/validation_roc_curve.png
└── test/
    ├── confusion_matrix.png
    └── roc_curve.png
```

The `pretraining` and `quantum` figures describe development-validation
performance. The `test` figures describe the one-time final evaluation. Models
I–III use separate official test datasets; Models IV–V use carved 15% holdouts.

See [`RESULTS.md`](../../RESULTS.md) for metrics, qualifications, and links to
the selected checkpoints.
