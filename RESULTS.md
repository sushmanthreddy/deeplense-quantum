# Selected Model Results

This page indexes the deliberately selected evaluation evidence and final
validation-selected checkpoints for Models I–V. The tracked notebooks remain
source-only and reproducible; generated executed notebooks remain in the local,
Git-ignored `results/` export.

## Final test metrics

| Model | Test kind | Samples | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC | Selected quantum epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I | Separate official test | 15,000 | 98.55% | 98.55% | 98.55% | 99.86% | 48 |
| II | Separate official test | 15,000 | 99.87% | 99.87% | 99.87% | 99.98% | 38 |
| III | Separate official test | 15,000 | 99.98% | 99.98% | 99.98% | 100.00% | 15 |
| IV | Carved 15% holdout | 8,205 | 35.95% | 35.54% | 30.06% | 52.75% | 27 |
| V | Carved 15% holdout | 6,882 | 51.00% | 33.33% | 22.52% | 49.23% | 7 |

Models I–III used separate test roots that were unopened during training and
unused for model selection. Models IV–V had no separate test datasets, so their
reported tests are fixed class-stratified 15% file-level holdouts.

Model IV is explicitly a research-only result: its dataset signal audit was
inconclusive and the training gate was deliberately overridden. Model V
predicted only the majority `no_sub` class on the held-out split.

## Repository artifacts

- Source notebooks: [`notebooks/d4_orqb/`](notebooks/d4_orqb/)
- Selected checkpoints and hashes: [`weights/README.md`](weights/README.md)
- ROC and confusion-matrix figures: [`assets/model_evaluation/README.md`](assets/model_evaluation/README.md)

Generated histories, structured metrics, logs, caches, last-epoch checkpoints,
and executed notebooks are intentionally excluded from Git.
