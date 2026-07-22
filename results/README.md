# Final notebook images

This directory contains only final evaluation images produced by the executed
quantum notebooks under `notebooks/d4_orqb/`. Files are grouped by model and by
the notebook that produced them.

Each image directory contains:

- `test_roc_curve.png`
- `test_confusion_matrix.png`

Models I–III use separate official test datasets. Models IV–V use their fixed
held-out test splits, as documented in `RESULTS.md`.

| Notebook | Image directory | Final images |
| --- | --- | ---: |
| `train_model_i.ipynb` | `model_1/train_model_i/images/` | 2 |
| `train_model_ii.ipynb` | `model_2/train_model_ii/images/` | 2 |
| `train_model_iii.ipynb` | `model_3/train_model_iii/images/` | 2 |
| `train_model_iv.ipynb` | `model_4/train_model_iv/images/` | 2 |
| `train_model_v.ipynb` | `model_5/train_model_v/images/` | 2 |

The classical notebooks did not calculate ROC-AUC or produce PNG files, so no
classical ROC images are included. Their final scalar metrics and confusion
matrix values remain saved in the executed notebooks.
