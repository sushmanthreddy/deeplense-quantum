# Selected Checkpoints

Each `model_N/best.pt` file is the validation-selected quantum-stage checkpoint
that was reloaded for the corresponding final evaluation. The pretraining and
last-epoch checkpoints remain in ignored run storage.

| Model | Checkpoint | Selected epoch | Test kind | Test samples | SHA-256 |
| --- | --- | ---: | --- | ---: | --- |
| I | [`model_1/best.pt`](model_1/best.pt) | 48 | Official | 15,000 | `44dd6e2244a585bf25d7d20695a73dd8c7583aa94d75802b59683f574f6cdd29` |
| II | [`model_2/best.pt`](model_2/best.pt) | 38 | Official | 15,000 | `ccb3f4666858ae4eb4c0b84841fb60590bdbde1370852f15da695e067cc6cdb5` |
| III | [`model_3/best.pt`](model_3/best.pt) | 15 | Official | 15,000 | `64202aaa7c7b050540b08b8c013f7d2d94b8acedaaf364dca7344ba3ed89a77c` |
| IV | [`model_4/best.pt`](model_4/best.pt) | 27 | Carved holdout | 8,205 | `724e2e4d8b54dafcb7ceb8e231dc72bf033b134c93ceb5d5fc84b6b41dca9444` |
| V | [`model_5/best.pt`](model_5/best.pt) | 7 | Carved holdout | 6,882 | `df4ce736a9fa9108635a56d11c389d5e7bf90d949255214f9ecf4b39407aef51` |

Load each checkpoint only with its matching notebook/package architecture and
the exact TorchQuantum commit pinned in `src/requirements.txt`.
