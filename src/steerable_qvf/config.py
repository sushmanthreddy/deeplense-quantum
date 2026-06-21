"""Configuration for the Steerable-QVF Equivariant-QCNN pipeline.

All hyperparameters live in the :class:`Config` dataclass so they can be
constructed explicitly (for scripts / tests) or built from environment
variables via :meth:`Config.from_env` (mirrors the original notebook).

Security note (codeguard-1-hardcoded-credentials): this module contains NO
secrets, API keys, tokens, or credentials. The dataset paths below are
filesystem locations, not credentials, and are overridable via env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


# Same dataset layout used by the EQNN-for-HEP lensing notebooks.
DATASET_ROOTS = {
    "model_1": "/home/jovyan/ssh-test-datavol-1/dataset/Model_I",
    "model_2": "/home/jovyan/ssh-test-datavol-1/dataset/Model_II",
    "model_3": "/home/jovyan/ssh-test-datavol-1/dataset/Model_III",
    "model_4": "/home/jovyan/ssh-test-datavol-1/dataset/Model_IV",
}
TEST_ROOTS = {
    "model_1": "/home/jovyan/ssh-test-datavol-1/dataset/Model_I_test",
    "model_2": "/home/jovyan/ssh-test-datavol-1/dataset/Model_II_test",
    "model_3": "/home/jovyan/ssh-test-datavol-1/dataset/Model_III_test",
    "model_4": "/home/jovyan/ssh-test-datavol-1/dataset/Model_IV_test",
}
VALID_DATASET_IDS = set(DATASET_ROOTS)

VALID_ENCODINGS = ("amplitude", "angle", "reupload")


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    slug = "".join(ch if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in slug.split("_") if part) or "model_1"


@dataclass
class Config:
    """Typed hyperparameters for the whole pipeline."""

    # ---- Quantum / encoding ----
    n_qubits: int = 8
    # "amplitude" | "angle" | "reupload"
    encoding: str = "amplitude"
    reupload_layers: int = 2

    # ---- Steerable CNN ----
    group_n: int = 8
    use_reflections: bool = False
    base_width: int = 8
    softmax_temperature: float = 1.0
    encoder_dropout: float = 0.3

    # ---- Quantum readout ----
    readout_paulis: tuple = ("Z", "X", "Y")
    readout_zz: bool = True

    # ---- Hybrid residual ----
    hybrid_residual: bool = True
    residual_dim: int = 16

    # ---- Augmentation (train only) ----
    augment: bool = True
    aug_max_rotation: float = 15.0
    aug_max_translate: float = 0.1

    # ---- Training ----
    load_img_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 50
    patience: int = 12

    # ---- Data ----
    in_channels: int = 1
    num_classes: int = 3
    notebook_name: str = "steerable_qvf_quantum"
    dataset_id: str = "model_1"
    val_split: float = 0.20

    # ---- Paths (resolved in __post_init__) ----
    data_root: str = ""
    test_dir: str = ""
    run_dir: Path = field(default=None)

    # ---- Performance knobs ----
    cache_data_in_memory: bool = True
    num_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 0))
    prefetch_factor: int = 4
    seed: int = 42

    @property
    def amp_dim(self) -> int:
        return 2 ** self.n_qubits

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"best_{self.notebook_name}_{self.dataset_id}.pth"

    @property
    def group_name(self) -> str:
        return f"{'D' if self.use_reflections else 'C'}{self.group_n}"

    def __post_init__(self):
        self.dataset_id = slugify(self.dataset_id)
        if self.dataset_id not in VALID_DATASET_IDS:
            raise ValueError(
                f"dataset_id must be one of {sorted(VALID_DATASET_IDS)}, got {self.dataset_id!r}"
            )
        if self.encoding not in VALID_ENCODINGS:
            raise ValueError(f"encoding must be one of {VALID_ENCODINGS}, got {self.encoding!r}")
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be between 0 and 1, got {self.val_split}")

        if not self.data_root:
            self.data_root = DATASET_ROOTS[self.dataset_id]
        if not self.test_dir:
            self.test_dir = TEST_ROOTS[self.dataset_id]
        if self.run_dir is None:
            self.run_dir = Path.cwd() / self.dataset_id
        else:
            self.run_dir = Path(self.run_dir)

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build a Config, reading the same env vars as the original notebook.

        Explicit ``overrides`` take precedence over environment variables.
        """
        dataset_id = overrides.pop("dataset_id", os.environ.get("DEEPLENSE_DATASET_ID", "model_1"))
        env = dict(
            dataset_id=dataset_id,
            data_root=os.environ.get("DEEPLENSE_DATA_ROOT", ""),
            test_dir=os.environ.get("DEEPLENSE_TEST_DIR", ""),
            val_split=float(os.environ.get("DEEPLENSE_VAL_SPLIT", "0.20")),
            cache_data_in_memory=os.environ.get("DEEPLENSE_CACHE_DATA", "1") != "0",
            num_workers=int(
                os.environ.get("DEEPLENSE_NUM_WORKERS", str(min(4, os.cpu_count() or 0)))
            ),
            prefetch_factor=int(os.environ.get("DEEPLENSE_PREFETCH_FACTOR", "4")),
        )
        env.update(overrides)
        return cls(**env)

    def with_(self, **overrides) -> "Config":
        """Return a copy with the given fields replaced."""
        return replace(self, **overrides)
