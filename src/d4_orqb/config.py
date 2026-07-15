"""Configuration for the selected D4-ORQB training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal


DATASET_IDS = (
    "model_i",
    "model_ii",
    "model_iii",
    "model_iv",
    "model_v",
)
DatasetID = Literal[
    "model_i", "model_ii", "model_iii", "model_iv", "model_v"
]


@dataclass(slots=True)
class Config:
    """Runtime settings with no committed dataset location.

    ``stage="all"`` recreates the initialization used by the selected run:
    an 18-epoch classical context pretrain followed by a fresh 40-epoch
    quantum model initialized from the shared image backbone.
    """

    dataset_id: DatasetID = "model_i"
    development_root: str = ""
    validation_root: str = ""
    cache_root: str = ""
    output_dir: str = ""
    stage: Literal["all", "pretrain", "quantum", "audit"] = "all"
    backbone_checkpoint: str = ""
    freeze_backbone_during_quantum: bool = False
    allow_inconclusive_model_iv_audit: bool = False

    image_size: int = 96
    batch_size: int = 256
    workers: int = 4
    io_workers: int = 8
    val_fraction: float = 0.20
    split_seed: int = 42

    heads: int = 4
    reuploads: int = 2
    dropout: float = 0.10

    pretrain_epochs: int = 18
    pretrain_patience: int = 6
    pretrain_seed: int = 0
    pretrain_learning_rate: float = 4e-3
    pretrain_core_learning_rate: float = 6e-3

    quantum_epochs: int = 40
    quantum_patience: int = 41
    quantum_seed: int = 2
    encoder_learning_rate: float = 5e-4
    learning_rate: float = 3e-3
    core_learning_rate: float = 5e-3

    weight_decay: float = 1e-4
    label_smoothing: float = 0.02
    deterministic: bool = False

    def validate(self) -> None:
        if self.dataset_id not in DATASET_IDS:
            raise ValueError(
                f"Unknown dataset_id: {self.dataset_id}; "
                f"choose one of {DATASET_IDS}"
            )
        if not self.development_root.strip():
            raise ValueError(
                "Dataset path is empty. Set --development-root to the selected "
                "dataset's development directory on the training machine."
            )
        if self.dataset_id == "model_iv" and not self.validation_root.strip():
            raise ValueError(
                "Model IV requires --validation-root so its supplied "
                "development-validation split is preserved."
            )
        if self.validation_root.strip() and self.dataset_id != "model_iv":
            raise ValueError(
                "--validation-root is reserved for Model IV's supplied "
                "development-validation split. Official test evaluation is "
                "not supported by this training entry point."
            )
        if (
            self.validation_root.strip()
            and self.development_path == self.validation_path
        ):
            raise ValueError(
                "Development and validation roots must be different directories"
            )
        if self.stage != "audit" and not self.cache_root.strip():
            raise ValueError("Set --cache-root to a writable cache directory.")
        if not self.output_dir.strip():
            raise ValueError("Set --output-dir to a new run directory.")
        if self.stage not in ("all", "pretrain", "quantum", "audit"):
            raise ValueError(f"Unknown stage: {self.stage}")
        if self.stage == "audit" and self.dataset_id != "model_iv":
            raise ValueError("--stage audit is defined only for Model IV")
        if (
            self.allow_inconclusive_model_iv_audit
            and self.dataset_id != "model_iv"
        ):
            raise ValueError(
                "--allow-inconclusive-model-iv-audit applies only to Model IV"
            )
        if self.stage == "audit" and self.allow_inconclusive_model_iv_audit:
            raise ValueError(
                "The audit-only stage reports its real status and cannot be overridden"
            )
        if self.stage == "quantum" and self.backbone_checkpoint:
            checkpoint = Path(self.backbone_checkpoint).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        if self.freeze_backbone_during_quantum:
            if self.stage not in ("all", "quantum"):
                raise ValueError(
                    "--freeze-backbone-during-quantum is only valid when a "
                    "quantum stage is requested"
                )
            if self.stage == "quantum" and not self.backbone_checkpoint:
                raise ValueError(
                    "--freeze-backbone-during-quantum with --stage quantum "
                    "requires --backbone-checkpoint"
                )
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError("image_size and batch_size must be positive")
        if self.workers < 0 or self.io_workers <= 0:
            raise ValueError("workers must be nonnegative and io_workers positive")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be between zero and one")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.heads != 4 or self.reuploads != 2:
            raise ValueError(
                "The selected D4-ORQB circuit uses 4 heads and 2 reuploads"
            )
        for name in (
            "pretrain_epochs",
            "pretrain_patience",
            "quantum_epochs",
            "quantum_patience",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "pretrain_learning_rate",
            "pretrain_core_learning_rate",
            "encoder_learning_rate",
            "learning_rate",
            "core_learning_rate",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")

    @property
    def development_path(self) -> Path:
        return Path(self.development_root).expanduser().resolve()

    @property
    def validation_path(self) -> Path | None:
        if not self.validation_root.strip():
            return None
        return Path(self.validation_root).expanduser().resolve()

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_root).expanduser().resolve()

    @property
    def cache_key(self) -> str:
        """Return a stable dataset- and resize-specific cache key."""

        return f"{self.dataset_id}_{self.image_size}"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
