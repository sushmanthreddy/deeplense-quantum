import argparse
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from d4_orqb.train import (
    seed_everything,
    training_rng_state_digests,
    validate_paired_spatial_initializer_binding,
    validate_paired_spatial_training_contract,
)
from d4_orqb.model import D4OrbitClassifier
from d4_orqb.spatial_paired_init import build_paired_initializers, file_sha256


def spatial_args(**overrides):
    values = {
        "paired_spatial_init_report": "/tmp/paired/seed-0/report.json",
        "image_size": 96,
        "encoder_variant": "micro-stat",
        "physics_variant": "base",
        "physics_summary": "moments",
        "include_context": False,
        "quantum_encoding": "angle",
        "observable_readout": "pair",
        "heads": 4,
        "reuploads": 3,
        "epochs": 40,
        "patience": 41,
        "validation_interval": 40,
        "batch_size": 256,
        "workers": 4,
        "io_workers": 8,
        "split_seed": 42,
        "max_train_per_class": 11_667,
        "train_subset_protocol": "hash-v1",
        "evaluate_test": False,
        "deterministic": True,
        "fixed_final_validation_only": True,
        "save_last_validation_predictions": True,
        "save_stochastic_trace": True,
        "encoder_learning_rate": 5e-4,
        "learning_rate": 3e-3,
        "core_learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.02,
        "dropout": 0.10,
        "photon_noise_probability": 0.5,
        "photon_count_min": 2048.0,
        "photon_count_max": 8192.0,
        "val_fraction": 0.20,
        "core": "quantum",
        "seed": 0,
        "training_rng_seed": 20_000,
        "init_full_checkpoint": "/tmp/paired/seed-0/quantum-init.pt",
        "init_backbone_checkpoint": None,
        "init_compatible_backbone_checkpoint": None,
        "tied_mean_dispersion": False,
        "haar_subtype_residual": False,
        "shared_late_refinement": False,
        "r2_entanglers": False,
        "equatorial_readout": False,
        "meridional_readout": False,
        "subtype_specialist": False,
        "oof_teacher_fold_index": None,
        "distillation_teacher_checkpoint": None,
        "oof_distillation_artifact": None,
        "development_root": "/workspace/data/datasets/DEEPLENS_DATASETS/Model_I",
        "test_root": "/workspace/data/datasets/DEEPLENS_DATASETS/UNUSED_LOCKED_TEST",
        "output_dir": "/workspace/data/outputs/spatial/seed-0/quantum",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SpatialTrainingProtocolTests(unittest.TestCase):
    def test_exact_contract_accepts_both_paired_cores(self):
        validate_paired_spatial_training_contract(spatial_args())
        validate_paired_spatial_training_contract(
            spatial_args(
                core="classical",
                init_full_checkpoint="/tmp/paired/seed-0/classical-init.pt",
            )
        )

    def test_contract_rejects_endpoint_rng_and_official_test_drift(self):
        with self.assertRaisesRegex(ValueError, "protocol drift"):
            validate_paired_spatial_training_contract(
                spatial_args(validation_interval=2)
            )
        with self.assertRaisesRegex(ValueError, "20000"):
            validate_paired_spatial_training_contract(
                spatial_args(training_rng_seed=0)
            )
        with self.assertRaisesRegex(ValueError, "Model_I_test"):
            validate_paired_spatial_training_contract(
                spatial_args(
                    test_root=(
                        "/workspace/data/datasets/DEEPLENS_DATASETS/Model_I_test"
                    )
                )
            )

    def test_trace_requires_deterministic_explicit_rng_reset(self):
        args = argparse.Namespace(
            paired_spatial_init_report=None,
            save_stochastic_trace=True,
            deterministic=False,
            training_rng_seed=None,
        )
        with self.assertRaisesRegex(ValueError, "requires deterministic"):
            validate_paired_spatial_training_contract(args)

    def test_rng_digest_replays_after_reset(self):
        seed_everything(20_002, deterministic=True)
        first = training_rng_state_digests()
        random.random()
        np.random.random()
        torch.rand(4)
        changed = training_rng_state_digests()
        self.assertNotEqual(first, changed)
        seed_everything(20_002, deterministic=True)
        self.assertEqual(first, training_rng_state_digests())

    def test_initializer_report_binds_seed_core_and_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = D4OrbitClassifier(
                num_classes=3,
                heads=4,
                reuploads=3,
                core="quantum",
                include_context=True,
                encoder_variant="micro",
                physics_variant="base",
                physics_summary="moments",
                quantum_encoding="angle",
                observable_readout="pair",
                dropout=0.10,
            )
            backbone = root / "backbone.pt"
            torch.save({"model": source.state_dict(), "epoch": 1}, backbone)
            initializers = root / "initializers"
            build_paired_initializers(
                backbone,
                initializers,
                [0, 1, 2],
                expected_backbone_sha256=file_sha256(backbone),
            )
            report = initializers / "report.json"
            quantum_path = initializers / "seed-0" / "quantum-init.pt"
            quantum_checkpoint = torch.load(
                quantum_path, map_location="cpu", weights_only=False
            )
            args = spatial_args(
                paired_spatial_init_report=str(report),
                init_full_checkpoint=str(quantum_path),
            )
            binding = validate_paired_spatial_initializer_binding(
                args,
                quantum_path,
                quantum_checkpoint,
                dict(quantum_checkpoint["model"]),
            )
            self.assertEqual(binding["seed"], 0)
            self.assertEqual(binding["core"], "quantum")
            self.assertTrue(binding["cross_wire_rejected"])

            classical_path = initializers / "seed-0" / "classical-init.pt"
            classical_checkpoint = torch.load(
                classical_path, map_location="cpu", weights_only=False
            )
            with self.assertRaisesRegex(RuntimeError, "cross-wired"):
                validate_paired_spatial_initializer_binding(
                    args,
                    classical_path,
                    classical_checkpoint,
                    dict(classical_checkpoint["model"]),
                )


if __name__ == "__main__":
    unittest.main()
