import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from d4_orqb.data import hash_ranked_subset
from d4_orqb.evaluate_locked import canonical_model_i_split, file_fingerprint
from d4_orqb.evaluate_spatial_replication import (
    CLASS_NAMES,
    COMPLEMENT_COUNTS,
    COMPLEMENT_MEMBERSHIP_SHA256,
    COMPLEMENT_SAMPLES,
    DEVELOPMENT_CACHE_SHA256,
    DEVELOPMENT_COUNTS,
    FULL_TRAIN_MEMBERSHIP_SHA256,
    HALF_TRAIN_MEMBERSHIP_SHA256,
    RUN_KEYS,
    VALIDATION_MEMBERSHIP_SHA256,
    accuracy_decision_gates,
    analyze_dataset,
    binomial_half_upper_tail,
    build_strict_model,
    canonical_consistency_gates,
    clustered_fixed_seed_bootstrap,
    derive_development_partitions,
    guard_forbidden_path,
    validate_exact_canonical_replay,
    validate_history,
    validate_paired_initializer_binding,
    validate_run_config,
)
from d4_orqb.model import D4OrbitClassifier
from d4_orqb.spatial_paired_init import (
    build_paired_initializers,
    file_sha256,
)
from d4_orqb.train import validate_paired_spatial_initializer_binding


def spatial_config(run: Path, seed: int = 0, core: str = "quantum") -> dict:
    return {
        "development_root": "/workspace/data/datasets/DEEPLENS_DATASETS/Model_I",
        "test_root": "/workspace/data/datasets/DEEPLENS_DATASETS/UNUSED_LOCKED_TEST",
        "cache_root": "/workspace/data/cache/d4-orqb",
        "output_dir": str(run),
        "image_size": 96,
        "encoder_variant": "micro-stat",
        "physics_variant": "base",
        "physics_summary": "moments",
        "core": core,
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
        "encoder_learning_rate": 5e-4,
        "learning_rate": 3e-3,
        "core_learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.02,
        "dropout": 0.10,
        "photon_noise_probability": 0.5,
        "photon_count_min": 2048.0,
        "photon_count_max": 8192.0,
        "seed": seed,
        "training_rng_seed": 20_000 + seed,
        "split_seed": 42,
        "val_fraction": 0.20,
        "max_train_per_class": 11_667,
        "max_val_per_class": None,
        "train_subset_protocol": "hash-v1",
        "evaluate_test": False,
        "deterministic": True,
        "fixed_final_validation_only": True,
        "save_last_validation_predictions": True,
        "save_stochastic_trace": True,
        "init_full_checkpoint": str(run.parent / f"{core}-init.pt"),
        "paired_spatial_init_report": str(run.parent / "report.json"),
        "init_backbone_checkpoint": None,
        "init_compatible_backbone_checkpoint": None,
        "reinitialize_core_after_init": False,
        "tied_mean_dispersion": False,
        "haar_subtype_residual": False,
        "haar_subtype_max_envelope": False,
        "freeze_haar_subtype_residual_at_zero": False,
        "freeze_base_for_haar_subtype_residual": False,
        "shared_late_refinement": False,
        "r2_entanglers": False,
        "freeze_r2_entanglers_at_zero": False,
        "equatorial_readout": False,
        "freeze_equatorial_readout_at_zero": False,
        "meridional_readout": False,
        "freeze_meridional_readout_at_zero": False,
        "subtype_specialist": False,
        "oof_teacher_fold_index": None,
        "distillation_teacher_checkpoint": None,
        "oof_distillation_artifact": None,
        "oof_distillation_report": None,
        "distillation_weight": 0.0,
        "hierarchical_loss_weight": 0.0,
        "branch_loss_weight": 0.0,
        "max_translation_pixels": 0,
        "translation_probability": 1.0,
        "psf_blur_probability": 0.0,
        "read_noise_std": 0.0,
        "subtype_mixup_probability": 0.0,
    }


def logits_for_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    logits = np.full((len(labels), len(CLASS_NAMES)), -3.0, dtype=np.float32)
    logits[np.arange(len(labels)), predictions] = 3.0
    return logits


class SpatialReplicationEvaluationTests(unittest.TestCase):
    def test_directional_binomial_tail_is_exact_on_small_counts(self):
        self.assertEqual(binomial_half_upper_tail(0, 0), 1.0)
        self.assertAlmostEqual(binomial_half_upper_tail(2, 2), 0.25)
        self.assertAlmostEqual(binomial_half_upper_tail(1, 2), 0.75)
        self.assertAlmostEqual(binomial_half_upper_tail(0, 2), 1.0)
        with self.assertRaises(ValueError):
            binomial_half_upper_tail(3, 2)

    def test_exact_complement_membership(self):
        labels = np.concatenate(
            [
                np.full(samples, label, dtype=np.int64)
                for label, samples in enumerate(DEVELOPMENT_COUNTS)
            ]
        )
        full_train, validation = canonical_model_i_split(labels)
        half = hash_ranked_subset(
            full_train,
            labels,
            11_667,
            DEVELOPMENT_CACHE_SHA256["manifest.csv"],
        )
        complement, returned_validation, audit = derive_development_partitions(
            labels, half, validation
        )
        self.assertEqual(len(complement), COMPLEMENT_SAMPLES)
        self.assertEqual(
            tuple(np.bincount(labels[complement], minlength=3)), COMPLEMENT_COUNTS
        )
        self.assertTrue(np.array_equal(returned_validation, validation))
        self.assertEqual(
            audit["membership_sha256"],
            {
                "canonical_full_train": FULL_TRAIN_MEMBERSHIP_SHA256,
                "frozen_half_train": HALF_TRAIN_MEMBERSHIP_SHA256,
                "development_complement": COMPLEMENT_MEMBERSHIP_SHA256,
                "canonical_validation": VALIDATION_MEMBERSHIP_SHA256,
            },
        )
        self.assertEqual(np.intersect1d(complement, half).size, 0)
        self.assertEqual(np.intersect1d(complement, validation).size, 0)

    def test_cluster_bootstrap_carries_all_six_outcomes(self):
        labels = np.repeat(np.arange(3, dtype=np.int64), 100)
        outcomes = np.ones((len(labels), 6), dtype=bool)
        # Every classical arm misses the same 20 examples per class; each
        # example is still represented by one six-outcome cluster.
        for label in range(3):
            indices = np.flatnonzero(labels == label)[:20]
            outcomes[indices, 1::2] = False
        first = clustered_fixed_seed_bootstrap(
            labels, outcomes, samples=2_000, seed=123, chunk_size=100
        )
        second = clustered_fixed_seed_bootstrap(
            labels, outcomes, samples=2_000, seed=123, chunk_size=100
        )
        self.assertAlmostEqual(first["fixed_seed_mean_difference"], 0.2)
        self.assertEqual(first["seed_resampling"], False)
        self.assertEqual(first["random_seed_population_claim"], False)
        # The fixed algorithm and seed must reproduce the same interval.
        self.assertEqual(
            first["fixed_seed_mean_ci95_low"],
            second["fixed_seed_mean_ci95_low"],
        )
        self.assertEqual(
            first["fixed_seed_mean_ci95_high"],
            second["fixed_seed_mean_ci95_high"],
        )

    def test_analysis_applies_hierarchical_gates_and_mcnemar(self):
        labels = np.repeat(np.arange(3, dtype=np.int64), 100)
        quantum_predictions = labels.copy()
        classical_predictions = labels.copy()
        for label in range(3):
            indices = np.flatnonzero(labels == label)[:20]
            classical_predictions[indices] = (label + 1) % 3
        logits = {}
        for seed, core in RUN_KEYS:
            predictions = (
                quantum_predictions if core == "quantum" else classical_predictions
            )
            logits[(seed, core)] = logits_for_predictions(labels, predictions)
        analysis = analyze_dataset(
            labels,
            logits,
            dataset_name="synthetic",
            role="unit-test",
            bootstrap_samples=2_000,
            bootstrap_seed=456,
        )
        self.assertTrue(analysis["accuracy_gates"]["noninferiority"]["passed"])
        self.assertTrue(analysis["accuracy_gates"]["strict_superiority"]["passed"])
        for seed in ("0", "1", "2"):
            self.assertLess(
                analysis["per_seed"][seed][
                    "mcnemar_exact_one_sided_quantum_greater_p"
                ],
                0.05,
            )
        self.assertTrue(
            all(
                value > 0
                for value in analysis[
                    "seed_mean_recall_delta_by_class_quantum_minus_classical"
                ].values()
            )
        )

    def test_noninferiority_rejects_seed_or_class_harm(self):
        bootstrap = {
            "fixed_seed_mean_difference": 0.003,
            "fixed_seed_mean_ci95_low": -0.001,
            "per_seed": {
                "0": {"difference": 0.01},
                "1": {"difference": 0.01},
                "2": {"difference": -0.006},
            },
        }
        per_seed = {
            str(seed): {"mcnemar_exact_one_sided_quantum_greater_p": 0.01}
            for seed in range(3)
        }
        gates = accuracy_decision_gates(
            bootstrap,
            per_seed,
            {"axion": 0.01, "cdm": -0.006, "no_sub": 0.01},
        )
        self.assertFalse(gates["noninferiority"]["passed"])
        self.assertFalse(gates["strict_superiority"]["passed"])

    def test_canonical_consistency_uses_no_p_value(self):
        passing = canonical_consistency_gates(
            {
                "clustered_accuracy_bootstrap": {
                    "fixed_seed_mean_difference": 0.001,
                    "per_seed": {
                        "0": {"difference": 0.001},
                        "1": {"difference": 0.0},
                        "2": {"difference": 0.002},
                    },
                }
            }
        )
        self.assertTrue(passing["noninferiority_consistency"]["passed"])
        self.assertTrue(passing["superiority_consistency"]["passed"])
        self.assertIn("no canonical-validation", passing["role"])

    def test_config_and_history_are_fixed_final_and_test_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary).resolve() / "seed-0" / "quantum"
            run.mkdir(parents=True)
            config = spatial_config(run)
            self.assertEqual(validate_run_config(config, run, 0, "quantum"), config)
            damaged = dict(config)
            damaged["test_root"] = "/workspace/data/Model_I_test"
            with self.assertRaisesRegex(RuntimeError, "official Model-I test"):
                validate_run_config(damaged, run, 0, "quantum")
            history = [
                {"epoch": epoch, "validation": {} if epoch == 40 else None}
                for epoch in range(1, 41)
            ]
            self.assertEqual(validate_history(history, 0, "quantum")["epoch"], 40)
            history[0]["validation"] = {}
            with self.assertRaisesRegex(RuntimeError, "exactly one validation"):
                validate_history(history, 0, "quantum")
        with self.assertRaisesRegex(RuntimeError, "official Model-I test"):
            guard_forbidden_path("/workspace/data/Model_I_test", "input")

    def test_strict_model_reconstruction_rejects_state_damage(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary).resolve() / "seed-0" / "quantum"
            run.mkdir(parents=True)
            config = spatial_config(run)
            torch.manual_seed(0)
            model = D4OrbitClassifier(
                num_classes=3,
                heads=4,
                reuploads=3,
                core="quantum",
                include_context=False,
                dropout=0.1,
                encoder_variant="micro-stat",
                physics_variant="base",
                physics_summary="moments",
                quantum_encoding="angle",
                observable_readout="pair",
            )
            loaded = build_strict_model(config, model.state_dict(), 0, "quantum")
            self.assertEqual(
                sum(parameter.numel() for parameter in loaded.parameters()),
                122_573,
            )
            damaged = dict(model.state_dict())
            damaged.pop("core.params")
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                build_strict_model(config, damaged, 0, "quantum")

    def test_exact_canonical_replay_rejects_one_logit_bit(self):
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        logits = logits_for_predictions(labels, labels)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loaded = {
            key: {
                "endpoint_predictions": {
                    "indices": np.arange(3, dtype=np.int64),
                    "labels": labels,
                    "logits": logits.copy(),
                    "probabilities": probabilities.copy(),
                }
            }
            for key in RUN_KEYS
        }
        replayed = {
            "indices": np.arange(3, dtype=np.int64),
            "labels": labels,
            "logits_by_run": {key: logits.copy() for key in RUN_KEYS},
        }
        report = validate_exact_canonical_replay(loaded, replayed)
        self.assertEqual(len(report), 6)
        replayed["logits_by_run"][(2, "classical")][0, 0] = np.nextafter(
            replayed["logits_by_run"][(2, "classical")][0, 0],
            np.float32(np.inf),
        )
        with self.assertRaisesRegex(RuntimeError, "not bitwise exact"):
            validate_exact_canonical_replay(loaded, replayed)

    def test_paired_binding_rejects_cross_wired_initializer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            backbone = root / "backbone.pt"
            torch.manual_seed(0)
            source = D4OrbitClassifier(
                num_classes=3,
                heads=4,
                reuploads=2,
                core="classical",
                include_context=True,
                dropout=0.1,
                encoder_variant="micro",
                physics_variant="base",
                physics_summary="moments",
                quantum_encoding="angle",
                observable_readout="pair",
            )
            torch.save({"model": source.state_dict(), "epoch": 1}, backbone)
            initializers = root / "initializers"
            build_paired_initializers(
                backbone,
                initializers,
                [0, 1, 2],
                expected_backbone_sha256=file_sha256(backbone),
            )
            report_path = initializers / "report.json"
            checkpoint_path = initializers / "seed-0" / "quantum-init.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            args = argparse.Namespace(
                paired_spatial_init_report=str(report_path),
                seed=0,
                core="quantum",
            )
            binding = validate_paired_spatial_initializer_binding(
                args,
                checkpoint_path,
                checkpoint,
                checkpoint["model"],
            )
            initialization = {"paired_spatial_binding": binding}
            config = {
                "init_full_checkpoint": str(checkpoint_path),
                "paired_spatial_init_report": str(report_path),
            }
            validated = validate_paired_initializer_binding(
                initialization,
                config,
                0,
                "quantum",
                file_fingerprint(checkpoint_path),
                file_fingerprint(report_path),
            )
            self.assertTrue(validated["semantic_cross_wire_check_replayed"])
            cross_wired = dict(config)
            cross_wired["init_full_checkpoint"] = str(
                initializers / "seed-0" / "classical-init.pt"
            )
            with self.assertRaisesRegex(RuntimeError, "cross-wired"):
                validate_paired_initializer_binding(
                    initialization,
                    cross_wired,
                    0,
                    "quantum",
                    file_fingerprint(cross_wired["init_full_checkpoint"]),
                    file_fingerprint(report_path),
                )


if __name__ == "__main__":
    unittest.main()
