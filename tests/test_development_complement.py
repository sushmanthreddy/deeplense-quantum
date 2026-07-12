import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from d4_orqb.data import hash_ranked_subset
from d4_orqb.evaluate_development_complement import (
    CANDIDATE_NAME,
    CLASS_NAMES,
    COMPLEMENT_COUNTS,
    COMPLEMENT_MEMBERSHIP_SHA256,
    COMPLEMENT_SAMPLES,
    CONTROL_NAME,
    DEVELOPMENT_CACHE_SHA256,
    DEVELOPMENT_COUNTS,
    FULL_TRAIN_MEMBERSHIP_SHA256,
    HALF_TRAIN_MEMBERSHIP_SHA256,
    VALIDATION_MEMBERSHIP_SHA256,
    analyze_pair,
    build_subtype_model_strict,
    derive_complement_membership,
    guard_forbidden_test_path,
    load_locked_arm,
    paired_gate,
    refuse_existing_output,
    validate_canonical_validation_replay,
)
from d4_orqb.evaluate_locked import canonical_model_i_split, softmax_numpy
from d4_orqb.model import D4OrbitClassifier


class DevelopmentComplementTests(unittest.TestCase):
    @staticmethod
    def _config(*, frozen: bool) -> dict:
        return {
            "image_size": 96,
            "heads": 4,
            "reuploads": 2,
            "core": "quantum",
            "include_context": False,
            "dropout": 0.1,
            "encoder_variant": "deep-se-haar-morph",
            "physics_variant": "base",
            "physics_summary": "moments-morphology-haar",
            "quantum_encoding": "angle",
            "observable_readout": "pair",
            "tied_mean_dispersion": False,
            "haar_subtype_residual": True,
            "shared_late_refinement": False,
            "haar_subtype_max_envelope": False,
            "r2_entanglers": False,
            "equatorial_readout": False,
            "meridional_readout": False,
            "freeze_haar_subtype_residual_at_zero": frozen,
            "freeze_base_for_haar_subtype_residual": False,
            "evaluate_test": False,
            "deterministic": True,
            "epochs": 15,
            "validation_interval": 15,
            "fixed_final_validation_only": True,
            "seed": 0,
            "split_seed": 42,
            "max_train_per_class": 11_667,
            "train_subset_protocol": "hash-v1",
            "output_dir": "synthetic-test-only",
        }

    @classmethod
    def _model(cls) -> D4OrbitClassifier:
        config = cls._config(frozen=False)
        return D4OrbitClassifier(
            num_classes=3,
            heads=config["heads"],
            reuploads=config["reuploads"],
            core=config["core"],
            include_context=config["include_context"],
            dropout=config["dropout"],
            encoder_variant=config["encoder_variant"],
            physics_variant=config["physics_variant"],
            physics_summary=config["physics_summary"],
            quantum_encoding=config["quantum_encoding"],
            observable_readout=config["observable_readout"],
            tied_mean_dispersion=config["tied_mean_dispersion"],
            haar_subtype_residual=config["haar_subtype_residual"],
            shared_late_refinement=config["shared_late_refinement"],
            haar_subtype_max_envelope=config["haar_subtype_max_envelope"],
            r2_entanglers=config["r2_entanglers"],
            equatorial_readout=config["equatorial_readout"],
            meridional_readout=config["meridional_readout"],
        )

    def test_exact_canonical_complement_membership(self):
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
        complement, returned_validation, audit = derive_complement_membership(
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
                "complement": COMPLEMENT_MEMBERSHIP_SHA256,
                "canonical_validation": VALIDATION_MEMBERSHIP_SHA256,
            },
        )
        self.assertEqual(np.intersect1d(complement, half).size, 0)
        self.assertEqual(np.intersect1d(complement, validation).size, 0)

    def test_forbidden_path_and_existing_output_refusal(self):
        with self.assertRaisesRegex(RuntimeError, "official Model-I test"):
            guard_forbidden_test_path(
                Path("/workspace/data/datasets/DEEPLENS_DATASETS/Model_I_test"),
                "input",
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "result"
            self.assertEqual(refuse_existing_output(output), output)
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "existing output"):
                refuse_existing_output(output)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                guard_forbidden_test_path(alias / "new-result", "aliased output")

    def test_strict_subtype_checkpoint_loader(self):
        source = self._model()
        with torch.no_grad():
            source.haar_subtype_residual.weight[0] = 0.125
        config = self._config(frozen=False)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "config.json").write_text(json.dumps(config))
            torch.save(
                {"model": source.state_dict(), "epoch": 15, "record": {}},
                run / "best.pt",
            )
            loaded, loaded_config = load_locked_arm(run, expected_frozen=False)
            self.assertEqual(loaded_config, config)
            self.assertEqual(
                float(loaded.haar_subtype_residual.weight[0].detach()), 0.125
            )
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))

        damaged_state = dict(source.state_dict())
        damaged_state.pop(next(iter(damaged_state)))
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            build_subtype_model_strict(
                config, damaged_state, expected_frozen=False
            )

    def test_validation_replay_accepts_exact_and_rejects_drift(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        indices = np.asarray([5, 0, 4, 1, 3, 2], dtype=np.int64)
        logits = np.asarray(
            [
                [3.0, 0.0, -1.0],
                [-1.0, 3.0, 0.0],
                [0.0, -1.0, 3.0],
                [2.0, 1.0, -1.0],
                [-1.0, 2.0, 1.0],
                [1.0, -1.0, 2.0],
            ],
            dtype=np.float32,
        )
        probabilities = softmax_numpy(logits)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (CANDIDATE_NAME, CONTROL_NAME):
                run = root / name
                run.mkdir()
                np.savez_compressed(
                    run / "best_validation_predictions.npz",
                    indices=indices,
                    labels=labels[indices],
                    logits=logits,
                    probabilities=probabilities,
                )
            replayed = {
                "indices": indices,
                "labels": labels[indices],
                "candidate_probabilities": probabilities.copy(),
                "control_probabilities": probabilities.copy(),
            }
            report = validate_canonical_validation_replay(
                root, indices, labels, replayed
            )
            self.assertEqual(set(report), {CANDIDATE_NAME, CONTROL_NAME})
            self.assertTrue(report[CANDIDATE_NAME]["predicted_classes_exact"])

            drifted = dict(replayed)
            drifted["candidate_probabilities"] = probabilities.copy()
            drifted["candidate_probabilities"][0, 0] -= 0.01
            drifted["candidate_probabilities"][0, 1] += 0.01
            with self.assertRaisesRegex(RuntimeError, "replay failed"):
                validate_canonical_validation_replay(root, indices, labels, drifted)

    def test_predeclared_gate_is_a_strict_conjunction(self):
        passing = paired_gate(
            candidate_correct=101,
            control_correct=100,
            bootstrap={"ci95_low": 0.001},
            mcnemar={"two_sided_exact_p": 0.01},
            per_class_delta=dict(zip(CLASS_NAMES, (1, 0, 0))),
            candidate_metrics={"macro_auc_ovr": 0.91, "nll": 0.10},
            control_metrics={"macro_auc_ovr": 0.90, "nll": 0.11},
        )
        self.assertTrue(passing["passed"])
        self.assertTrue(all(passing["conditions"].values()))

        failing = paired_gate(
            candidate_correct=101,
            control_correct=100,
            bootstrap={"ci95_low": 0.0},
            mcnemar={"two_sided_exact_p": 0.01},
            per_class_delta=dict(zip(CLASS_NAMES, (2, 0, -1))),
            candidate_metrics={"macro_auc_ovr": 0.89, "nll": 0.12},
            control_metrics={"macro_auc_ovr": 0.90, "nll": 0.11},
        )
        self.assertFalse(failing["passed"])
        self.assertFalse(failing["conditions"]["every_class_correct_delta_nonnegative"])

    def test_analyze_pair_returns_complete_finite_result(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        candidate = np.asarray(
            [
                [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
            ],
            dtype=np.float64,
        )
        control = candidate.copy()
        control[0] = [0.4, 0.5, 0.1]
        with mock.patch(
            "d4_orqb.evaluate_development_complement."
            "stratified_paired_bootstrap_accuracy",
            return_value={
                "difference": 1.0 / 6.0,
                "ci95_low": 0.01,
                "ci95_high": 0.30,
                "bootstrap_samples": 100_000,
                "seed": 20260805,
            },
        ):
            result = analyze_pair(
                {
                    "labels": labels,
                    "candidate_probabilities": candidate,
                    "control_probabilities": control,
                }
            )
        self.assertEqual(result["candidate_correct"], 6)
        self.assertEqual(result["control_correct"], 5)
        self.assertEqual(set(result["metrics"]), {CANDIDATE_NAME, CONTROL_NAME})
        self.assertIn("predeclared_complement_gate", result)
        self.assertIsInstance(result["passed_full_gate"], bool)


if __name__ == "__main__":
    unittest.main()
