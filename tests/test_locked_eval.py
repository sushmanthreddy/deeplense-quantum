import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from d4_orqb.evaluate_locked import (
    CACHE_ARTIFACTS,
    MAX_VALIDATION_REPLAY_METRIC_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
    MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
    MODEL_I_CLASSES,
    RUN_ARTIFACTS,
    SCHEMA_VERSION,
    _code_fingerprints,
    _metric_close,
    _parse_comparison,
    _validate_replay_tolerances,
    atomic_json,
    assert_no_symlink_components,
    assert_fingerprints_equal,
    file_fingerprint,
    fingerprint_named_files,
    build_parser,
    load_and_verify_seal,
    load_validation_predictions,
    mcnemar_exact,
    metrics_from_probabilities,
    probability_replay_diagnostics,
    resolve_model_config,
    runtime_fingerprint,
    sha256_file,
    softmax_numpy,
    stratified_paired_bootstrap_accuracy,
    uniform_probability_ensemble,
    validate_frozen_manifest,
    validate_resumable_output,
    validate_split_indices,
    wilson_interval,
)


class LockedEvaluationHelperTests(unittest.TestCase):
    def test_locked_loader_rejects_shared_late_refinement(self):
        with self.assertRaisesRegex(RuntimeError, "shared late refinement"):
            resolve_model_config(
                "d4-orqb", {"shared_late_refinement": True}, {}
            )
        with self.assertRaisesRegex(RuntimeError, "shared late refinement"):
            resolve_model_config(
                "d4-orqb",
                {},
                {"encoder.shared_refinement_gates": np.zeros(4)},
            )

    def test_locked_loader_rejects_haar_subtype_residual(self):
        with self.assertRaisesRegex(RuntimeError, "max-preserving"):
            resolve_model_config(
                "d4-orqb", {"haar_subtype_max_envelope": True}, {}
            )
        with self.assertRaisesRegex(RuntimeError, "Haar subtype residual"):
            resolve_model_config(
                "d4-orqb", {"haar_subtype_residual": True}, {}
            )
        with self.assertRaisesRegex(RuntimeError, "Haar subtype residual"):
            resolve_model_config(
                "d4-orqb", {}, {"haar_subtype_residual.weight": np.zeros(15)}
            )

    @staticmethod
    def frozen_manifest():
        fingerprint = lambda: {"bytes": 0, "sha256": "0" * 64}
        members = {
            name: {
                "kind": "d4-orqb",
                "source_run": f"/runs/{name}",
                "artifacts": {artifact: fingerprint() for artifact in RUN_ARTIFACTS},
                "resolved_config": {},
                "config_migrations": [],
            }
            for name in ("q", "c")
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": {
                "id": "model-i-lock",
                "two_stage": True,
                "test_unavailable_during_freeze": True,
                "validation_replay_required_before_test_marker": True,
                "ensemble_rule": "unweighted arithmetic mean of member probabilities",
                "bootstrap_samples": 1000,
                "analysis_seed": 9,
                "validation_replay_probability_atol": (
                    MAX_VALIDATION_REPLAY_PROBABILITY_ATOL
                ),
                "validation_replay_probability_mean_atol": (
                    MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL
                ),
                "validation_replay_probability_p99_atol": (
                    MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL
                ),
                "validation_replay_metric_atol": MAX_VALIDATION_REPLAY_METRIC_ATOL,
                "inference": {
                    "batch_size": 128,
                    "workers": 8,
                    "loader_seed": 42,
                    "autocast": "cuda-bfloat16",
                },
            },
            "development_cache": {
                "path": "/cache/development",
                "classes": MODEL_I_CLASSES,
                "samples": 87_525,
                "image_size": 96,
                "artifacts": {artifact: fingerprint() for artifact in CACHE_ARTIFACTS},
            },
            "expected_official_test": {
                "canonical_job_path": "/cache/test",
                "classes": MODEL_I_CLASSES,
                "samples": 15_000,
                "image_size": 96,
                "artifact_sha256": {artifact: "1" * 64 for artifact in CACHE_ARTIFACTS},
            },
            "members": members,
            "ensembles": {"qc": ["q", "c"]},
            "comparisons": [
                {
                    "a": "q",
                    "b": "c",
                    "minimum_acceptable_accuracy_difference": -0.01,
                }
            ],
            "validation_plan_metrics": {"q": {}, "c": {}, "qc": {}},
        }

    def test_uniform_ensemble_averages_probabilities_not_logits(self):
        logits_a = np.array([[8.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        logits_b = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 6.0]])
        probabilities_a = softmax_numpy(logits_a)
        probabilities_b = softmax_numpy(logits_b)
        actual = uniform_probability_ensemble([probabilities_a, probabilities_b])
        np.testing.assert_allclose(actual, (probabilities_a + probabilities_b) / 2)
        self.assertFalse(
            np.allclose(actual, softmax_numpy((logits_a + logits_b) / 2), atol=1e-4)
        )

    def test_wilson_interval_handles_extreme_accuracy(self):
        zero = wilson_interval(0, 10)
        perfect = wilson_interval(10, 10)
        self.assertEqual(zero["low"], 0.0)
        self.assertGreater(zero["high"], 0.0)
        self.assertLess(perfect["low"], 1.0)
        self.assertEqual(perfect["high"], 1.0)

    def test_stratified_paired_statistics_are_deterministic(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        perfect = np.eye(3)[labels]
        wrong = np.roll(perfect, 1, axis=1)
        first = stratified_paired_bootstrap_accuracy(
            labels, perfect, wrong, samples=200, seed=9, chunk_size=17
        )
        second = stratified_paired_bootstrap_accuracy(
            labels, perfect, wrong, samples=200, seed=9, chunk_size=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first["difference"], 1.0)
        self.assertEqual(first["ci95_low"], 1.0)
        exact = mcnemar_exact(labels, perfect, wrong)
        self.assertEqual(exact["a_correct_b_wrong"], 6)
        self.assertEqual(exact["a_wrong_b_correct"], 0)
        self.assertAlmostEqual(exact["two_sided_exact_p"], 0.03125, places=14)

    def test_split_and_prediction_validation_refuse_manifest_drift(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        train, val = validate_split_indices(labels, [0, 2, 4], [1, 3, 5])
        logits = np.eye(3)[labels[val]] * 4.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            np.savez_compressed(
                path,
                indices=val,
                labels=labels[val],
                logits=logits,
                probabilities=softmax_numpy(logits),
            )
            loaded = load_validation_predictions(path, val, labels, ["a", "b", "c"])
            np.testing.assert_array_equal(loaded["indices"], val)
            with self.assertRaisesRegex(RuntimeError, "order"):
                load_validation_predictions(path, val[::-1], labels, ["a", "b", "c"])
        with self.assertRaisesRegex(RuntimeError, "complete"):
            validate_split_indices(labels, train[:-1], val)

    def test_atomic_json_and_fingerprint_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            atomic_json(path, {"locked": True})
            self.assertEqual(json.loads(path.read_text()), {"locked": True})
            expected = {"artifact.json": file_fingerprint(path)}
            actual = {"artifact.json": file_fingerprint(path)}
            assert_fingerprints_equal(actual, expected, "synthetic seal")
            path.write_text('{"locked": false}\n')
            changed = {"artifact.json": file_fingerprint(path)}
            with self.assertRaisesRegex(RuntimeError, "changed"):
                assert_fingerprints_equal(changed, expected, "synthetic seal")

    def test_code_fingerprint_materializes_kind_generator(self):
        lens_only = _code_fingerprints(iter(["lenspinn-repaired"]))
        lens_first = _code_fingerprints(iter(["lenspinn-repaired", "d4-orqb"]))
        self.assertIn("lenspinn.py", lens_only)
        self.assertIn("lenspinn.py", lens_first)
        self.assertIn("__init__.py", lens_first)
        self.assertIn("model.py", lens_first)
        self.assertIn("quantum.py", lens_first)

    def test_manifest_rejects_reversed_comparisons_and_missing_test_hash(self):
        manifest = self.frozen_manifest()
        validate_frozen_manifest(manifest)
        duplicate = copy.deepcopy(manifest)
        duplicate["comparisons"].append(
            {
                "a": "c",
                "b": "q",
                "minimum_acceptable_accuracy_difference": None,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate or reversed"):
            validate_frozen_manifest(duplicate)
        missing_hash = copy.deepcopy(manifest)
        del missing_hash["expected_official_test"]["artifact_sha256"]["images.npy"]
        with self.assertRaisesRegex(RuntimeError, "artifact set"):
            validate_frozen_manifest(missing_hash)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "within"):
            _parse_comparison("q,c,1.1")

    def test_metric_replay_checks_calibration_and_per_class_values(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        probabilities = np.full((6, 3), 0.05)
        probabilities[np.arange(6), labels] = 0.90
        metrics = metrics_from_probabilities(labels, probabilities, ["a", "b", "c"])
        self.assertEqual(
            _metric_close(metrics, copy.deepcopy(metrics), "synthetic replay"), 0.0
        )
        changed = copy.deepcopy(metrics)
        changed["ece_15"] += 1e-3
        with self.assertRaisesRegex(RuntimeError, "ece_15"):
            _metric_close(metrics, changed, "synthetic replay")
        changed = copy.deepcopy(metrics)
        changed["per_class"]["a"]["recall"] -= 1e-3
        with self.assertRaisesRegex(RuntimeError, "a.recall"):
            _metric_close(metrics, changed, "synthetic replay")
        large_expected = copy.deepcopy(metrics)
        large_expected["samples"] = 10_000
        relative_only_would_pass = copy.deepcopy(large_expected)
        relative_only_would_pass["samples"] = 10_001
        with self.assertRaisesRegex(RuntimeError, "samples"):
            _metric_close(
                relative_only_would_pass,
                large_expected,
                "absolute replay gate",
                tolerance=2e-4,
            )

    def test_replay_diagnostics_and_all_hard_gate_bounds(self):
        expected = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        actual = np.array([[0.69, 0.21, 0.1], [0.11, 0.79, 0.1]])
        diagnostics = probability_replay_diagnostics(actual, expected)
        self.assertAlmostEqual(
            diagnostics["max_probability_absolute_difference"], 0.01
        )
        self.assertAlmostEqual(
            diagnostics["mean_probability_absolute_difference"], 4 * 0.01 / 6
        )
        self.assertAlmostEqual(
            diagnostics["p99_probability_absolute_difference"], 0.01
        )
        self.assertTrue(diagnostics["predicted_classes_exact"])
        changed_class = expected.copy()
        changed_class[0] = [0.2, 0.7, 0.1]
        self.assertFalse(
            probability_replay_diagnostics(changed_class, expected)[
                "predicted_classes_exact"
            ]
        )

        validated = _validate_replay_tolerances(
            MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
            MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
            MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
            MAX_VALIDATION_REPLAY_METRIC_ATOL,
        )
        self.assertEqual(
            validated["probability_atol"], MAX_VALIDATION_REPLAY_PROBABILITY_ATOL
        )
        values = [
            MAX_VALIDATION_REPLAY_PROBABILITY_ATOL,
            MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
            MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
            MAX_VALIDATION_REPLAY_METRIC_ATOL,
        ]
        for index, upper_bound in enumerate(values):
            invalid = values.copy()
            invalid[index] = np.nextafter(upper_bound, np.inf)
            with self.assertRaisesRegex(ValueError, "outside"):
                _validate_replay_tolerances(*invalid)

        manifest = self.frozen_manifest()
        validate_frozen_manifest(manifest)
        legacy_manifest = copy.deepcopy(manifest)
        legacy_manifest["schema_version"] = 1
        with self.assertRaisesRegex(RuntimeError, "schema"):
            validate_frozen_manifest(legacy_manifest)
        invalid_manifest = copy.deepcopy(manifest)
        invalid_manifest["protocol"]["validation_replay_metric_atol"] = 2.01e-4
        with self.assertRaisesRegex(RuntimeError, "outside"):
            validate_frozen_manifest(invalid_manifest)

    def test_freeze_parser_defaults_to_predeclared_replay_gates(self):
        arguments = build_parser().parse_args(
            [
                "freeze",
                "--development-cache",
                "/development",
                "--seal-dir",
                "/seal",
                "--protocol-id",
                "model-i-v3",
                "--expected-test-manifest-sha256",
                "0" * 64,
                "--expected-test-images-sha256",
                "1" * 64,
                "--expected-test-labels-sha256",
                "2" * 64,
                "--expected-test-metadata-sha256",
                "3" * 64,
                "--expected-development-manifest-sha256",
                "4" * 64,
                "--expected-test-cache-path",
                "/test",
                "--member",
                "q=d4-orqb=/run",
            ]
        )
        self.assertEqual(arguments.replay_atol, MAX_VALIDATION_REPLAY_PROBABILITY_ATOL)
        self.assertEqual(
            arguments.replay_mean_atol,
            MAX_VALIDATION_REPLAY_PROBABILITY_MEAN_ATOL,
        )
        self.assertEqual(
            arguments.replay_p99_atol,
            MAX_VALIDATION_REPLAY_PROBABILITY_P99_ATOL,
        )
        self.assertEqual(arguments.replay_metric_atol, MAX_VALIDATION_REPLAY_METRIC_ATOL)

    def test_symlink_components_and_resumed_prediction_redirect_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                assert_no_symlink_components(link / "child", "synthetic")
            regular_file = root / "regular.bin"
            regular_file.write_bytes(b"locked")
            file_link = root / "file-link.bin"
            file_link.symlink_to(regular_file)
            with self.assertRaisesRegex(RuntimeError, "non-symlink"):
                file_fingerprint(file_link)

            output = root / "output"
            output.mkdir()
            (output / "TEST_ACCESS_MARKER.json").write_text("{}")
            (output / "predictions").symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                validate_resumable_output(output, ["q", "c"])

    def test_legacy_encoder_inference_fails_closed_when_tensors_are_missing(self):
        config = {
            "heads": 4,
            "reuploads": 2,
            "core": "quantum",
            "include_context": False,
            "image_size": 96,
        }
        with self.assertRaisesRegex(RuntimeError, "missing tensors"):
            resolve_model_config("d4-orqb", config, {})

    def test_seal_verification_rejects_unmanifested_root_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            seal_dir = Path(temporary) / "seal"
            members_dir = seal_dir / "members"
            members_dir.mkdir(parents=True)
            manifest = self.frozen_manifest()
            for name in manifest["members"]:
                member_dir = members_dir / name
                member_dir.mkdir()
                for artifact in RUN_ARTIFACTS:
                    (member_dir / artifact).write_bytes(artifact.encode())
                manifest["members"][name]["artifacts"] = fingerprint_named_files(
                    member_dir, RUN_ARTIFACTS
                )
            manifest["code"] = _code_fingerprints(["d4-orqb"])
            manifest["runtime"] = runtime_fingerprint(["d4-orqb"])
            atomic_json(seal_dir / "manifest.json", manifest)
            atomic_json(
                seal_dir / "seal.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_utc": "synthetic",
                    "manifest_sha256": sha256_file(seal_dir / "manifest.json"),
                },
            )
            loaded, _ = load_and_verify_seal(seal_dir)
            self.assertEqual(set(loaded["members"]), {"q", "c"})
            (seal_dir / "unexpected.txt").write_text("ambiguous")
            with self.assertRaisesRegex(RuntimeError, "unexpected root"):
                load_and_verify_seal(seal_dir)


if __name__ == "__main__":
    unittest.main()
