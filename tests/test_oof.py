import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from d4_orqb.build_oof_targets import sha256_file, softmax_temperature
from d4_orqb.data import index_membership_sha256
from d4_orqb import train as train_module
from d4_orqb.train import load_oof_distillation_artifact


class OOFArtifactTests(unittest.TestCase):
    def test_oof_student_data_contract_rejects_cache_and_membership_drift(self):
        train_indices = np.asarray([1, 3, 5], dtype=np.int64)
        validation_indices = np.asarray([2, 4], dtype=np.int64)
        validation_holder = {"indices": validation_indices}
        membership = {
            "train": train_module.OOF_FULL_HALF_MEMBERSHIP_SHA256,
            "validation": train_module.OOF_CANONICAL_VAL_MEMBERSHIP_SHA256,
        }
        fingerprints = {
            "images.npy": train_module.OOF_DEVELOPMENT_IMAGES_SHA256,
            "labels.npy": train_module.OOF_DEVELOPMENT_LABELS_SHA256,
            "metadata.json": train_module.OOF_DEVELOPMENT_METADATA_SHA256,
        }

        def fake_file_sha256(path):
            return fingerprints[path.name]

        def fake_membership(indices):
            if indices is train_indices:
                return membership["train"]
            if indices is validation_holder["indices"]:
                return membership["validation"]
            return "unexpected"

        metadata = {
            "class_counts": {"axion": 28897, "cdm": 29772, "no_sub": 28856}
        }
        with mock.patch.object(
            train_module, "file_sha256", side_effect=fake_file_sha256
        ), mock.patch.object(
            train_module, "index_membership_sha256", side_effect=fake_membership
        ):
            actual = train_module.verify_oof_student_data_contract(
                Path("/locked-development-cache"),
                metadata,
                ["axion", "cdm", "no_sub"],
                train_indices,
                validation_indices,
                train_module.OOF_DEVELOPMENT_MANIFEST_SHA256,
            )
            self.assertEqual(actual["images.npy"], fingerprints["images.npy"])

            for filename in ("images.npy", "labels.npy", "metadata.json"):
                original = fingerprints[filename]
                fingerprints[filename] = "0" * 64
                with self.subTest(filename=filename), self.assertRaisesRegex(
                    RuntimeError, "cache identity drifted"
                ):
                    train_module.verify_oof_student_data_contract(
                        Path("/locked-development-cache"),
                        metadata,
                        ["axion", "cdm", "no_sub"],
                        train_indices,
                        validation_indices,
                        train_module.OOF_DEVELOPMENT_MANIFEST_SHA256,
                    )
                fingerprints[filename] = original

            with self.assertRaisesRegex(RuntimeError, "cache identity drifted"):
                train_module.verify_oof_student_data_contract(
                    Path("/locked-development-cache"),
                    metadata,
                    ["axion", "cdm", "no_sub"],
                    train_indices,
                    validation_indices,
                    "0" * 64,
                )

            for bad_names, bad_metadata in (
                (["cdm", "axion", "no_sub"], metadata),
                (
                    ["axion", "cdm", "no_sub"],
                    {
                        "class_counts": {
                            "axion": 28896,
                            "cdm": 29772,
                            "no_sub": 28856,
                        }
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "canonical-validation contract"
                ):
                    train_module.verify_oof_student_data_contract(
                        Path("/locked-development-cache"),
                        bad_metadata,
                        bad_names,
                        train_indices,
                        validation_indices,
                        train_module.OOF_DEVELOPMENT_MANIFEST_SHA256,
                    )

            for key in ("train", "validation"):
                original = membership[key]
                membership[key] = "0" * 64
                with self.subTest(membership=key), self.assertRaisesRegex(
                    RuntimeError, "canonical-validation contract"
                ):
                    train_module.verify_oof_student_data_contract(
                        Path("/locked-development-cache"),
                        metadata,
                        ["axion", "cdm", "no_sub"],
                        train_indices,
                        validation_indices,
                        train_module.OOF_DEVELOPMENT_MANIFEST_SHA256,
                    )
                membership[key] = original

            overlapping_validation = np.asarray([1, 4], dtype=np.int64)
            validation_holder["indices"] = overlapping_validation
            with self.assertRaisesRegex(RuntimeError, "canonical-validation contract"):
                train_module.verify_oof_student_data_contract(
                    Path("/locked-development-cache"),
                    metadata,
                    ["axion", "cdm", "no_sub"],
                    train_indices,
                    overlapping_validation,
                    train_module.OOF_DEVELOPMENT_MANIFEST_SHA256,
                )

    def test_oof_artifact_replays_gates_targets_and_global_lookup(self):
        indices = np.arange(6, dtype=np.int64)
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        morphology = np.asarray(
            [
                [4, 0, 0],
                [3, 1, 0],
                [0, 0, 4],
                [4, 0, 0],
                [0, 4, 0],
                [4, 0, 0],
            ],
            dtype=np.float32,
        )
        spatial = np.asarray(
            [
                [0, 4, 0],
                [0, 4, 0],
                [0, 4, 1],
                [4, 0, 0],
                [0, 4, 0],
                [0, 4, 1],
            ],
            dtype=np.float32,
        )
        morphology_correct = morphology.argmax(1) == labels
        spatial_correct = spatial.argmax(1) == labels
        gate = morphology_correct | spatial_correct
        denominator = (
            morphology_correct.astype(np.float64)
            + spatial_correct.astype(np.float64)
        ).clip(1.0)
        target = (
            morphology_correct[:, None] * softmax_temperature(morphology, 2.0)
            + spatial_correct[:, None] * softmax_temperature(spatial, 2.0)
        ) / denominator[:, None]
        target[~gate] = np.eye(3)[labels[~gate]]
        target = target.astype(np.float32)
        routing = {
            "both_correct": int((morphology_correct & spatial_correct).sum()),
            "morphology_only_correct": int(
                (morphology_correct & ~spatial_correct).sum()
            ),
            "spatial_only_correct": int(
                (~morphology_correct & spatial_correct).sum()
            ),
            "neither_correct": int(
                (~morphology_correct & ~spatial_correct).sum()
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "targets.npz"
            report_path = root / "targets.json"
            np.savez_compressed(
                artifact_path,
                indices=indices,
                labels=labels,
                morphology_logits=morphology,
                spatial_logits=spatial,
                source_fold=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
                target_probabilities=target,
                gate=gate,
            )
            report = {
                "schema_version": 1,
                "protocol": "two-fold-correctness-gated-morphology-spatial-v1",
                "artifact_sha256": sha256_file(artifact_path),
                "samples": len(indices),
                "train_membership_sha256": index_membership_sha256(indices),
                "development_manifest_sha256": "a" * 64,
                "canonical_development_validation_samples_used": 0,
                "official_test_samples_used": 0,
                "checkpoint_selection": "fixed final epoch only",
                "temperature": 2.0,
                "routing_counts": routing,
                "gate_sha256": hashlib.sha256(
                    gate.astype(np.uint8).tobytes()
                ).hexdigest(),
                "target_probability_content_sha256": hashlib.sha256(
                    target.astype("<f4").tobytes()
                ).hexdigest(),
            }
            report_path.write_text(json.dumps(report))
            loaded = load_oof_distillation_artifact(
                artifact_path, report_path, indices, labels, "a" * 64
            )
            self.assertEqual(loaded["routing_counts"], routing)
            np.testing.assert_allclose(
                loaded["morphology_logits"][indices.tolist()].numpy(), morphology
            )
            report["routing_counts"]["neither_correct"] += 1
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "routing counts"):
                load_oof_distillation_artifact(
                    artifact_path, report_path, indices, labels, "a" * 64
                )


if __name__ == "__main__":
    unittest.main()
