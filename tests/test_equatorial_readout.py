import copy
import unittest

import torch

from d4_orqb.model import D4OrbitClassifier
from d4_orqb.quantum import (
    D4_ELEMENTS,
    D4OrbitQuantumBottleneck,
    _expectation_pauli_strings,
    right_regular_permutation,
)
from d4_orqb.train import (
    ANNULAR_HAAR_BASE_CHECKPOINT_SHA256,
    OOF_DEVELOPMENT_MANIFEST_SHA256,
    OOF_FULL_HALF_MEMBERSHIP_SHA256,
    configure_equatorial_optimization_and_report,
    equatorial_readout_initialization_record,
    equatorial_readout_probability_replay,
    remap_haar_to_equatorial_readout,
    validate_equatorial_source_contract,
)


class EquatorialReadoutTests(unittest.TestCase):
    @staticmethod
    def _haar_model(equatorial_readout: bool = False) -> D4OrbitClassifier:
        return D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            core="quantum",
            heads=4,
            reuploads=2,
            equatorial_readout=equatorial_readout,
            dropout=0.0,
        )

    def test_core_zero_replay_all_phase_gradients_and_d4_covariance(self):
        torch.manual_seed(17)
        source = D4OrbitQuantumBottleneck(heads=4, reuploads=2)
        target = D4OrbitQuantumBottleneck(
            heads=4, reuploads=2, equatorial_readout=True
        )
        target.params.data.copy_(source.params.data)
        self.assertEqual(target.parameter_report()["quantum_trainable"], 104)
        self.assertEqual(
            target.parameter_report()["state_preparation_trainable"], 88
        )
        self.assertEqual(
            target.parameter_report()["equatorial_readout_trainable"], 16
        )
        self.assertTrue(
            torch.equal(
                target.readout_phases,
                torch.zeros_like(target.readout_phases),
            )
        )

        angles = torch.randn(2, 4, 2, 8)
        source_features = source(angles)
        target_features = target(angles)
        self.assertTrue(torch.equal(target_features, source_features))

        weights = torch.linspace(
            0.1, 2.0, target_features.numel()
        ).reshape_as(target_features)
        (target_features * weights).sum().backward()
        gradients = target.readout_phases.grad
        self.assertIsNotNone(gradients)
        self.assertTrue(torch.isfinite(gradients).all())
        self.assertTrue((gradients.abs() > 1e-8).all())

        target.zero_grad(set_to_none=True)
        with torch.no_grad():
            target.readout_phases.normal_(mean=0.0, std=0.2)
        reference, reference_equivariant = target(
            angles, return_equivariant=True
        )
        self.assertEqual(
            tuple(reference_equivariant["equatorial"].shape),
            (2, 4, 4, 8),
        )
        for element in D4_ELEMENTS:
            permutation = right_regular_permutation(element)
            transformed, transformed_equivariant = target(
                angles.index_select(-1, permutation),
                return_equivariant=True,
            )
            torch.testing.assert_close(
                transformed, reference, rtol=2e-5, atol=2e-5
            )
            for observable in ("z", "x", "equatorial"):
                expected = reference_equivariant[observable].index_select(
                    -1, permutation
                )
                torch.testing.assert_close(
                    transformed_equivariant[observable],
                    expected,
                    rtol=2e-5,
                    atol=2e-5,
                )

    def test_xy_string_expectations_match_dense_pauli_algebra(self):
        torch.manual_seed(19)
        core = D4OrbitQuantumBottleneck(heads=4, reuploads=2)
        real = torch.randn(2, 256)
        imaginary = torch.randn(2, 256)
        state = torch.complex(real, imaginary)
        state = state / state.norm(dim=1, keepdim=True)
        strings = (
            ((0, "Y"),),
            ((3, "X"),),
            ((0, "X"), (5, "Y")),
            ((1, "Y"), (6, "X")),
            ((2, "Y"), (7, "Y")),
        )
        actual = _expectation_pauli_strings(
            state, strings, 8, core.z_signs
        )
        identity = torch.eye(2, dtype=torch.complex64)
        pauli_x = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64
        )
        pauli_y = torch.tensor(
            [[0.0, -1j], [1j, 0.0]], dtype=torch.complex64
        )
        expected = []
        for terms in strings:
            by_wire = dict(terms)
            operator = torch.ones(1, 1, dtype=torch.complex64)
            for wire in range(8):
                factor = {
                    "X": pauli_x,
                    "Y": pauli_y,
                }.get(by_wire.get(wire), identity)
                operator = torch.kron(operator, factor)
            transformed = state @ operator.transpose(0, 1)
            expected.append((state.conj() * transformed).sum(dim=1).real)
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_exact_budget_gauge_remap_probability_replay_and_frozen_report(self):
        torch.manual_seed(23)
        source = self._haar_model(False)
        target = self._haar_model(True)
        source_report = source.parameter_report()
        target_report = target.parameter_report()
        self.assertEqual(source_report["total"], 122_595)
        self.assertEqual(source_report["quantum"], 88)
        self.assertEqual(target_report["total"], 122_610)
        self.assertEqual(target_report["quantum"], 104)
        self.assertEqual(target_report["quantum_state_preparation_trainable"], 88)
        self.assertEqual(target_report["equatorial_readout_trainable"], 16)
        self.assertEqual(target_report["head_and_context"], 1_304)
        self.assertEqual(target_report["classifier_bias_trainable"], 2)
        self.assertEqual(target_report["classifier_bias_gauge_degrees"], 2)

        source_state = source.state_dict()
        remapped, adaptations = remap_haar_to_equatorial_readout(
            source_state, target.state_dict()
        )
        self.assertEqual(
            [item["method"] for item in adaptations],
            [
                "zero-new-d4-equatorial-measurement-phases",
                "remove-softmax-common-logit-gauge",
            ],
        )
        self.assertTrue(
            torch.equal(
                remapped["core.readout_phases"], torch.zeros(4, 4)
            )
        )
        expected_bias = (
            source_state["head.classifier.bias"][:2]
            - source_state["head.classifier.bias"][2]
        )
        torch.testing.assert_close(
            remapped["head.classifier.bias"],
            expected_bias,
            rtol=0.0,
            atol=0.0,
        )
        target.load_state_dict(remapped, strict=True)
        record = equatorial_readout_initialization_record(target)
        self.assertTrue(record["all_phases_zero_after_remap"])

        core_angles = torch.randn(2, 4, 2, 8)
        self.assertTrue(
            torch.equal(source.core(core_angles), target.core(core_angles))
        )
        images = torch.rand(1, 1, 32, 32)
        replay = equatorial_readout_probability_replay(
            source, target, images
        )
        self.assertTrue(replay["float32"]["features_bitwise_equal"])
        self.assertTrue(replay["float32"]["predictions_equal"])
        self.assertTrue(
            replay["float32"]["probabilities_equal_within_tolerance"]
        )
        self.assertTrue(
            replay["bfloat16_autocast"]["features_bitwise_equal"]
        )
        self.assertTrue(replay["bfloat16_autocast"]["predictions_equal"])
        self.assertTrue(
            replay["bfloat16_autocast"][
                "probabilities_equal_within_tolerance"
            ]
        )
        self.assertEqual(
            replay["bfloat16_autocast"]["probability_tolerance"], 2e-3
        )

        trainable_report = configure_equatorial_optimization_and_report(
            target, freeze_at_zero=False
        )
        self.assertEqual(trainable_report["inference_total"], 122_610)
        self.assertEqual(
            trainable_report["optimization_trainable_total"], 122_610
        )
        self.assertEqual(
            trainable_report["quantum_optimization_trainable"], 104
        )
        self.assertEqual(
            trainable_report["equatorial_readout_optimization_trainable"], 16
        )

        frozen = self._haar_model(True)
        frozen.load_state_dict(remapped, strict=True)
        frozen_report = configure_equatorial_optimization_and_report(
            frozen, freeze_at_zero=True
        )
        self.assertEqual(frozen_report["total"], 122_610)
        self.assertEqual(frozen_report["quantum"], 104)
        self.assertEqual(frozen_report["inference_total"], 122_610)
        self.assertEqual(
            frozen_report["optimization_trainable_total"], 122_594
        )
        self.assertEqual(frozen_report["quantum_optimization_trainable"], 88)
        self.assertEqual(
            frozen_report["equatorial_readout_optimization_trainable"], 0
        )
        self.assertFalse(frozen.core.readout_phases.requires_grad)

    def test_checkpoint_data_and_test_locks_fail_closed(self):
        config = {
            "image_size": 96,
            "encoder_variant": "deep-se-haar-morph",
            "physics_variant": "base",
            "physics_summary": "moments-morphology-haar",
            "heads": 4,
            "reuploads": 2,
            "quantum_encoding": "angle",
            "observable_readout": "pair",
            "include_context": False,
            "core": "quantum",
            "evaluate_test": False,
            "train_subset_protocol": "hash-v1",
            "max_train_per_class": 11667,
        }
        parameters = {"total": 122595, "quantum": 88}
        source_data = {
            "train_size": 35001,
            "train_membership_sha256": OOF_FULL_HALF_MEMBERSHIP_SHA256,
            "development_manifest_sha256": OOF_DEVELOPMENT_MANIFEST_SHA256,
            "class_names": ["axion", "cdm", "no_sub"],
            "official_test_cache_opened": False,
        }
        summary = {"official_test_evaluated": False}
        target_data = copy.deepcopy(source_data)
        record = validate_equatorial_source_contract(
            ANNULAR_HAAR_BASE_CHECKPOINT_SHA256,
            config,
            parameters,
            source_data,
            summary,
            target_data,
        )
        self.assertTrue(record["same_training_membership"])
        self.assertFalse(record["source_official_test_opened"])

        mutations = []
        mutations.append(("checkpoint", "0" * 64))
        bad_config = copy.deepcopy(config)
        bad_config["equatorial_readout"] = True
        mutations.append(("config", bad_config))
        bad_parameters = copy.deepcopy(parameters)
        bad_parameters["total"] = 122610
        mutations.append(("parameters", bad_parameters))
        bad_source_data = copy.deepcopy(source_data)
        bad_source_data["official_test_cache_opened"] = True
        mutations.append(("source_data", bad_source_data))
        bad_summary = {"official_test_evaluated": True}
        mutations.append(("summary", bad_summary))
        bad_target_data = copy.deepcopy(target_data)
        bad_target_data["train_membership_sha256"] = "bad"
        mutations.append(("target_data", bad_target_data))
        opened_target_data = copy.deepcopy(target_data)
        opened_target_data["test"] = {"classes": []}
        mutations.append(("target_data", opened_target_data))

        for field, value in mutations:
            with self.subTest(field=field, value=str(value)[:40]):
                arguments = {
                    "checkpoint_sha256": ANNULAR_HAAR_BASE_CHECKPOINT_SHA256,
                    "source_config": config,
                    "source_parameters": parameters,
                    "source_data": source_data,
                    "source_summary": summary,
                    "target_data": target_data,
                }
                arguments[
                    {
                        "checkpoint": "checkpoint_sha256",
                        "config": "source_config",
                        "parameters": "source_parameters",
                        "source_data": "source_data",
                        "summary": "source_summary",
                        "target_data": "target_data",
                    }[field]
                ] = value
                with self.assertRaises(RuntimeError):
                    validate_equatorial_source_contract(**arguments)


if __name__ == "__main__":
    unittest.main()
