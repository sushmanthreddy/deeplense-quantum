import copy
import unittest
from types import SimpleNamespace

import torch

from d4_orqb.model import D4OrbitClassifier
from d4_orqb.quantum import (
    D4_ELEMENTS,
    R_EDGES,
    D4OrbitQuantumBottleneck,
    _expectation_symmetric_xz,
    right_regular_permutation,
)
from d4_orqb.train import (
    ANNULAR_HAAR_BASE_CHECKPOINT_SHA256,
    OOF_DEVELOPMENT_MANIFEST_SHA256,
    OOF_FULL_HALF_MEMBERSHIP_SHA256,
    configure_meridional_optimization_and_report,
    meridional_readout_initialization_record,
    meridional_readout_probability_replay,
    remap_haar_to_meridional_readout,
    validate_meridional_flag_contract,
    validate_meridional_source_contract,
)


class MeridionalReadoutTests(unittest.TestCase):
    @staticmethod
    def _haar_model(meridional_readout: bool = False) -> D4OrbitClassifier:
        return D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            core="quantum",
            heads=4,
            reuploads=2,
            meridional_readout=meridional_readout,
            dropout=0.0,
        )

    def test_symmetric_xz_matches_dense_pauli_algebra(self):
        torch.manual_seed(101)
        core = D4OrbitQuantumBottleneck(heads=4, reuploads=2)
        real = torch.randn(2, 256)
        imaginary = torch.randn(2, 256)
        state = torch.complex(real, imaginary)
        state = state / state.norm(dim=1, keepdim=True)
        edges = R_EDGES[:3]
        actual = _expectation_symmetric_xz(
            state, edges, 8, core.z_signs
        )

        identity = torch.eye(2, dtype=torch.complex64)
        pauli_x = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64
        )
        pauli_z = torch.tensor(
            [[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64
        )
        expected = []
        for a, b in edges:
            operators = []
            for x_wire, z_wire in ((a, b), (b, a)):
                operator = torch.ones(1, 1, dtype=torch.complex64)
                for wire in range(8):
                    factor = (
                        pauli_x
                        if wire == x_wire
                        else (pauli_z if wire == z_wire else identity)
                    )
                    operator = torch.kron(operator, factor)
                operators.append(operator)
            dense = operators[0] + operators[1]
            transformed = state @ dense.transpose(0, 1)
            expected.append((state.conj() * transformed).sum(dim=1).real)
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

        with self.assertRaises(ValueError):
            _expectation_symmetric_xz(state, ((0, 0),), 8, core.z_signs)

    def test_zero_replay_all_phase_gradients_and_d4_covariance(self):
        torch.manual_seed(103)
        source = D4OrbitQuantumBottleneck(heads=4, reuploads=2)
        target = D4OrbitQuantumBottleneck(
            heads=4, reuploads=2, meridional_readout=True
        )
        target.params.data.copy_(source.params.data)
        report = target.parameter_report()
        self.assertEqual(report["quantum_trainable"], 104)
        self.assertEqual(report["state_preparation_trainable"], 88)
        self.assertEqual(report["meridional_readout_trainable"], 16)
        self.assertEqual(report["equatorial_readout_trainable"], 0)
        self.assertTrue(
            torch.equal(
                target.meridional_phases,
                torch.zeros_like(target.meridional_phases),
            )
        )

        angles = torch.randn(3, 4, 2, 8)
        source_features = source(angles)
        target_features = target(angles)
        self.assertTrue(torch.equal(target_features, source_features))

        weights = torch.linspace(
            0.1, 2.0, target_features.numel()
        ).reshape_as(target_features)
        (target_features * weights).sum().backward()
        gradients = target.meridional_phases.grad
        self.assertIsNotNone(gradients)
        self.assertTrue(torch.isfinite(gradients).all())
        self.assertTrue((gradients.abs() > 1e-8).all())

        target.zero_grad(set_to_none=True)
        with torch.no_grad():
            target.meridional_phases.normal_(mean=0.0, std=0.2)
        reference, reference_equivariant = target(
            angles, return_equivariant=True
        )
        self.assertEqual(
            tuple(reference_equivariant["meridional"].shape),
            (3, 4, 4, 8),
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
            for observable in ("z", "x", "meridional"):
                expected = reference_equivariant[observable].index_select(
                    -1, permutation
                )
                torch.testing.assert_close(
                    transformed_equivariant[observable],
                    expected,
                    rtol=2e-5,
                    atol=2e-5,
                )

    def test_exact_budget_gauge_remap_replay_and_frozen_report(self):
        torch.manual_seed(107)
        source = self._haar_model(False)
        target = self._haar_model(True)
        source_report = source.parameter_report()
        target_report = target.parameter_report()
        self.assertEqual(source_report["total"], 122_595)
        self.assertEqual(source_report["quantum"], 88)
        self.assertEqual(target_report["total"], 122_610)
        self.assertEqual(target_report["quantum"], 104)
        self.assertEqual(target_report["quantum_state_preparation_trainable"], 88)
        self.assertEqual(target_report["meridional_readout_trainable"], 16)
        self.assertEqual(target_report["equatorial_readout_trainable"], 0)
        self.assertEqual(target_report["head_and_context"], 1_304)
        self.assertEqual(target_report["classifier_bias_trainable"], 2)
        self.assertEqual(target_report["classifier_bias_gauge_degrees"], 2)

        source_state = source.state_dict()
        remapped, adaptations = remap_haar_to_meridional_readout(
            source_state, target.state_dict()
        )
        self.assertEqual(
            [item["method"] for item in adaptations],
            [
                "zero-new-d4-meridional-measurement-phases",
                "remove-softmax-common-logit-gauge",
            ],
        )
        self.assertEqual(
            adaptations[0]["observable"],
            "P(phi)=cos(phi)X+sin(phi)Z",
        )
        self.assertTrue(
            torch.equal(
                remapped["core.meridional_phases"], torch.zeros(4, 4)
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
        record = meridional_readout_initialization_record(target)
        self.assertTrue(record["all_phases_zero_after_remap"])
        self.assertEqual(record["mixed_pair_sector"], ["XZ", "ZX"])

        core_angles = torch.randn(2, 4, 2, 8)
        self.assertTrue(
            torch.equal(source.core(core_angles), target.core(core_angles))
        )
        images = torch.rand(1, 1, 32, 32)
        replay = meridional_readout_probability_replay(
            source, target, images
        )
        for mode in ("float32", "bfloat16_autocast"):
            self.assertTrue(replay[mode]["features_bitwise_equal"])
            self.assertTrue(replay[mode]["predictions_equal"])
            self.assertTrue(
                replay[mode]["probabilities_equal_within_tolerance"]
            )
        self.assertEqual(
            replay["bfloat16_autocast"]["probability_tolerance"], 2e-3
        )

        trainable_report = configure_meridional_optimization_and_report(
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
            trainable_report["meridional_readout_optimization_trainable"], 16
        )

        frozen = self._haar_model(True)
        frozen.load_state_dict(remapped, strict=True)
        frozen_report = configure_meridional_optimization_and_report(
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
            frozen_report["meridional_readout_optimization_trainable"], 0
        )
        self.assertFalse(frozen.core.meridional_phases.requires_grad)

        wrong = self._haar_model(False)
        with self.assertRaises(ValueError):
            configure_meridional_optimization_and_report(wrong, False)
        changed = self._haar_model(True)
        changed.load_state_dict(remapped, strict=True)
        changed.core.meridional_phases.data[0, 0] = 0.1
        with self.assertRaises(RuntimeError):
            configure_meridional_optimization_and_report(changed, False)

    def test_checkpoint_data_extension_and_test_locks_fail_closed(self):
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
        source_summary = {"official_test_evaluated": False}
        target_data = copy.deepcopy(source_data)
        record = validate_meridional_source_contract(
            ANNULAR_HAAR_BASE_CHECKPOINT_SHA256,
            config,
            parameters,
            source_data,
            source_summary,
            target_data,
        )
        self.assertEqual(
            record["checkpoint_sha256"],
            ANNULAR_HAAR_BASE_CHECKPOINT_SHA256,
        )
        self.assertTrue(record["same_training_membership"])
        self.assertFalse(record["source_official_test_opened"])

        mutations = []
        mutations.append(("checkpoint_sha256", "0" * 64))
        for extension in (
            "meridional_readout",
            "equatorial_readout",
            "r2_entanglers",
        ):
            bad = copy.deepcopy(config)
            bad[extension] = True
            mutations.append(("source_config", bad))
        bad_parameters = copy.deepcopy(parameters)
        bad_parameters["total"] = 122610
        mutations.append(("source_parameters", bad_parameters))
        bad_source_data = copy.deepcopy(source_data)
        bad_source_data["official_test_cache_opened"] = True
        mutations.append(("source_data", bad_source_data))
        mutations.append(
            ("source_summary", {"official_test_evaluated": True})
        )
        bad_target_data = copy.deepcopy(target_data)
        bad_target_data["train_membership_sha256"] = "bad"
        mutations.append(("target_data", bad_target_data))
        opened_target = copy.deepcopy(target_data)
        opened_target["test"] = {"classes": []}
        mutations.append(("target_data", opened_target))

        for field, value in mutations:
            with self.subTest(field=field, value=str(value)[:50]):
                arguments = {
                    "checkpoint_sha256": ANNULAR_HAAR_BASE_CHECKPOINT_SHA256,
                    "source_config": config,
                    "source_parameters": parameters,
                    "source_data": source_data,
                    "source_summary": source_summary,
                    "target_data": target_data,
                }
                arguments[field] = value
                with self.assertRaises(RuntimeError):
                    validate_meridional_source_contract(**arguments)

    def test_extension_flags_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            D4OrbitQuantumBottleneck(
                meridional_readout=True, equatorial_readout=True
            )
        with self.assertRaises(ValueError):
            D4OrbitQuantumBottleneck(
                meridional_readout=True, r2_entanglers=True
            )
        with self.assertRaises(ValueError):
            D4OrbitClassifier(
                encoder_variant="deep-se-haar-morph",
                physics_variant="base",
                physics_summary="moments-morphology-haar",
                core="quantum",
                heads=4,
                reuploads=2,
                meridional_readout=True,
                equatorial_readout=True,
            )

        clean = SimpleNamespace(
            meridional_readout=True,
            freeze_meridional_readout_at_zero=False,
            tied_mean_dispersion=False,
            haar_subtype_residual=False,
            shared_late_refinement=False,
            r2_entanglers=False,
            equatorial_readout=False,
            reinitialize_core_after_init=False,
        )
        validate_meridional_flag_contract(clean)
        orphan_freeze = copy.copy(clean)
        orphan_freeze.meridional_readout = False
        orphan_freeze.freeze_meridional_readout_at_zero = True
        with self.assertRaises(ValueError):
            validate_meridional_flag_contract(orphan_freeze)
        for conflict in ("equatorial_readout", "r2_entanglers"):
            combined = copy.copy(clean)
            setattr(combined, conflict, True)
            with self.subTest(conflict=conflict), self.assertRaises(ValueError):
                validate_meridional_flag_contract(combined)


if __name__ == "__main__":
    unittest.main()
