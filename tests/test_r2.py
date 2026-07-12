import unittest

import torch

from d4_orqb.model import D4OrbitClassifier
from d4_orqb.quantum import (
    D4_ELEMENTS,
    D4OrbitQuantumBottleneck,
    right_regular_permutation,
)
from d4_orqb.train import (
    configure_r2_optimization_and_report,
    remap_haar_to_r2_entanglers,
    r2_entangler_initialization_record,
    r2_entangler_probability_replay,
)


class R2EntanglerTests(unittest.TestCase):
    def test_r2_core_is_trainable_invariant_and_zero_start(self):
        torch.manual_seed(7)
        core = D4OrbitQuantumBottleneck(
            heads=4, reuploads=2, r2_entanglers=True
        )
        self.assertEqual(core.parameter_report()["quantum_trainable"], 104)
        self.assertEqual(core.parameter_report()["r2_entangler_trainable"], 16)
        self.assertTrue(torch.equal(core.r2_params, torch.zeros_like(core.r2_params)))

        angles = torch.randn(2, 4, 2, 8)
        with torch.no_grad():
            core.r2_params.normal_(mean=0.0, std=0.1)
        reference, reference_equivariant = core(
            angles, return_equivariant=True
        )
        for element in D4_ELEMENTS:
            permutation = right_regular_permutation(element)
            transformed, transformed_equivariant = core(
                angles.index_select(-1, permutation),
                return_equivariant=True,
            )
            torch.testing.assert_close(
                transformed, reference, rtol=2e-5, atol=2e-5
            )
            for observable in ("z", "x"):
                expected = reference_equivariant[observable].index_select(
                    -1, permutation
                )
                torch.testing.assert_close(
                    transformed_equivariant[observable],
                    expected,
                    rtol=2e-5,
                    atol=2e-5,
                )

        core.r2_params.data.zero_()
        loss = core(angles).square().sum()
        loss.backward()
        self.assertIsNotNone(core.r2_params.grad)
        self.assertTrue(torch.isfinite(core.r2_params.grad).all())
        self.assertGreater(float(core.r2_params.grad.norm()), 0.0)

    def test_r2_exact_budget_bias_gauge_and_probability_replay(self):
        torch.manual_seed(11)
        source = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            core="quantum",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        target = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            core="quantum",
            heads=4,
            reuploads=2,
            r2_entanglers=True,
            dropout=0.0,
        )
        source_report = source.parameter_report()
        target_report = target.parameter_report()
        self.assertEqual(source_report["total"], 122_595)
        self.assertEqual(source_report["quantum"], 88)
        self.assertEqual(target_report["total"], 122_610)
        self.assertEqual(target_report["quantum"], 104)
        self.assertEqual(target_report["head_and_context"], 1_304)
        self.assertEqual(target_report["classifier_bias_trainable"], 2)
        self.assertEqual(target_report["classifier_bias_gauge_degrees"], 2)

        source_state = source.state_dict()
        remapped, adaptations = remap_haar_to_r2_entanglers(
            source_state, target.state_dict()
        )
        self.assertEqual(
            [item["method"] for item in adaptations],
            [
                "zero-new-r2-edge-zz-xx-entanglers",
                "remove-softmax-common-logit-gauge",
            ],
        )
        self.assertTrue(
            torch.equal(remapped["core.r2_params"], torch.zeros(4, 2, 2))
        )
        expected_bias = (
            source_state["head.classifier.bias"][:2]
            - source_state["head.classifier.bias"][2]
        )
        torch.testing.assert_close(
            remapped["head.classifier.bias"], expected_bias, rtol=0.0, atol=0.0
        )
        target.load_state_dict(remapped, strict=True)
        record = r2_entangler_initialization_record(target)
        self.assertTrue(record["all_angles_zero_after_remap"])

        images = torch.rand(1, 1, 32, 32)
        source.eval()
        target.eval()
        with torch.no_grad():
            source_logits = source(images)
            target_logits = target(images)
        torch.testing.assert_close(
            target_logits.softmax(1),
            source_logits.softmax(1),
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertTrue(
            torch.equal(target_logits.argmax(1), source_logits.argmax(1))
        )
        shift = target_logits - source_logits
        torch.testing.assert_close(
            shift, shift[:, :1].expand_as(shift), rtol=2e-6, atol=2e-6
        )
        replay = r2_entangler_probability_replay(source, target, images)
        self.assertTrue(replay["float32"]["predictions_equal"])
        self.assertTrue(
            replay["float32"]["probabilities_equal_within_tolerance"]
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

        trainable_report = configure_r2_optimization_and_report(
            target, freeze_at_zero=False
        )
        self.assertEqual(trainable_report["inference_total"], 122_610)
        self.assertEqual(
            trainable_report["optimization_trainable_total"], 122_610
        )
        self.assertEqual(trainable_report["quantum_optimization_trainable"], 104)

        frozen = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            core="quantum",
            heads=4,
            reuploads=2,
            r2_entanglers=True,
            dropout=0.0,
        )
        frozen.load_state_dict(remapped, strict=True)
        frozen_report = configure_r2_optimization_and_report(
            frozen, freeze_at_zero=True
        )
        self.assertEqual(frozen_report["total"], 122_610)
        self.assertEqual(frozen_report["quantum"], 104)
        self.assertEqual(frozen_report["inference_total"], 122_610)
        self.assertEqual(
            frozen_report["optimization_trainable_total"], 122_594
        )
        self.assertEqual(frozen_report["quantum_optimization_trainable"], 88)
        self.assertEqual(frozen_report["r2_entangler_optimization_trainable"], 0)
        self.assertFalse(frozen.core.r2_params.requires_grad)


if __name__ == "__main__":
    unittest.main()
