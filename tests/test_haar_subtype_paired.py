import copy
import unittest
from types import SimpleNamespace

import torch

from d4_orqb.model import D4OrbitClassifier
from d4_orqb.train import (
    HAAR_SUBTYPE_SELECTION_SPEC_SHA256,
    configure_haar_subtype_optimization_and_report,
    haar_subtype_residual_update_record,
    remap_haar_to_subtype_residual,
    validate_haar_subtype_freeze_contract,
)


class HaarSubtypePairedTests(unittest.TestCase):
    @staticmethod
    def _model(*, max_envelope: bool = False) -> D4OrbitClassifier:
        return D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            core="quantum",
            heads=4,
            reuploads=2,
            haar_subtype_residual=True,
            haar_subtype_max_envelope=max_envelope,
            dropout=0.0,
        )

    def test_flag_requires_plain_residual_and_never_freezes_base(self):
        clean = SimpleNamespace(
            freeze_haar_subtype_residual_at_zero=True,
            haar_subtype_residual=True,
            haar_subtype_max_envelope=False,
            freeze_base_for_haar_subtype_residual=False,
        )
        validate_haar_subtype_freeze_contract(clean)

        orphan = copy.copy(clean)
        orphan.haar_subtype_residual = False
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_haar_subtype_freeze_contract(orphan)

        envelope = copy.copy(clean)
        envelope.haar_subtype_max_envelope = True
        with self.assertRaisesRegex(ValueError, "max envelope"):
            validate_haar_subtype_freeze_contract(envelope)

        frozen_base = copy.copy(clean)
        frozen_base.freeze_base_for_haar_subtype_residual = True
        with self.assertRaisesRegex(ValueError, "freezes only the residual"):
            validate_haar_subtype_freeze_contract(frozen_base)

    def test_allocated_and_optimizer_budgets_for_candidate_and_control(self):
        candidate = self._model()
        candidate_report = configure_haar_subtype_optimization_and_report(
            candidate, freeze_at_zero=False
        )
        self.assertEqual(candidate_report["total"], 122_610)
        self.assertEqual(candidate_report["inference_total"], 122_610)
        self.assertEqual(candidate_report["quantum"], 88)
        self.assertEqual(candidate_report["inference_quantum"], 88)
        self.assertEqual(
            candidate_report["optimization_trainable_total"], 122_610
        )
        self.assertEqual(
            candidate_report["haar_subtype_residual_trainable"], 15
        )
        self.assertEqual(
            candidate_report[
                "haar_subtype_residual_optimization_trainable"
            ],
            15,
        )
        self.assertFalse(
            candidate_report["haar_subtype_residual_frozen_at_zero"]
        )

        control = self._model()
        control_report = configure_haar_subtype_optimization_and_report(
            control, freeze_at_zero=True
        )
        self.assertEqual(control_report["total"], 122_610)
        self.assertEqual(control_report["inference_total"], 122_610)
        self.assertEqual(control_report["quantum"], 88)
        self.assertEqual(
            control_report["optimization_trainable_total"], 122_595
        )
        self.assertEqual(
            control_report["haar_subtype_residual_trainable"], 15
        )
        self.assertEqual(
            control_report[
                "haar_subtype_residual_optimization_trainable"
            ],
            0,
        )
        self.assertTrue(
            control_report["haar_subtype_residual_frozen_at_zero"]
        )
        self.assertFalse(control.haar_subtype_residual.weight.requires_grad)
        self.assertTrue(control.encoder.stem[0].weight.requires_grad)
        self.assertTrue(control.core.params.requires_grad)
        self.assertTrue(control.head.projection.weight.requires_grad)

    def test_endpoint_audit_requires_zero_control_and_updated_candidate(self):
        initial = torch.zeros(15)
        control = self._model()
        configure_haar_subtype_optimization_and_report(
            control, freeze_at_zero=True
        )
        frozen = haar_subtype_residual_update_record(
            control, initial, freeze_at_zero=True
        )
        self.assertEqual(frozen["optimization_trainable"], 0)
        self.assertEqual(frozen["weight_update_l2"], 0.0)
        self.assertTrue(frozen["weights_exact_zero"])

        changed_control = self._model()
        configure_haar_subtype_optimization_and_report(
            changed_control, freeze_at_zero=True
        )
        with torch.no_grad():
            changed_control.haar_subtype_residual.weight[0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "Frozen Haar subtype"):
            haar_subtype_residual_update_record(
                changed_control, initial, freeze_at_zero=True
            )

        candidate = self._model()
        configure_haar_subtype_optimization_and_report(
            candidate, freeze_at_zero=False
        )
        with self.assertRaisesRegex(RuntimeError, "received no parameter update"):
            haar_subtype_residual_update_record(
                candidate, initial, freeze_at_zero=False
            )
        with torch.no_grad():
            candidate.haar_subtype_residual.weight[3] = -0.2
        updated = haar_subtype_residual_update_record(
            candidate, initial, freeze_at_zero=False
        )
        self.assertEqual(updated["optimization_trainable"], 15)
        self.assertGreater(updated["weight_update_l2"], 0.0)
        self.assertFalse(updated["weights_exact_zero"])

    def test_base_remap_is_zero_and_selection_spec_is_locked(self):
        source = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            core="quantum",
            heads=4,
            reuploads=2,
            dropout=0.0,
        )
        target = self._model()
        remapped, adaptations = remap_haar_to_subtype_residual(
            source.state_dict(), target.state_dict()
        )
        self.assertEqual(
            adaptations[0]["method"],
            "zero-new-invariant-haar-subtype-residual",
        )
        self.assertTrue(
            torch.equal(
                remapped["haar_subtype_residual.weight"], torch.zeros(15)
            )
        )
        self.assertEqual(
            HAAR_SUBTYPE_SELECTION_SPEC_SHA256,
            "8c8aca6a0cf66ce4d547ec746851c7c698fbc9fe85b6262d70c94085d098d470",
        )

        with self.assertRaises(ValueError):
            configure_haar_subtype_optimization_and_report(
                self._model(max_envelope=True), freeze_at_zero=False
            )
        changed = self._model()
        with torch.no_grad():
            changed.haar_subtype_residual.weight[0] = 0.01
        with self.assertRaisesRegex(RuntimeError, "changed before optimizer"):
            configure_haar_subtype_optimization_and_report(
                changed, freeze_at_zero=False
            )


if __name__ == "__main__":
    unittest.main()
