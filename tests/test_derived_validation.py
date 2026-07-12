import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from d4_orqb.evaluate_derived_validation import (
    DERIVED_PARAMETERS,
    RESIDUAL_STATE_KEYS,
    assert_path_sets_disjoint,
    build_parser,
    canonical_spec_sha256,
    compose_derived_state,
    configure_training_runtime,
    load_sha_locked_checkpoint,
    require_sha256,
    synthetic_envelope_audit,
    validate_output_destination,
)
from d4_orqb.model import D4OrbitClassifier, max_preserving_subtype_envelope


class DerivedValidationTests(unittest.TestCase):
    def test_training_runtime_replays_frozen_cuda_backend_contract(self):
        previous = {
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
            "precision": torch.get_float32_matmul_precision(),
        }
        try:
            configure_training_runtime(0)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous["cuda_tf32"]
            torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
            torch.backends.cudnn.benchmark = previous["benchmark"]
            torch.backends.cudnn.deterministic = previous["deterministic"]
            torch.set_float32_matmul_precision(previous["precision"])

    def test_max_envelope_fp32_bfloat16_zero_replay_and_ties(self):
        base_float32 = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [2.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
                [-3.0, -2.0, -2.5],
                [4.0, -5.0, 3.0],
            ]
        )
        delta_float32 = torch.tensor([9.0, -7.0, -8.0, 0.5, 6.0, -20.0])
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                base = base_float32.to(dtype)
                delta = delta_float32.to(dtype)
                replay = max_preserving_subtype_envelope(
                    base, torch.zeros_like(delta)
                )
                self.assertTrue(torch.equal(replay, base))
                adjusted = max_preserving_subtype_envelope(base, delta)
                self.assertTrue(
                    torch.equal(
                        adjusted[:, :2].amax(1), base[:, :2].amax(1)
                    )
                )
                self.assertTrue(
                    torch.equal(adjusted.argmax(1) == 2, base.argmax(1) == 2)
                )
                self.assertTrue(torch.equal(adjusted[:, 2], base[:, 2]))

        audit = synthetic_envelope_audit()
        self.assertTrue(audit["float32"]["zero_delta_bitwise_replay"])
        self.assertTrue(audit["bfloat16"]["no_sub_argmax_indicator_exact"])

    def test_max_envelope_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            max_preserving_subtype_envelope(torch.zeros(2, 4), torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "shape"):
            max_preserving_subtype_envelope(torch.zeros(2, 3), torch.zeros(2, 1))
        with self.assertRaisesRegex(RuntimeError, "finite"):
            max_preserving_subtype_envelope(
                torch.tensor([[0.0, float("nan"), 1.0]]), torch.zeros(1)
            )

    def test_selective_state_composition_never_copies_donor_base(self):
        torch.manual_seed(101)
        primary_model = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            dropout=0.0,
        )
        target_model = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            haar_subtype_residual=True,
            haar_subtype_max_envelope=True,
            dropout=0.0,
        )
        primary = {
            key: value.detach().clone()
            for key, value in primary_model.state_dict().items()
        }
        donor = {
            key: (
                target_model.state_dict()[key].detach().clone()
                if key in RESIDUAL_STATE_KEYS
                else primary[key].detach().clone()
            )
            for key in target_model.state_dict()
        }
        donor["haar_subtype_residual.weight"] = torch.linspace(-0.2, 0.2, 15)
        donor["encoder.stem.0.weight"].add_(100.0)
        composed = compose_derived_state(
            primary, donor, target_model.state_dict()
        )
        for key, value in primary.items():
            self.assertTrue(torch.equal(composed[key], value), msg=key)
        for key in RESIDUAL_STATE_KEYS:
            self.assertTrue(torch.equal(composed[key], donor[key]), msg=key)
        self.assertFalse(
            torch.equal(
                composed["encoder.stem.0.weight"],
                donor["encoder.stem.0.weight"],
            )
        )

        bad_donor = dict(donor)
        bad_donor["unexpected"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "tensor count drifted"):
            compose_derived_state(primary, bad_donor, target_model.state_dict())

    def test_derived_architecture_has_exact_budget_and_flag_contract(self):
        model = D4OrbitClassifier(
            encoder_variant="deep-se-haar-morph",
            physics_summary="moments-morphology-haar",
            haar_subtype_residual=True,
            haar_subtype_max_envelope=True,
        )
        report = model.parameter_report()
        self.assertEqual(report["total"], DERIVED_PARAMETERS)
        self.assertEqual(report["haar_subtype_residual_trainable"], 15)
        self.assertTrue(report["haar_subtype_max_envelope"])
        with self.assertRaisesRegex(ValueError, "requires the Haar subtype residual"):
            D4OrbitClassifier(haar_subtype_max_envelope=True)

    def test_sha_locked_checkpoint_and_file_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "last.pt"
            torch.save(
                {
                    "model": {"weight": torch.tensor([1.0])},
                    "epoch": 20,
                    "record": {"epoch": 20},
                },
                checkpoint_path,
            )
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            self.assertEqual(
                require_sha256(checkpoint_path, digest, "fixture")["sha256"],
                digest,
            )
            checkpoint, state, _ = load_sha_locked_checkpoint(
                checkpoint_path, digest, 20, "fixture checkpoint"
            )
            self.assertEqual(checkpoint["epoch"], 20)
            self.assertTrue(torch.equal(state["weight"], torch.tensor([1.0])))
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                require_sha256(checkpoint_path, "0" * 64, "fixture")
            with self.assertRaisesRegex(RuntimeError, "epoch drift"):
                load_sha_locked_checkpoint(
                    checkpoint_path, digest, 19, "fixture checkpoint"
                )

    def test_selection_digest_output_refusal_and_cli_surface(self):
        selection = {"b": 2, "a": [1, 3]}
        digest = canonical_spec_sha256(selection, "selection_spec_sha256")
        selection["selection_spec_sha256"] = digest
        self.assertEqual(
            canonical_spec_sha256(selection, "selection_spec_sha256"), digest
        )
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("test", destinations)
        self.assertNotIn("test_cache", destinations)
        self.assertNotIn("test_root", destinations)
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "overwrite or resume"):
                validate_output_destination(existing)
            protected = existing / "protected"
            protected.mkdir()
            with self.assertRaisesRegex(RuntimeError, "disjoint"):
                assert_path_sets_disjoint(
                    protected / "derived", {"source": protected}
                )


if __name__ == "__main__":
    unittest.main()
