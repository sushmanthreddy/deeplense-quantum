import unittest

import torch
import torch.nn.functional as F

from d4_orqb.model import (
    D4OrbitClassifier,
    PhysicsChannelBank,
    cross_scale_scattering_summary,
    d4_transform,
    d4_views,
)
from d4_orqb.quantum import D4_ELEMENTS, right_regular_permutation


class CrossScaleReuploadTests(unittest.TestCase):
    @staticmethod
    def _model(
        core: str = "quantum", *, cross_scale_reupload: bool = True
    ) -> D4OrbitClassifier:
        return D4OrbitClassifier(
            num_classes=3,
            heads=4,
            reuploads=2,
            core=core,
            include_context=False,
            dropout=0.0,
            encoder_variant="deep-se-haar-morph",
            physics_variant="base",
            physics_summary="moments-morphology-haar",
            quantum_encoding="angle",
            observable_readout="pair",
            cross_scale_reupload=cross_scale_reupload,
        )

    def test_cross_scale_summary_definition_shape_finite_and_view_covariance(self):
        torch.manual_seed(101)
        images = torch.rand(2, 1, 32, 32)
        bank = PhysicsChannelBank(variant="base")

        physics = bank(images)
        summary = cross_scale_scattering_summary(physics)
        self.assertEqual(tuple(summary.shape), (2, 32))
        self.assertTrue(torch.isfinite(summary).all())

        # Coefficient zero is path (1,2), the H/V family, annulus [4,8).
        # Recompute it independently to pin the second-order cascade and the
        # feature ordering rather than testing only the output width.
        field = physics[:, 1]
        height = field.shape[-1]

        def centered_modulus(
            values: torch.Tensor, offset: int, dy: int, dx: int
        ) -> torch.Tensor:
            padded = F.pad(
                values[:, None],
                (offset, offset, offset, offset),
                mode="reflect",
            )[:, 0]
            return (
                padded[
                    :,
                    offset + dy : offset + dy + height,
                    offset + dx : offset + dx + height,
                ]
                - padded[
                    :,
                    offset - dy : offset - dy + height,
                    offset - dx : offset - dx + height,
                ]
            ).abs()

        first_h = centered_modulus(field, 1, 0, 1)
        first_v = centered_modulus(field, 1, 1, 0)
        second_h = centered_modulus(first_h, 2, 0, 2)
        second_v = centered_modulus(first_v, 2, 2, 0)
        axis_response = 0.5 * (
            torch.log1p(8.0 * second_h) + torch.log1p(8.0 * second_v)
        )
        coordinate = torch.arange(32, dtype=axis_response.dtype) - 15.5
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        first_annulus = (radius >= 4.0 * 32 / 96) & (
            radius < 8.0 * 32 / 96
        )
        expected = axis_response[:, first_annulus].mean(dim=1)
        torch.testing.assert_close(summary[:, 0], expected, rtol=2e-7, atol=2e-7)

        def view_summaries(value: torch.Tensor) -> torch.Tensor:
            views = d4_views(value)
            batch, group, channels, height, width = views.shape
            flat = views.reshape(batch * group, channels, height, width)
            return cross_scale_scattering_summary(bank(flat)).reshape(
                batch, group, 32
            )

        reference = view_summaries(images[:1])
        for element in D4_ELEMENTS:
            actual = view_summaries(d4_transform(images[:1], *element))
            permutation = right_regular_permutation(element)
            torch.testing.assert_close(
                actual,
                reference.index_select(1, permutation),
                rtol=0.0,
                atol=0.0,
            )

    def test_parameter_budget_fixed_walsh_buffers_and_matched_cores(self):
        base = self._model(cross_scale_reupload=False)
        self.assertEqual(base.parameter_report()["total"], 122_595)
        self.assertFalse(base.parameter_report()["cross_scale_reupload"])
        self.assertEqual(
            base.parameter_report()["cross_scale_reupload_gate_trainable"], 0
        )

        for core in ("quantum", "classical"):
            model = self._model(core)
            report = model.parameter_report()
            self.assertEqual(report["total"], 122_599)
            self.assertEqual(report["core"], 88)
            self.assertEqual(report["quantum"], 88 if core == "quantum" else 0)
            self.assertEqual(
                report["parallel_classical"], 0 if core == "quantum" else 88
            )
            self.assertTrue(report["cross_scale_reupload"])
            self.assertEqual(report["cross_scale_reupload_gate_trainable"], 4)
            self.assertEqual(report["cross_scale_scattering_dim"], 32)
            self.assertEqual(report["cross_scale_walsh_channels"], 8)
            self.assertEqual(tuple(model.cross_scale_mean.shape), (32,))
            self.assertEqual(tuple(model.cross_scale_scale.shape), (32,))
            self.assertEqual(tuple(model.cross_scale_walsh.shape), (8, 32))
            self.assertTrue(
                torch.equal(
                    model.cross_scale_reupload_gates,
                    torch.zeros_like(model.cross_scale_reupload_gates),
                )
            )
            torch.testing.assert_close(
                model.cross_scale_walsh @ model.cross_scale_walsh.T,
                torch.eye(8),
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertEqual(
                [key for key in model.state_dict() if "cross_scale" in key],
                [
                    "cross_scale_reupload_gates",
                    "cross_scale_mean",
                    "cross_scale_scale",
                    "cross_scale_walsh",
                ],
            )

        model = self._model()
        mean = torch.linspace(-1.0, 1.0, 32)
        scale = torch.linspace(0.5, 1.5, 32)
        model.set_cross_scale_normalization(mean, scale)
        torch.testing.assert_close(model.cross_scale_mean, mean)
        torch.testing.assert_close(model.cross_scale_scale, scale)
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            model.set_cross_scale_normalization(torch.zeros(31), torch.ones(31))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            model.set_cross_scale_normalization(torch.zeros(32), torch.zeros(32))
        with self.assertRaisesRegex(RuntimeError, "no cross-scale"):
            base.set_cross_scale_normalization(torch.zeros(32), torch.ones(32))

    def test_zero_gate_exact_logits_probabilities_base_gradients_and_gate_gradients(self):
        kwargs_seed = 20260813
        torch.manual_seed(kwargs_seed)
        base = self._model(cross_scale_reupload=False)
        torch.manual_seed(kwargs_seed)
        candidate = self._model(cross_scale_reupload=True)
        base.eval()
        candidate.eval()

        candidate_state = candidate.state_dict()
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, candidate_state[name]), name)

        torch.manual_seed(19)
        images = torch.rand(1, 1, 32, 32)
        base_logits = base(images)
        candidate_logits, auxiliary = candidate(images, return_aux=True)
        self.assertTrue(torch.equal(candidate_logits, base_logits))
        self.assertTrue(
            torch.equal(candidate_logits.softmax(1), base_logits.softmax(1))
        )
        self.assertTrue(
            torch.equal(
                auxiliary["cross_scale_delta"],
                torch.zeros_like(auxiliary["cross_scale_delta"]),
            )
        )
        self.assertTrue(
            torch.equal(
                auxiliary["base_invariants"], auxiliary["perturbed_invariants"]
            )
        )
        self.assertEqual(tuple(auxiliary["cross_scale_summary"].shape), (1, 8, 32))
        self.assertEqual(tuple(auxiliary["cross_scale_tokens"].shape), (1, 4, 2, 8))

        loss_weights = torch.tensor([[0.75, -1.25, 0.5]])
        base.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        (base_logits * loss_weights).sum().backward()
        (candidate_logits * loss_weights).sum().backward()
        candidate_parameters = dict(candidate.named_parameters())
        for name, parameter in base.named_parameters():
            candidate_parameter = candidate_parameters[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(candidate_parameter.grad, name)
            self.assertTrue(
                torch.equal(parameter.grad, candidate_parameter.grad), name
            )
        gate_gradient = candidate.cross_scale_reupload_gates.grad
        self.assertIsNotNone(gate_gradient)
        self.assertTrue(torch.isfinite(gate_gradient).all())
        self.assertTrue((gate_gradient != 0).all())

    def test_nonzero_reupload_is_d4_invariant_for_quantum_and_classical(self):
        torch.manual_seed(23)
        images = torch.rand(1, 1, 32, 32)
        for core in ("quantum", "classical"):
            torch.manual_seed(29)
            model = self._model(core)
            model.eval()
            with torch.no_grad():
                model.cross_scale_reupload_gates.copy_(
                    torch.tensor((-0.4, -0.1, 0.2, 0.5))
                )
                reference_logits, reference_auxiliary = model(
                    images, return_aux=True
                )
                for element in D4_ELEMENTS:
                    actual_logits, actual_auxiliary = model(
                        d4_transform(images, *element), return_aux=True
                    )
                    permutation = right_regular_permutation(element)
                    torch.testing.assert_close(
                        actual_auxiliary["cross_scale_tokens"],
                        reference_auxiliary["cross_scale_tokens"].index_select(
                            -1, permutation
                        ),
                        rtol=0.0,
                        atol=0.0,
                    )
                    torch.testing.assert_close(
                        actual_auxiliary["cross_scale_delta"],
                        reference_auxiliary["cross_scale_delta"].index_select(
                            -1, permutation
                        ),
                        rtol=0.0,
                        atol=0.0,
                    )
                    torch.testing.assert_close(
                        actual_logits,
                        reference_logits,
                        rtol=5e-5,
                        atol=5e-5,
                    )

    def test_constructor_rejects_noncanonical_and_extension_combinations(self):
        canonical = {
            "num_classes": 3,
            "heads": 4,
            "reuploads": 2,
            "core": "quantum",
            "include_context": False,
            "encoder_variant": "deep-se-haar-morph",
            "physics_variant": "base",
            "physics_summary": "moments-morphology-haar",
            "quantum_encoding": "angle",
            "observable_readout": "pair",
            "cross_scale_reupload": True,
        }
        invalid_changes = (
            {"num_classes": 2},
            {"heads": 3},
            {"reuploads": 3},
            {"core": "hybrid"},
            {"include_context": True},
            {"encoder_variant": "deep-se-morph", "physics_summary": "moments-morphology"},
            {"physics_variant": "radial"},
            {"physics_summary": "moments"},
            {"quantum_encoding": "gibbs"},
            {"observable_readout": "plaquette"},
            {"tied_mean_dispersion": True},
            {"haar_subtype_residual": True},
            {"shared_late_refinement": True},
            {"r2_entanglers": True},
            {"equatorial_readout": True},
            {"meridional_readout": True},
        )
        for change in invalid_changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                D4OrbitClassifier(**{**canonical, **change})


if __name__ == "__main__":
    unittest.main()
