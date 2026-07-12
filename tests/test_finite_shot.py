import unittest

import torch

from d4_orqb.finite_shot import (
    _infer_architecture,
    _normalize_probabilities,
    analytic_invariants_from_probabilities,
    finite_shot_invariants,
    hadamard_all,
    logits_from_invariants,
    outcomes_to_signs,
    sample_joint_bitstrings,
    stratified_paired_accuracy_bootstrap,
    z_and_x_probabilities,
)
from d4_orqb.model import D4OrbitClassifier
from d4_orqb.quantum import D4OrbitQuantumBottleneck, R_EDGES


class FiniteShotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_finite_shot_loader_rejects_shared_late_refinement(self):
        with self.assertRaisesRegex(RuntimeError, "shared late refinement"):
            _infer_architecture({"shared_late_refinement": True}, {})
        with self.assertRaisesRegex(RuntimeError, "shared late refinement"):
            _infer_architecture(
                {}, {"encoder.shared_refinement_gates": torch.zeros(4)}
            )

    def test_finite_shot_loader_rejects_haar_subtype_residual(self):
        with self.assertRaisesRegex(RuntimeError, "max-preserving"):
            _infer_architecture({"haar_subtype_max_envelope": True}, {})
        with self.assertRaisesRegex(RuntimeError, "Haar subtype residual"):
            _infer_architecture({"haar_subtype_residual": True}, {})
        with self.assertRaisesRegex(RuntimeError, "Haar subtype residual"):
            _infer_architecture(
                {}, {"haar_subtype_residual.weight": torch.zeros(15)}
            )

    def test_hadamard_all_is_normalized_and_self_inverse(self):
        state = torch.randn(3, 256) + 1j * torch.randn(3, 256)
        state = state / state.norm(dim=1, keepdim=True)
        transformed = hadamard_all(state)
        torch.testing.assert_close(
            transformed.abs().square().sum(1), torch.ones(3), rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(hadamard_all(transformed), state, rtol=2e-5, atol=2e-5)

    def test_probability_normalization_fails_closed_on_invalid_values(self):
        with self.assertRaises(RuntimeError):
            _normalize_probabilities(torch.tensor([[0.5, float("nan")]]))
        with self.assertRaises(RuntimeError):
            _normalize_probabilities(torch.tensor([[1.1, -0.1]]))

    def test_joint_bitstrings_preserve_ghz_z_correlations(self):
        core = D4OrbitQuantumBottleneck(heads=1, reuploads=1)
        probabilities = torch.zeros(1, 256)
        probabilities[0, 0] = 0.5
        probabilities[0, 255] = 0.5
        generator = torch.Generator().manual_seed(91)
        outcomes = sample_joint_bitstrings(probabilities, 1024, generator)
        signs = outcomes_to_signs(outcomes, core.z_signs)
        for a, b in R_EDGES:
            torch.testing.assert_close(
                signs[..., a] * signs[..., b], torch.ones(1, 1024), rtol=0.0, atol=0.0
            )
        self.assertLess(abs(float(signs[..., 0].mean())), 0.10)

    def test_256_shots_are_a_strict_prefix_of_the_1024_draw(self):
        probabilities = torch.full((2, 256), 1.0 / 256)
        maximum = sample_joint_bitstrings(
            probabilities, 1024, torch.Generator().manual_seed(17)
        )
        torch.testing.assert_close(maximum[:, :256], maximum.narrow(1, 0, 256))

    def test_finite_shot_invariants_execute_for_both_estimators(self):
        core = D4OrbitQuantumBottleneck(heads=2, reuploads=1)
        probabilities = torch.full((4, 256), 1.0 / 256)
        z = outcomes_to_signs(
            sample_joint_bitstrings(
                probabilities, 256, torch.Generator().manual_seed(17)
            ),
            core.z_signs,
        )
        x = outcomes_to_signs(
            sample_joint_bitstrings(
                probabilities, 256, torch.Generator().manual_seed(1_000_020)
            ),
            core.z_signs,
        )
        for estimator in ("plugin", "ustat"):
            features = finite_shot_invariants(z, x, heads=2, estimator=estimator)
            self.assertEqual(tuple(features.shape), (2, 24))
            self.assertTrue(torch.isfinite(features).all())

    def test_ustat_quadratic_features_match_distinct_shot_definition(self):
        core = D4OrbitQuantumBottleneck(heads=1, reuploads=1)
        outcomes_z = torch.tensor([[0, 3, 15, 31, 63, 127, 255]], dtype=torch.long)
        outcomes_x = torch.tensor([[1, 2, 4, 8, 16, 32, 64]], dtype=torch.long)
        z = outcomes_to_signs(outcomes_z, core.z_signs)
        x = outcomes_to_signs(outcomes_x, core.z_signs)
        actual = finite_shot_invariants(z, x, heads=1, estimator="ustat")[0]

        def manual_basis(signs):
            shots = signs.shape[1]
            means = signs.mean(1)[0]
            mean_value = means.mean()
            squared_means = []
            for qubit in range(8):
                values = signs[0, :, qubit]
                squared_means.append(
                    sum(
                        float(values[i] * values[j])
                        for i in range(shots)
                        for j in range(shots)
                        if i != j
                    )
                    / (shots * (shots - 1))
                )
            orbit_values = signs[0].mean(1)
            orbit_mean_square = sum(
                float(orbit_values[i] * orbit_values[j])
                for i in range(shots)
                for j in range(shots)
                if i != j
            ) / (shots * (shots - 1))
            return mean_value, sum(squared_means) / 8 - orbit_mean_square

        z_mean, z_variance = manual_basis(z)
        x_mean, x_variance = manual_basis(x)
        self.assertAlmostEqual(float(actual[0]), float(z_mean), places=6)
        self.assertAlmostEqual(float(actual[1]), float(z_variance), places=6)
        self.assertAlmostEqual(float(actual[2]), float(x_mean), places=6)
        self.assertAlmostEqual(float(actual[3]), float(x_variance), places=6)

    def test_stratified_paired_bootstrap_is_deterministic_and_directional(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2]).numpy()
        analytic = labels.copy()
        finite = labels.copy()
        equal_first = stratified_paired_accuracy_bootstrap(
            labels, finite, analytic, samples=1000, seed=123
        )
        equal_second = stratified_paired_accuracy_bootstrap(
            labels, finite, analytic, samples=1000, seed=123
        )
        self.assertEqual(equal_first, equal_second)
        self.assertEqual(equal_first["difference_finite_minus_analytic"], 0.0)
        self.assertTrue(equal_first["noninferior_at_one_sided_95"])

        always_wrong = (labels + 1) % 3
        inferior = stratified_paired_accuracy_bootstrap(
            labels, always_wrong, analytic, samples=1000, seed=123
        )
        self.assertEqual(inferior["two_sided_95_ci_low"], -1.0)
        self.assertEqual(inferior["two_sided_95_ci_high"], -1.0)
        self.assertFalse(inferior["noninferior_at_one_sided_95"])

    def test_probability_analytic_features_match_quantum_core(self):
        core = D4OrbitQuantumBottleneck(heads=2, reuploads=2)
        angles = torch.randn(3, 2, 2, 8)
        expected = core(angles)
        state = core._run_statevector(angles)
        z_probabilities, x_probabilities = z_and_x_probabilities(state)
        actual = analytic_invariants_from_probabilities(
            z_probabilities, x_probabilities, core.z_signs, heads=2
        )
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

    def test_reconstructed_analytic_logits_match_model_forward(self):
        model = D4OrbitClassifier(heads=2, reuploads=1)
        model.eval()
        images = torch.rand(2, 1, 32, 32)
        expected = model(images)
        encoded, angles = model.orbit_encode(images)
        state = model.core._run_statevector(angles)
        z_probabilities, x_probabilities = z_and_x_probabilities(state)
        invariants = analytic_invariants_from_probabilities(
            z_probabilities, x_probabilities, model.core.z_signs, heads=2
        )
        actual = logits_from_invariants(model, invariants, encoded)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
