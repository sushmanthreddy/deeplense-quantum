import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from d4_orqb.cudaq_backend import (
    CudaQD4OrbitBackend,
    CudaQConfigurationError,
    CudaQUnavailableError,
    FIXTURE_SCHEMA,
    HEADS,
    INVARIANTS_PER_HEAD,
    N_QUBITS,
    OBSERVABLE_COUNT,
    OBSERVABLE_SPECS,
    PARAMETERS_PER_LAYER,
    REUPLOADS,
    R2_EDGES,
    R_EDGES,
    S_EDGES,
    _kernel_source,
    _require_cudaq,
    cudaq_exp_pauli_coefficient,
    invariants_from_expectations,
    load_fixture,
    pack_head_arguments,
    validate_arrays,
    validate_base_configuration,
)


class CudaQBackendTests(unittest.TestCase):
    def setUp(self):
        self.features = np.arange(2 * HEADS * 2 * N_QUBITS, dtype=np.float32).reshape(
            2, HEADS, 2, N_QUBITS
        )
        self.parameters = np.arange(
            HEADS * REUPLOADS * PARAMETERS_PER_LAYER, dtype=np.float32
        ).reshape(HEADS, REUPLOADS, PARAMETERS_PER_LAYER)

    def test_edges_and_observables_match_base_pair_readout(self):
        self.assertEqual(
            R_EDGES,
            ((0, 1), (0, 3), (1, 2), (2, 3), (4, 5), (4, 7), (5, 6), (6, 7)),
        )
        self.assertEqual(R2_EDGES, ((0, 2), (1, 3), (4, 6), (5, 7)))
        self.assertEqual(S_EDGES, ((0, 4), (1, 7), (2, 6), (3, 5)))
        self.assertEqual(OBSERVABLE_COUNT, 48)
        self.assertEqual([spec.label for spec in OBSERVABLE_SPECS[:3]], ["Z0", "Z1", "Z2"])
        self.assertEqual(OBSERVABLE_SPECS[0].word, "ZIIIIIII")
        self.assertEqual(OBSERVABLE_SPECS[16].word, "ZZIIIIII")
        self.assertEqual(OBSERVABLE_SPECS[-1].word, "IIIXIXII")

    def test_parameter_and_feature_packing_is_channel_and_layer_major(self):
        packed = pack_head_arguments(self.features, self.parameters)
        self.assertEqual(len(packed), 2 * HEADS)
        features, params = packed[1]
        np.testing.assert_array_equal(features[:8], self.features[0, 1, 0])
        np.testing.assert_array_equal(features[8:], self.features[0, 1, 1])
        np.testing.assert_array_equal(params[:11], self.parameters[1, 0])
        np.testing.assert_array_equal(params[-11:], self.parameters[1, -1])
        self.assertEqual(features.dtype, np.float64)
        self.assertEqual(params.dtype, np.float64)

    def test_pair_rotations_use_exact_cnot_rz_decomposition(self):
        values = np.asarray([-2.0, 0.0, 3.0])
        np.testing.assert_array_equal(cudaq_exp_pauli_coefficient(values), [1.0, -0.0, -1.5])
        source = _kernel_source()
        self.assertNotIn("exp_pauli(", source)
        self.assertIn("x.ctrl(qubits[0], qubits[1])", source)
        self.assertIn("rz(parameters[offset + 7], qubits[1])", source)
        self.assertIn("rz(parameters[offset + 10], qubits[4])", source)
        self.assertEqual(source.count("x.ctrl("), 48)

    def test_invariants_match_manual_pair_statistics(self):
        values = np.zeros((1, HEADS, OBSERVABLE_COUNT), dtype=np.float64)
        z = np.arange(1.0, 9.0)
        x = -np.arange(1.0, 9.0)
        values[..., :8] = z
        values[..., 8:16] = x
        cursor = 16
        family_lengths = [len(R_EDGES), len(R2_EDGES), len(S_EDGES)] * 2
        constants = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]
        for length, constant in zip(family_lengths, constants):
            values[..., cursor : cursor + length] = constant
            cursor += length
        result = invariants_from_expectations(values)
        self.assertEqual(result.shape, (1, HEADS, INVARIANTS_PER_HEAD))
        expected_prefix = [z.mean(), z.var(), x.mean(), x.var(), 2, 3, 5, 7, 11, 13]
        np.testing.assert_allclose(result[0, 0, :10], expected_prefix)
        expected_connected_z = np.mean([2.0 - z[a] * z[b] for a, b in R_EDGES])
        expected_connected_x = np.mean([7.0 - x[a] * x[b] for a, b in R_EDGES])
        self.assertAlmostEqual(result[0, 0, 10], expected_connected_z)
        self.assertAlmostEqual(result[0, 0, 11], expected_connected_x)

    def test_validation_rejects_non_base_models_and_bad_arrays(self):
        validate_base_configuration()
        with self.assertRaises(CudaQConfigurationError):
            validate_base_configuration(reuploads=2)
        with self.assertRaises(CudaQConfigurationError):
            validate_base_configuration(input_encoding="gibbs")
        with self.assertRaises(ValueError):
            validate_arrays(self.features[:, :3], self.parameters)
        bad = self.features.copy()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_arrays(bad, self.parameters)

    def test_cudaq_import_is_lazy_and_error_is_actionable(self):
        backend = CudaQD4OrbitBackend()
        self.assertIsNone(backend._cudaq)
        with mock.patch(
            "d4_orqb.cudaq_backend.importlib.import_module",
            side_effect=ModuleNotFoundError("cudaq"),
        ):
            with self.assertRaisesRegex(CudaQUnavailableError, "not installed"):
                _require_cudaq()

    def test_fixture_validation_without_cudaq(self):
        expected = np.zeros((2, HEADS * INVARIANTS_PER_HEAD), dtype=np.float32)
        metadata = {"schema": FIXTURE_SCHEMA, "seed": 9}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez_compressed(
                path,
                schema=np.asarray(FIXTURE_SCHEMA),
                orbit_features=self.features,
                parameters=self.parameters,
                expected_invariants=expected,
                metadata_json=np.asarray(__import__("json").dumps(metadata)),
            )
            features, parameters, loaded_expected, loaded_metadata = load_fixture(path)
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(parameters, self.parameters)
        np.testing.assert_array_equal(loaded_expected, expected)
        self.assertEqual(loaded_metadata, metadata)


@unittest.skipUnless(importlib.util.find_spec("cudaq"), "CUDA-Q is not installed")
class OptionalCudaQParityTests(unittest.TestCase):
    def test_qpp_cpu_matches_pytorch_reference(self):
        import torch

        from d4_orqb.quantum import D4OrbitQuantumBottleneck

        torch.manual_seed(101)
        core = D4OrbitQuantumBottleneck(heads=4, reuploads=3).eval()
        features = torch.randn(1, 4, 2, 8)
        with torch.no_grad():
            expected = np.asarray(core(features).tolist(), dtype=np.float32)
        backend = CudaQD4OrbitBackend(target="qpp-cpu", require_nvidia=False)
        actual = backend(
            np.asarray(features.tolist(), dtype=np.float32),
            np.asarray(core.params.detach().tolist(), dtype=np.float32),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
