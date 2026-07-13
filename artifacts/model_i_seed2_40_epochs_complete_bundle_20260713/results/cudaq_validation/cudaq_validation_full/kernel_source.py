"""Primary CUDA-Q circuit and analytic invariant backend.

This cell contains the actual decorated quantum kernel used for inference.
It deliberately has no PyTorch dependency and can be executed directly in
the pinned NVIDIA CUDA-Q container.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

CUDAQ_HEADS = 4
CUDAQ_REUPLOADS = 2
CUDAQ_QUBITS = 8
CUDAQ_PARAMETERS_PER_LAYER = 11
CUDAQ_INVARIANTS_PER_HEAD = 12
R_EDGES = (
    (0, 1), (0, 3), (1, 2), (2, 3),
    (4, 5), (4, 7), (5, 6), (6, 7),
)
R2_EDGES = ((0, 2), (1, 3), (4, 6), (5, 7))
S_EDGES = ((0, 4), (1, 7), (2, 6), (3, 5))
PAIR_FAMILIES = (R_EDGES, R2_EDGES, S_EDGES)

CUDAQ_AVAILABLE = importlib.util.find_spec("cudaq") is not None
if CUDAQ_AVAILABLE:
    import cudaq

    @cudaq.kernel
    def d4_orqb_q2_head(features: list[float], parameters: list[float]):
        qubits = cudaq.qvector(8)
        ry(parameters[0] * features[0] + parameters[1], qubits[7])
        rz(parameters[2] * features[8] + parameters[3], qubits[7])
        ry(parameters[0] * features[1] + parameters[1], qubits[6])
        rz(parameters[2] * features[9] + parameters[3], qubits[6])
        ry(parameters[0] * features[2] + parameters[1], qubits[5])
        rz(parameters[2] * features[10] + parameters[3], qubits[5])
        ry(parameters[0] * features[3] + parameters[1], qubits[4])
        rz(parameters[2] * features[11] + parameters[3], qubits[4])
        ry(parameters[0] * features[4] + parameters[1], qubits[3])
        rz(parameters[2] * features[12] + parameters[3], qubits[3])
        ry(parameters[0] * features[5] + parameters[1], qubits[2])
        rz(parameters[2] * features[13] + parameters[3], qubits[2])
        ry(parameters[0] * features[6] + parameters[1], qubits[1])
        rz(parameters[2] * features[14] + parameters[3], qubits[1])
        ry(parameters[0] * features[7] + parameters[1], qubits[0])
        rz(parameters[2] * features[15] + parameters[3], qubits[0])
        rx(parameters[4], qubits[7])
        ry(parameters[5], qubits[7])
        rz(parameters[6], qubits[7])
        rx(parameters[4], qubits[6])
        ry(parameters[5], qubits[6])
        rz(parameters[6], qubits[6])
        rx(parameters[4], qubits[5])
        ry(parameters[5], qubits[5])
        rz(parameters[6], qubits[5])
        rx(parameters[4], qubits[4])
        ry(parameters[5], qubits[4])
        rz(parameters[6], qubits[4])
        rx(parameters[4], qubits[3])
        ry(parameters[5], qubits[3])
        rz(parameters[6], qubits[3])
        rx(parameters[4], qubits[2])
        ry(parameters[5], qubits[2])
        rz(parameters[6], qubits[2])
        rx(parameters[4], qubits[1])
        ry(parameters[5], qubits[1])
        rz(parameters[6], qubits[1])
        rx(parameters[4], qubits[0])
        ry(parameters[5], qubits[0])
        rz(parameters[6], qubits[0])
        x.ctrl(qubits[7], qubits[6])
        rz(parameters[7], qubits[6])
        x.ctrl(qubits[7], qubits[6])
        x.ctrl(qubits[7], qubits[4])
        rz(parameters[7], qubits[4])
        x.ctrl(qubits[7], qubits[4])
        x.ctrl(qubits[6], qubits[5])
        rz(parameters[7], qubits[5])
        x.ctrl(qubits[6], qubits[5])
        x.ctrl(qubits[5], qubits[4])
        rz(parameters[7], qubits[4])
        x.ctrl(qubits[5], qubits[4])
        x.ctrl(qubits[3], qubits[2])
        rz(parameters[7], qubits[2])
        x.ctrl(qubits[3], qubits[2])
        x.ctrl(qubits[3], qubits[0])
        rz(parameters[7], qubits[0])
        x.ctrl(qubits[3], qubits[0])
        x.ctrl(qubits[2], qubits[1])
        rz(parameters[7], qubits[1])
        x.ctrl(qubits[2], qubits[1])
        x.ctrl(qubits[1], qubits[0])
        rz(parameters[7], qubits[0])
        x.ctrl(qubits[1], qubits[0])
        x.ctrl(qubits[7], qubits[3])
        rz(parameters[8], qubits[3])
        x.ctrl(qubits[7], qubits[3])
        x.ctrl(qubits[6], qubits[0])
        rz(parameters[8], qubits[0])
        x.ctrl(qubits[6], qubits[0])
        x.ctrl(qubits[5], qubits[1])
        rz(parameters[8], qubits[1])
        x.ctrl(qubits[5], qubits[1])
        x.ctrl(qubits[4], qubits[2])
        rz(parameters[8], qubits[2])
        x.ctrl(qubits[4], qubits[2])
        h(qubits[7])
        h(qubits[6])
        x.ctrl(qubits[7], qubits[6])
        rz(parameters[9], qubits[6])
        x.ctrl(qubits[7], qubits[6])
        h(qubits[7])
        h(qubits[6])
        h(qubits[7])
        h(qubits[4])
        x.ctrl(qubits[7], qubits[4])
        rz(parameters[9], qubits[4])
        x.ctrl(qubits[7], qubits[4])
        h(qubits[7])
        h(qubits[4])
        h(qubits[6])
        h(qubits[5])
        x.ctrl(qubits[6], qubits[5])
        rz(parameters[9], qubits[5])
        x.ctrl(qubits[6], qubits[5])
        h(qubits[6])
        h(qubits[5])
        h(qubits[5])
        h(qubits[4])
        x.ctrl(qubits[5], qubits[4])
        rz(parameters[9], qubits[4])
        x.ctrl(qubits[5], qubits[4])
        h(qubits[5])
        h(qubits[4])
        h(qubits[3])
        h(qubits[2])
        x.ctrl(qubits[3], qubits[2])
        rz(parameters[9], qubits[2])
        x.ctrl(qubits[3], qubits[2])
        h(qubits[3])
        h(qubits[2])
        h(qubits[3])
        h(qubits[0])
        x.ctrl(qubits[3], qubits[0])
        rz(parameters[9], qubits[0])
        x.ctrl(qubits[3], qubits[0])
        h(qubits[3])
        h(qubits[0])
        h(qubits[2])
        h(qubits[1])
        x.ctrl(qubits[2], qubits[1])
        rz(parameters[9], qubits[1])
        x.ctrl(qubits[2], qubits[1])
        h(qubits[2])
        h(qubits[1])
        h(qubits[1])
        h(qubits[0])
        x.ctrl(qubits[1], qubits[0])
        rz(parameters[9], qubits[0])
        x.ctrl(qubits[1], qubits[0])
        h(qubits[1])
        h(qubits[0])
        h(qubits[7])
        h(qubits[3])
        x.ctrl(qubits[7], qubits[3])
        rz(parameters[10], qubits[3])
        x.ctrl(qubits[7], qubits[3])
        h(qubits[7])
        h(qubits[3])
        h(qubits[6])
        h(qubits[0])
        x.ctrl(qubits[6], qubits[0])
        rz(parameters[10], qubits[0])
        x.ctrl(qubits[6], qubits[0])
        h(qubits[6])
        h(qubits[0])
        h(qubits[5])
        h(qubits[1])
        x.ctrl(qubits[5], qubits[1])
        rz(parameters[10], qubits[1])
        x.ctrl(qubits[5], qubits[1])
        h(qubits[5])
        h(qubits[1])
        h(qubits[4])
        h(qubits[2])
        x.ctrl(qubits[4], qubits[2])
        rz(parameters[10], qubits[2])
        x.ctrl(qubits[4], qubits[2])
        h(qubits[4])
        h(qubits[2])
        ry(parameters[11] * features[0] + parameters[12], qubits[7])
        rz(parameters[13] * features[8] + parameters[14], qubits[7])
        ry(parameters[11] * features[1] + parameters[12], qubits[6])
        rz(parameters[13] * features[9] + parameters[14], qubits[6])
        ry(parameters[11] * features[2] + parameters[12], qubits[5])
        rz(parameters[13] * features[10] + parameters[14], qubits[5])
        ry(parameters[11] * features[3] + parameters[12], qubits[4])
        rz(parameters[13] * features[11] + parameters[14], qubits[4])
        ry(parameters[11] * features[4] + parameters[12], qubits[3])
        rz(parameters[13] * features[12] + parameters[14], qubits[3])
        ry(parameters[11] * features[5] + parameters[12], qubits[2])
        rz(parameters[13] * features[13] + parameters[14], qubits[2])
        ry(parameters[11] * features[6] + parameters[12], qubits[1])
        rz(parameters[13] * features[14] + parameters[14], qubits[1])
        ry(parameters[11] * features[7] + parameters[12], qubits[0])
        rz(parameters[13] * features[15] + parameters[14], qubits[0])
        rx(parameters[15], qubits[7])
        ry(parameters[16], qubits[7])
        rz(parameters[17], qubits[7])
        rx(parameters[15], qubits[6])
        ry(parameters[16], qubits[6])
        rz(parameters[17], qubits[6])
        rx(parameters[15], qubits[5])
        ry(parameters[16], qubits[5])
        rz(parameters[17], qubits[5])
        rx(parameters[15], qubits[4])
        ry(parameters[16], qubits[4])
        rz(parameters[17], qubits[4])
        rx(parameters[15], qubits[3])
        ry(parameters[16], qubits[3])
        rz(parameters[17], qubits[3])
        rx(parameters[15], qubits[2])
        ry(parameters[16], qubits[2])
        rz(parameters[17], qubits[2])
        rx(parameters[15], qubits[1])
        ry(parameters[16], qubits[1])
        rz(parameters[17], qubits[1])
        rx(parameters[15], qubits[0])
        ry(parameters[16], qubits[0])
        rz(parameters[17], qubits[0])
        x.ctrl(qubits[7], qubits[6])
        rz(parameters[18], qubits[6])
        x.ctrl(qubits[7], qubits[6])
        x.ctrl(qubits[7], qubits[4])
        rz(parameters[18], qubits[4])
        x.ctrl(qubits[7], qubits[4])
        x.ctrl(qubits[6], qubits[5])
        rz(parameters[18], qubits[5])
        x.ctrl(qubits[6], qubits[5])
        x.ctrl(qubits[5], qubits[4])
        rz(parameters[18], qubits[4])
        x.ctrl(qubits[5], qubits[4])
        x.ctrl(qubits[3], qubits[2])
        rz(parameters[18], qubits[2])
        x.ctrl(qubits[3], qubits[2])
        x.ctrl(qubits[3], qubits[0])
        rz(parameters[18], qubits[0])
        x.ctrl(qubits[3], qubits[0])
        x.ctrl(qubits[2], qubits[1])
        rz(parameters[18], qubits[1])
        x.ctrl(qubits[2], qubits[1])
        x.ctrl(qubits[1], qubits[0])
        rz(parameters[18], qubits[0])
        x.ctrl(qubits[1], qubits[0])
        x.ctrl(qubits[7], qubits[3])
        rz(parameters[19], qubits[3])
        x.ctrl(qubits[7], qubits[3])
        x.ctrl(qubits[6], qubits[0])
        rz(parameters[19], qubits[0])
        x.ctrl(qubits[6], qubits[0])
        x.ctrl(qubits[5], qubits[1])
        rz(parameters[19], qubits[1])
        x.ctrl(qubits[5], qubits[1])
        x.ctrl(qubits[4], qubits[2])
        rz(parameters[19], qubits[2])
        x.ctrl(qubits[4], qubits[2])
        h(qubits[7])
        h(qubits[6])
        x.ctrl(qubits[7], qubits[6])
        rz(parameters[20], qubits[6])
        x.ctrl(qubits[7], qubits[6])
        h(qubits[7])
        h(qubits[6])
        h(qubits[7])
        h(qubits[4])
        x.ctrl(qubits[7], qubits[4])
        rz(parameters[20], qubits[4])
        x.ctrl(qubits[7], qubits[4])
        h(qubits[7])
        h(qubits[4])
        h(qubits[6])
        h(qubits[5])
        x.ctrl(qubits[6], qubits[5])
        rz(parameters[20], qubits[5])
        x.ctrl(qubits[6], qubits[5])
        h(qubits[6])
        h(qubits[5])
        h(qubits[5])
        h(qubits[4])
        x.ctrl(qubits[5], qubits[4])
        rz(parameters[20], qubits[4])
        x.ctrl(qubits[5], qubits[4])
        h(qubits[5])
        h(qubits[4])
        h(qubits[3])
        h(qubits[2])
        x.ctrl(qubits[3], qubits[2])
        rz(parameters[20], qubits[2])
        x.ctrl(qubits[3], qubits[2])
        h(qubits[3])
        h(qubits[2])
        h(qubits[3])
        h(qubits[0])
        x.ctrl(qubits[3], qubits[0])
        rz(parameters[20], qubits[0])
        x.ctrl(qubits[3], qubits[0])
        h(qubits[3])
        h(qubits[0])
        h(qubits[2])
        h(qubits[1])
        x.ctrl(qubits[2], qubits[1])
        rz(parameters[20], qubits[1])
        x.ctrl(qubits[2], qubits[1])
        h(qubits[2])
        h(qubits[1])
        h(qubits[1])
        h(qubits[0])
        x.ctrl(qubits[1], qubits[0])
        rz(parameters[20], qubits[0])
        x.ctrl(qubits[1], qubits[0])
        h(qubits[1])
        h(qubits[0])
        h(qubits[7])
        h(qubits[3])
        x.ctrl(qubits[7], qubits[3])
        rz(parameters[21], qubits[3])
        x.ctrl(qubits[7], qubits[3])
        h(qubits[7])
        h(qubits[3])
        h(qubits[6])
        h(qubits[0])
        x.ctrl(qubits[6], qubits[0])
        rz(parameters[21], qubits[0])
        x.ctrl(qubits[6], qubits[0])
        h(qubits[6])
        h(qubits[0])
        h(qubits[5])
        h(qubits[1])
        x.ctrl(qubits[5], qubits[1])
        rz(parameters[21], qubits[1])
        x.ctrl(qubits[5], qubits[1])
        h(qubits[5])
        h(qubits[1])
        h(qubits[4])
        h(qubits[2])
        x.ctrl(qubits[4], qubits[2])
        rz(parameters[21], qubits[2])
        x.ctrl(qubits[4], qubits[2])
        h(qubits[4])
        h(qubits[2])
else:
    cudaq = None
    d4_orqb_q2_head = None


def cudaq_primitive_expectations(state: np.ndarray) -> np.ndarray:
    """Compute ordered Z/X/ZZ/XX expectations from one CUDA-Q state."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state.shape != (1 << CUDAQ_QUBITS,):
        raise ValueError(f"Expected 256 amplitudes, got {state.shape}")
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, atol=2e-5):
        raise ValueError(f"Invalid CUDA-Q state norm {norm}")
    basis = np.arange(1 << CUDAQ_QUBITS, dtype=np.int64)
    probabilities = np.square(np.abs(state))
    z_values, x_values = [], []
    for logical in range(CUDAQ_QUBITS):
        mask = 1 << (CUDAQ_QUBITS - logical - 1)
        signs = np.where((basis & mask) != 0, -1.0, 1.0)
        z_values.append(float(probabilities @ signs))
        x_values.append(float(np.vdot(state, state[basis ^ mask]).real))
    values = z_values + x_values
    for pauli in ("z", "x"):
        for edges in PAIR_FAMILIES:
            for logical_a, logical_b in edges:
                mask_a = 1 << (CUDAQ_QUBITS - logical_a - 1)
                mask_b = 1 << (CUDAQ_QUBITS - logical_b - 1)
                if pauli == "z":
                    signs = np.where((basis & mask_a) != 0, -1.0, 1.0)
                    signs *= np.where((basis & mask_b) != 0, -1.0, 1.0)
                    values.append(float(probabilities @ signs))
                else:
                    values.append(
                        float(np.vdot(state, state[basis ^ mask_a ^ mask_b]).real)
                    )
    output = np.asarray(values, dtype=np.float64)
    if output.shape != (48,):
        raise AssertionError(output.shape)
    return output


def cudaq_invariants_from_expectations(values: np.ndarray) -> np.ndarray:
    """Reduce 48 primitive expectations to the exact 12 head invariants."""
    values = np.asarray(values, dtype=np.float64)
    cursor = 0
    z_values = values[cursor:cursor + 8]; cursor += 8
    x_values = values[cursor:cursor + 8]; cursor += 8
    zz_values, xx_values = [], []
    for destination in (zz_values, xx_values):
        for edges in PAIR_FAMILIES:
            destination.append(values[cursor:cursor + len(edges)])
            cursor += len(edges)
    z_mean, x_mean = z_values.mean(), x_values.mean()
    r_z_product = np.asarray([z_values[a] * z_values[b] for a, b in R_EDGES])
    r_x_product = np.asarray([x_values[a] * x_values[b] for a, b in R_EDGES])
    output = np.asarray([
        z_mean,
        np.square(z_values).mean() - z_mean**2,
        x_mean,
        np.square(x_values).mean() - x_mean**2,
        *(family.mean() for family in zz_values),
        *(family.mean() for family in xx_values),
        (zz_values[0] - r_z_product).mean(),
        (xx_values[0] - r_x_product).mean(),
    ], dtype=np.float64)
    if output.shape != (CUDAQ_INVARIANTS_PER_HEAD,):
        raise AssertionError(output.shape)
    return output


class CudaQD4OrbitBackend:
    """CUDA-Q execution of four independent selected q2 head circuits."""

    def __init__(self, target: str = "nvidia") -> None:
        self.target = target
        self.initialized = False

    def initialize(self) -> None:
        if self.initialized:
            return
        if not CUDAQ_AVAILABLE or d4_orqb_q2_head is None:
            raise RuntimeError(
                "CUDA-Q is not installed in this kernel. Use the pinned NVIDIA "
                "CUDA-Q container to launch quantum inference."
            )
        if self.target != "nvidia":
            raise ValueError("This analytic backend was audited for target='nvidia'")
        cudaq.set_target(self.target)
        self.initialized = True

    def _head(self, features: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        self.initialize()
        state = cudaq.get_state(
            d4_orqb_q2_head,
            np.asarray(features, dtype=np.float64).reshape(16).tolist(),
            np.asarray(parameters, dtype=np.float64).reshape(22).tolist(),
        )
        state_array = state.to_numpy() if hasattr(state, "to_numpy") else np.asarray(state)
        return cudaq_invariants_from_expectations(
            cudaq_primitive_expectations(state_array)
        )

    def __call__(self, angles: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles)
        parameters = np.asarray(parameters)
        if angles.ndim != 4 or angles.shape[1:] != (4, 2, 8):
            raise ValueError(f"Expected angles (B,4,2,8), got {angles.shape}")
        if parameters.shape != (4, 2, 11):
            raise ValueError(f"Expected parameters (4,2,11), got {parameters.shape}")
        output = np.empty((len(angles), 48), dtype=np.float32)
        for sample in range(len(angles)):
            for head in range(CUDAQ_HEADS):
                start = head * CUDAQ_INVARIANTS_PER_HEAD
                output[sample, start:start + CUDAQ_INVARIANTS_PER_HEAD] = self._head(
                    angles[sample, head], parameters[head]
                )
        return output


print({
    "primary_quantum_implementation": "CUDA-Q @cudaq.kernel",
    "cudaq_available_in_this_notebook_kernel": CUDAQ_AVAILABLE,
    "heads": CUDAQ_HEADS,
    "qubits_per_head": CUDAQ_QUBITS,
    "reuploads": CUDAQ_REUPLOADS,
    "parameters": CUDAQ_HEADS * CUDAQ_REUPLOADS * CUDAQ_PARAMETERS_PER_LAYER,
})
