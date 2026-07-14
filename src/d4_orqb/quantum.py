"""TorchQuantum implementation of the selected D4 orbit circuit.

The eight qubits are indexed by D4 group elements.  Parameters are tied over
complete left-Cayley edge orbits, so the circuit is equivariant to the regular
group action.  Orbit-averaged one- and two-qubit observables provide invariant
features to the classifier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

try:
    import torchquantum as tq
except ImportError as error:  # Keep --help and classical pretraining importable.
    tq = None
    _TORCHQUANTUM_IMPORT_ERROR: ImportError | None = error
else:
    _TORCHQUANTUM_IMPORT_ERROR = None


D4Element = Tuple[int, int]
D4_ELEMENTS: Tuple[D4Element, ...] = tuple(
    (rotation, reflected) for reflected in (0, 1) for rotation in range(4)
)
D4_INDEX = {element: index for index, element in enumerate(D4_ELEMENTS)}


def d4_multiply(left: D4Element, right: D4Element) -> D4Element:
    """Multiply ``r^k s^f`` elements using ``s r s = r^-1``."""

    k, f = left
    ell, m = right
    return ((k + (-1 if f else 1) * ell) % 4, (f + m) % 2)


def right_regular_permutation(element: D4Element) -> torch.Tensor:
    """Return ``p`` such that an orbit field maps as ``z'(g)=z(g*h)``."""

    return torch.tensor(
        [D4_INDEX[d4_multiply(group, element)] for group in D4_ELEMENTS],
        dtype=torch.long,
    )


def _unique_undirected_edges(
    generator: D4Element,
) -> Tuple[Tuple[int, int], ...]:
    edges = set()
    for group in D4_ELEMENTS:
        left = D4_INDEX[group]
        right = D4_INDEX[d4_multiply(generator, group)]
        if left != right:
            edges.add(tuple(sorted((left, right))))
    return tuple(sorted(edges))


R_EDGES = _unique_undirected_edges((1, 0))
R2_EDGES = _unique_undirected_edges((2, 0))
S_EDGES = _unique_undirected_edges((0, 1))


def _bit_mask(qubit: int, n_qubits: int) -> int:
    """Return the big-endian statevector bit used by TorchQuantum wire IDs."""

    return 1 << (n_qubits - qubit - 1)


def require_torchquantum() -> Any:
    """Return TorchQuantum or fail before a requested quantum run starts."""

    if tq is None:
        raise ModuleNotFoundError(
            "TorchQuantum is required for the D4 quantum stage. Install the "
            "dependencies from src/requirements.txt before training."
        ) from _TORCHQUANTUM_IMPORT_ERROR
    return tq


def _rzz(qdev: Any, theta: torch.Tensor, first: int, second: int) -> None:
    """Apply ``exp(-i theta Z⊗Z / 2)``, using a portable fallback."""

    if hasattr(qdev, "rzz"):
        qdev.rzz(wires=[first, second], params=theta)
        return
    qdev.cnot(wires=[first, second])
    qdev.rz(wires=second, params=theta)
    qdev.cnot(wires=[first, second])


def _rxx(qdev: Any, theta: torch.Tensor, first: int, second: int) -> None:
    """Apply ``exp(-i theta X⊗X / 2)``, using a portable fallback."""

    if hasattr(qdev, "rxx"):
        qdev.rxx(wires=[first, second], params=theta)
        return
    qdev.h(wires=first)
    qdev.h(wires=second)
    _rzz(qdev, theta, first, second)
    qdev.h(wires=first)
    qdev.h(wires=second)


def _edge_expectation_z(
    probabilities: torch.Tensor,
    z_signs: torch.Tensor,
    edges: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    observables = torch.stack([z_signs[a] * z_signs[b] for a, b in edges])
    return probabilities @ observables.transpose(0, 1)


def _expectation_x(
    state: torch.Tensor,
    qubits: Sequence[Tuple[int, ...]],
    n_qubits: int,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    basis = torch.arange(state.shape[1], device=state.device)
    for wires in qubits:
        mask = 0
        for wire in wires:
            mask |= _bit_mask(wire, n_qubits)
        flipped = state.index_select(1, basis ^ mask)
        values.append((state.conj() * flipped).sum(dim=1).real)
    return torch.stack(values, dim=1)


_QuantumModule = tq.QuantumModule if tq is not None else nn.Module


class D4OrbitQuantumBottleneck(_QuantumModule):
    """Batched eight-qubit D4-equivariant TorchQuantum circuit heads."""

    parameters_per_layer = 11
    invariants_per_head = 12

    def __init__(
        self, heads: int = 4, reuploads: int = 2, n_qubits: int = 8
    ) -> None:
        require_torchquantum()
        super().__init__()
        if n_qubits != len(D4_ELEMENTS):
            raise ValueError("The D4 regular register requires exactly 8 qubits")
        if heads < 1 or reuploads < 1:
            raise ValueError("heads and reuploads must both be positive")

        self.heads = heads
        self.reuploads = reuploads
        self.n_qubits = n_qubits
        self.input_encoding = "angle"
        self.observable_readout = "pair"

        parameters = torch.zeros(heads, reuploads, self.parameters_per_layer)
        parameters[..., 0] = 1.0
        parameters[..., 2] = 1.0
        parameters[..., 4:] = 0.02 * torch.randn_like(parameters[..., 4:])
        self.params = nn.Parameter(parameters)

        basis = torch.arange(1 << n_qubits)
        signs = []
        for qubit in range(n_qubits):
            bit = (basis & _bit_mask(qubit, n_qubits)) != 0
            signs.append(
                torch.where(bit, -torch.ones_like(basis), torch.ones_like(basis))
            )
        self.register_buffer(
            "z_signs", torch.stack(signs).float(), persistent=False
        )

        # QuantumDevice owns transient execution state, not learned model state.
        self._qdev_cache: Dict[Tuple[int, str], Any] = {}

    @property
    def output_dim(self) -> int:
        return self.heads * self.invariants_per_head

    def _quantum_device(self, batch: int, device: torch.device) -> Any:
        key = (batch, str(device))
        qdev = self._qdev_cache.get(key)
        if qdev is None:
            torchquantum = require_torchquantum()
            qdev = torchquantum.QuantumDevice(
                n_wires=self.n_qubits,
                bsz=batch,
                device=device,
                record_op=False,
            )
            self._qdev_cache[key] = qdev
        else:
            qdev.reset_states(batch)
        return qdev

    @staticmethod
    def _edge_rotations(
        qdev: Any,
        theta: torch.Tensor,
        edges: Sequence[Tuple[int, int]],
        pauli: str,
    ) -> None:
        if pauli == "z":
            operation = _rzz
        elif pauli == "x":
            operation = _rxx
        else:
            raise ValueError(f"Unsupported Pauli rotation: {pauli}")
        for first, second in edges:
            operation(qdev, theta, first, second)

    def _run_statevector(self, orbit_features: torch.Tensor) -> torch.Tensor:
        batch = orbit_features.shape[0]
        flat = orbit_features.reshape(batch * self.heads, 2, self.n_qubits).float()
        parameters = self.params.unsqueeze(0).expand(batch, -1, -1, -1)
        parameters = parameters.reshape(
            batch * self.heads,
            self.reuploads,
            self.parameters_per_layer,
        ).float()

        qdev = self._quantum_device(flat.shape[0], flat.device)
        for layer in range(self.reuploads):
            layer_parameters = parameters[:, layer]

            # Data re-upload: two orbit channels become RY and RZ angles.
            for qubit in range(self.n_qubits):
                ry_angle = (
                    layer_parameters[:, 0] * flat[:, 0, qubit]
                    + layer_parameters[:, 1]
                )
                rz_angle = (
                    layer_parameters[:, 2] * flat[:, 1, qubit]
                    + layer_parameters[:, 3]
                )
                qdev.ry(wires=qubit, params=ry_angle)
                qdev.rz(wires=qubit, params=rz_angle)

            # Shared single-qubit trainable gates.
            for qubit in range(self.n_qubits):
                qdev.rx(wires=qubit, params=layer_parameters[:, 4])
                qdev.ry(wires=qubit, params=layer_parameters[:, 5])
                qdev.rz(wires=qubit, params=layer_parameters[:, 6])

            # Complete rotation/reflection Cayley edge orbits.
            self._edge_rotations(
                qdev, layer_parameters[:, 7], R_EDGES, "z"
            )
            self._edge_rotations(
                qdev, layer_parameters[:, 8], S_EDGES, "z"
            )
            self._edge_rotations(
                qdev, layer_parameters[:, 9], R_EDGES, "x"
            )
            self._edge_rotations(
                qdev, layer_parameters[:, 10], S_EDGES, "x"
            )

        return qdev.get_states_1d()

    def forward(
        self, orbit_features: torch.Tensor, return_equivariant: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        expected = (self.heads, 2, self.n_qubits)
        if orbit_features.ndim != 4 or orbit_features.shape[1:] != expected:
            raise ValueError(
                f"Expected (B,{self.heads},2,{self.n_qubits}), "
                f"got {tuple(orbit_features.shape)}"
            )

        with torch.autocast(
            device_type=orbit_features.device.type, enabled=False
        ):
            state = self._run_statevector(orbit_features)
            probabilities = state.abs().square()
            z = probabilities @ self.z_signs.transpose(0, 1)
            x = _expectation_x(
                state,
                [(qubit,) for qubit in range(self.n_qubits)],
                self.n_qubits,
            )
            edge_families = (R_EDGES, R2_EDGES, S_EDGES)
            zz = tuple(
                _edge_expectation_z(probabilities, self.z_signs, edges)
                for edges in edge_families
            )
            xx = tuple(
                _expectation_x(state, edges, self.n_qubits)
                for edges in edge_families
            )

            def edge_product(
                values: torch.Tensor, edges: Sequence[Tuple[int, int]]
            ) -> torch.Tensor:
                return torch.stack(
                    [values[:, a] * values[:, b] for a, b in edges], dim=1
                )

            z_mean = z.mean(dim=1)
            x_mean = x.mean(dim=1)
            invariant_features = [
                z_mean,
                z.square().mean(dim=1) - z_mean.square(),
                x_mean,
                x.square().mean(dim=1) - x_mean.square(),
                *(values.mean(dim=1) for values in zz),
                *(values.mean(dim=1) for values in xx),
                (zz[0] - edge_product(z, R_EDGES)).mean(dim=1),
                (xx[0] - edge_product(x, R_EDGES)).mean(dim=1),
            ]
            invariant = torch.stack(invariant_features, dim=1).reshape(
                orbit_features.shape[0], self.output_dim
            )

        if not return_equivariant:
            return invariant
        equivariant = {
            "z": z.reshape(
                orbit_features.shape[0], self.heads, self.n_qubits
            ),
            "x": x.reshape(
                orbit_features.shape[0], self.heads, self.n_qubits
            ),
        }
        return invariant, equivariant

    def parameter_report(self) -> Dict[str, int | str]:
        return {
            "qubits": self.n_qubits,
            "heads": self.heads,
            "reuploads": self.reuploads,
            "quantum_trainable": self.params.numel(),
            "input_encoding": self.input_encoding,
            "observable_readout": self.observable_readout,
            "execution_backend": "torchquantum",
            "statevector_dimension": 1 << self.n_qubits,
            "invariants": self.output_dim,
        }


def smoke_test_torchquantum(device: torch.device) -> Dict[str, int | str]:
    """Fail fast on backend, device, forward, and autograd incompatibility."""

    circuit = D4OrbitQuantumBottleneck().to(device)
    features = torch.linspace(
        -0.3, 0.3, 4 * 2 * 8, device=device, dtype=torch.float32
    ).reshape(1, 4, 2, 8)
    features.requires_grad_(True)
    output = circuit(features)
    output.square().mean().backward()
    gradients = (features.grad, circuit.params.grad)
    if not bool(torch.isfinite(output).all()) or any(
        gradient is None or not bool(torch.isfinite(gradient).all())
        for gradient in gradients
    ):
        raise RuntimeError("TorchQuantum forward/backward smoke test failed")
    return {
        "backend": "torchquantum",
        "device": str(device),
        "quantum_parameters": circuit.params.numel(),
        "invariant_features": circuit.output_dim,
    }
