"""Equivariant quantum circuit: EQNN_for_HEP p4m QCNN ported to TorchQuantum.

Direct TorchQuantum port of EQNN_for_HEP/Equivariant_QCNN/models (matches
frozen_quantum_model_1.ipynb):
  * U2_equiv  = RX,RX, IsingZZ, RX,RX, IsingYY  (IsingZZ/IsingYY from native gates)
  * Pooling_ansatz_equiv = RX,RX,RY,RZ,CRX
  * wired in the p4m orbit from QCNN_circuit.p4m_QCNN_structure.
33 quantum parameters: conv1(6) pool1(5) conv2(6) pool2(5) conv3(6) pool3(5).

NOTE on the conv filter: p4m_QCNN_structure() is called with U4_equiv in the repo
but IGNORES that argument and hardcodes U2_equiv for all three conv stages, so this
port uses u2_equiv everywhere (U4_equiv / DiagonalQubitUnitary is never executed).

Readout adaptation (multiclass): the repo returns probs(wires=4) for BINARY
cross-entropy. For multi-class we measure <P> for P in paulis on all wires (+
optional <Z_iZ_j>) and feed a small linear head. The final Hadamard(4) is KEPT
(faithful port).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def amplitude_encode(qdev: tq.QuantumDevice, features: torch.Tensor) -> None:
    """Load real amplitudes (B, 2**n_wires) into the device state (like AmplitudeEmbedding)."""
    bsz, dim = features.shape
    n_wires = qdev.n_wires
    assert dim == 2 ** n_wires, f"expected {2 ** n_wires} amplitudes, got {dim}"
    qdev.reset_states(bsz)
    states = features.to(torch.complex64).reshape([bsz] + [2] * n_wires)
    qdev.set_states(states)


# Prefer native batched Ising gates when available: each native IsingZZ/IsingYY is a
# single fused kernel instead of 3 / 7 gate calls, a big win on GPU where this circuit
# is launch-bound. Falls back to the explicit decomposition on older versions.
_HAS_RZZ = hasattr(tq.QuantumDevice, "rzz")
_HAS_RYY = hasattr(tq.QuantumDevice, "ryy")


def _rzz(qdev, theta, w0, w1):
    """IsingZZ(theta) = exp(-i theta/2 Z_w0 Z_w1), SWAP-symmetric."""
    if _HAS_RZZ:
        qdev.rzz(wires=[w0, w1], params=theta)
        return
    qdev.cnot(wires=[w0, w1])
    qdev.rz(wires=w1, params=theta)
    qdev.cnot(wires=[w0, w1])


def _ryy(qdev, theta, w0, w1, pi2):
    """IsingYY(theta) = exp(-i theta/2 Y_w0 Y_w1), SWAP-symmetric."""
    if _HAS_RYY:
        qdev.ryy(wires=[w0, w1], params=theta)
        return
    qdev.rx(wires=w0, params=pi2)
    qdev.rx(wires=w1, params=pi2)
    qdev.cnot(wires=[w0, w1])
    qdev.rz(wires=w1, params=theta)
    qdev.cnot(wires=[w0, w1])
    qdev.rx(wires=w0, params=-pi2)
    qdev.rx(wires=w1, params=-pi2)


def u2_equiv(qdev, p, w0, w1, pi2):
    """Port of unitary.U2_equiv (6 params). RX angles tied across the SWAP-paired wires."""
    qdev.rx(wires=w0, params=p[:, 0])
    qdev.rx(wires=w1, params=p[:, 1])
    _rzz(qdev, p[:, 2], w0, w1)
    qdev.rx(wires=w0, params=p[:, 3])
    qdev.rx(wires=w1, params=p[:, 4])
    _ryy(qdev, p[:, 5], w0, w1, pi2)


def pooling_equiv(qdev, phi, w0, w1):
    """Port of unitary.Pooling_ansatz_equiv (5 params). wires order = [w0, w1] as in the repo."""
    qdev.rx(wires=w1, params=phi[:, 0])
    qdev.rx(wires=w0, params=phi[:, 1])
    qdev.ry(wires=w0, params=phi[:, 2])
    qdev.rz(wires=w0, params=phi[:, 3])
    qdev.crx(wires=[w0, w1], params=phi[:, 4])


class EquivQCNN_TQ(tq.QuantumModule):
    """TorchQuantum p4m EquivQCNN with a configurable multi-observable readout."""

    # p4m orbits used by QCNN_circuit.p4m_QCNN_structure
    CONV1_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (5, 6), (0, 3), (4, 7)]
    POOL1_EDGES = [(1, 0), (3, 2), (5, 4), (7, 6)]
    CONV2_EDGES = [(0, 2), (4, 6)]
    POOL2_EDGES = [(2, 0), (6, 4)]
    CONV3_EDGES = [(0, 4)]
    POOL3_EDGE = (0, 4)

    def __init__(self, n_qubits: int = 8, encoding="amplitude", reupload_layers=2,
                 paulis=("Z", "X", "Y"), zz_edges=None):
        super().__init__()
        assert n_qubits == 8, "EquivQCNN_TQ follows the 8-qubit p4m construction"
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.n_layers = reupload_layers if encoding == "reupload" else 1
        self.paulis = tuple(paulis)
        self.zz_edges = list(zz_edges) if zz_edges else []
        self.readout_dim = len(self.paulis) * n_qubits + len(self.zz_edges)

        # One weight-tied param set per conv/pool stage (shared across the orbit).
        # Shape (n_layers, k): independent quantum params for each re-upload layer.
        L = self.n_layers
        self.conv1 = nn.Parameter(0.1 * torch.randn(L, 6))
        self.pool1 = nn.Parameter(0.1 * torch.randn(L, 5))
        self.conv2 = nn.Parameter(0.1 * torch.randn(L, 6))
        self.pool2 = nn.Parameter(0.1 * torch.randn(L, 5))
        self.conv3 = nn.Parameter(0.1 * torch.randn(L, 6))
        self.pool3 = nn.Parameter(0.1 * torch.randn(L, 5))

        self.measure = tq.MeasureAll(tq.PauliZ)

        # Per-wire Z-eigenvalue (+/-1) signs for every basis state, used to form <Z_i Z_j>
        # directly from |psi|^2. Bit convention matches amplitude_encode: wire i is bit (n-1-i).
        idx = torch.arange(2 ** n_qubits)
        signs = torch.stack(
            [1.0 - 2.0 * ((idx >> (n_qubits - 1 - i)) & 1).float() for i in range(n_qubits)]
        )
        self.register_buffer("_z_signs", signs)

        # Reuse one QuantumDevice per (bsz, device) instead of reallocating each forward.
        self._qdev_cache = {}

    def _get_qdev(self, bsz: int, dev: torch.device) -> tq.QuantumDevice:
        key = (bsz, str(dev))
        qdev = self._qdev_cache.get(key)
        if qdev is None:
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=dev)
            self._qdev_cache[key] = qdev
        return qdev

    def _set_state(self, qdev, psi):
        bsz = psi.shape[0]
        qdev.set_states(psi.reshape([bsz] + [2] * self.n_qubits))

    @staticmethod
    def _expand(params: torch.Tensor, bsz: int) -> torch.Tensor:
        return params.unsqueeze(0).expand(bsz, -1)

    def _qcnn_block(self, qdev, l, bsz, pi2):
        """One faithful p4m pass (conv1->pool1->conv2->pool2->conv3->pool3) with layer-l params."""
        c1 = self._expand(self.conv1[l], bsz)
        p1 = self._expand(self.pool1[l], bsz)
        c2 = self._expand(self.conv2[l], bsz)
        p2 = self._expand(self.pool2[l], bsz)
        c3 = self._expand(self.conv3[l], bsz)
        p3 = self._expand(self.pool3[l], bsz)

        for (a, b) in self.CONV1_EDGES:
            u2_equiv(qdev, c1, a, b, pi2)
        for (a, b) in self.POOL1_EDGES:
            pooling_equiv(qdev, p1, a, b)

        for (a, b) in self.CONV2_EDGES:
            u2_equiv(qdev, c2, a, b, pi2)
        for (a, b) in self.POOL2_EDGES:
            pooling_equiv(qdev, p2, a, b)

        for (a, b) in self.CONV3_EDGES:
            u2_equiv(qdev, c3, a, b, pi2)
        pooling_equiv(qdev, p3, *self.POOL3_EDGE)

    def _readout(self, qdev, pi2):
        """Multi-observable readout: <Z>,<X>,<Y> per wire + optional <Z_iZ_j>."""
        psi = qdev.get_states_1d().clone()
        outs = []
        for p in self.paulis:
            self._set_state(qdev, psi)
            if p == "X":
                for w in range(self.n_qubits):
                    qdev.h(wires=w)
            elif p == "Y":
                for w in range(self.n_qubits):
                    qdev.rx(wires=w, params=pi2)
            elif p != "Z":
                raise ValueError(f"Unsupported readout Pauli: {p!r}")
            outs.append(self.measure(qdev))

        if self.zz_edges:
            probs = (psi.abs() ** 2).to(self._z_signs.dtype)
            zz = [probs @ (self._z_signs[i] * self._z_signs[j]) for (i, j) in self.zz_edges]
            outs.append(torch.stack(zz, dim=-1))

        return torch.cat(outs, dim=-1)

    def forward(self, q_in: torch.Tensor) -> torch.Tensor:
        """q_in: amplitude (B, 2**n) | angle (B, n) | reupload (B, L, n) -> (B, readout_dim)."""
        bsz = q_in.shape[0]
        dev = q_in.device
        pi2 = torch.full((bsz,), np.pi / 2, device=dev, dtype=q_in.dtype)

        qdev = self._get_qdev(bsz, dev)

        if self.encoding == "amplitude":
            amplitude_encode(qdev, q_in)
            self._qcnn_block(qdev, 0, bsz, pi2)
        elif self.encoding == "angle":
            qdev.reset_states(bsz)
            for i in range(self.n_qubits):
                qdev.ry(wires=i, params=q_in[:, i])
            self._qcnn_block(qdev, 0, bsz, pi2)
        else:  # reupload
            qdev.reset_states(bsz)
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qdev.ry(wires=i, params=q_in[:, l, i])
                self._qcnn_block(qdev, l, bsz, pi2)

        # Final Hadamard(4) from p4m_QCNN_structure (faithful to the repo).
        qdev.h(wires=4)
        return self._readout(qdev, pi2)
