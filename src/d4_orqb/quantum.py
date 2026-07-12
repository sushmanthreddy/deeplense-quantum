"""Pure-PyTorch statevector implementation of a D4-equivariant quantum bottleneck.

The eight qubits are indexed by D4 group elements. Feature channels therefore
transform in the regular representation. Every trainable single-qubit gate is
tied over the register and every two-qubit Hamiltonian is tied over a complete
left-Cayley edge orbit. The Pauli terms within each orbit commute, so their
product is equivariant to the right-regular action induced by orbit lifting.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


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
    """Return p such that an orbit field transforms as ``z'(g)=z(g*element)``."""

    return torch.tensor(
        [D4_INDEX[d4_multiply(g, element)] for g in D4_ELEMENTS], dtype=torch.long
    )


def _unique_undirected_edges(generator: D4Element) -> Tuple[Tuple[int, int], ...]:
    edges = set()
    for g in D4_ELEMENTS:
        a = D4_INDEX[g]
        b = D4_INDEX[d4_multiply(generator, g)]
        if a != b:
            edges.add(tuple(sorted((a, b))))
    return tuple(sorted(edges))


R_EDGES = _unique_undirected_edges((1, 0))
R2_EDGES = _unique_undirected_edges((2, 0))
S_EDGES = _unique_undirected_edges((0, 1))
RS_EDGES = _unique_undirected_edges((1, 1))
R2S_EDGES = _unique_undirected_edges((2, 1))
R3S_EDGES = _unique_undirected_edges((3, 1))
CAYLEY_EDGE_FAMILIES = (
    R_EDGES,
    R2_EDGES,
    S_EDGES,
    RS_EDGES,
    R2S_EDGES,
    R3S_EDGES,
)


def _unique_left_plaquettes(
    rotation: D4Element,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Return unique ``{g, t g, s g, t s g}`` four-site motifs.

    Left multiplication is intentional: the complete motif family is closed
    under the right-regular action induced by the image orbit lift.
    """

    reflection = (0, 1)
    motifs = set()
    for group_element in D4_ELEMENTS:
        rotated = d4_multiply(rotation, group_element)
        reflected = d4_multiply(reflection, group_element)
        rotated_reflection = d4_multiply(rotation, reflected)
        motif = tuple(
            sorted(
                D4_INDEX[element]
                for element in (
                    group_element,
                    rotated,
                    reflected,
                    rotated_reflection,
                )
            )
        )
        if len(set(motif)) != 4:
            raise RuntimeError(f"Degenerate D4 plaquette: {motif}")
        motifs.add(motif)
    return tuple(sorted(motifs))


RS_PLAQUETTES = _unique_left_plaquettes((1, 0))
R2S_PLAQUETTES = _unique_left_plaquettes((2, 0))


def _bit_mask(qubit: int, n_qubits: int) -> int:
    return 1 << (n_qubits - qubit - 1)


def _apply_ry(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    view = state.reshape(state.shape[0], left, 2, right)
    a0, a1 = view[:, :, 0, :], view[:, :, 1, :]
    c = torch.cos(theta / 2).view(-1, 1, 1)
    s = torch.sin(theta / 2).view(-1, 1, 1)
    out = torch.stack((c * a0 - s * a1, s * a0 + c * a1), dim=2)
    return out.reshape_as(state)


def _apply_rx(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    view = state.reshape(state.shape[0], left, 2, right)
    a0, a1 = view[:, :, 0, :], view[:, :, 1, :]
    c = torch.cos(theta / 2).view(-1, 1, 1)
    s = (-1j * torch.sin(theta / 2)).view(-1, 1, 1)
    out = torch.stack((c * a0 + s * a1, s * a0 + c * a1), dim=2)
    return out.reshape_as(state)


def _apply_rz(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    view = state.reshape(state.shape[0], left, 2, right)
    phase0 = torch.exp(-0.5j * theta).view(-1, 1, 1)
    phase1 = torch.exp(0.5j * theta).view(-1, 1, 1)
    out = torch.stack((phase0 * view[:, :, 0, :], phase1 * view[:, :, 1, :]), dim=2)
    return out.reshape_as(state)


def _apply_pair_rotation(
    state: torch.Tensor,
    theta: torch.Tensor,
    q0: int,
    q1: int,
    pauli: str,
    z_signs: torch.Tensor,
) -> torch.Tensor:
    """Apply exp(-i theta P_q0 P_q1 / 2), for P in {X,Z}."""

    c = torch.cos(theta / 2).view(-1, 1)
    s = (-1j * torch.sin(theta / 2)).view(-1, 1)
    if pauli == "z":
        transformed = state * (z_signs[q0] * z_signs[q1]).view(1, -1)
    elif pauli == "x":
        mask = _bit_mask(q0, z_signs.shape[0]) | _bit_mask(q1, z_signs.shape[0])
        permutation = torch.arange(state.shape[1], device=state.device) ^ mask
        transformed = state.index_select(1, permutation)
    else:
        raise ValueError(f"Unsupported Pauli operator: {pauli}")
    return c * state + s * transformed


def _edge_expectation_z(
    probabilities: torch.Tensor, z_signs: torch.Tensor, edges: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    observables = torch.stack([z_signs[a] * z_signs[b] for a, b in edges])
    return probabilities @ observables.transpose(0, 1)


def _expectation_z(
    probabilities: torch.Tensor,
    z_signs: torch.Tensor,
    qubits: Sequence[Tuple[int, ...]],
) -> torch.Tensor:
    observables = torch.stack(
        [z_signs[list(wires)].prod(dim=0) for wires in qubits], dim=0
    ).to(probabilities.dtype)
    return probabilities @ observables.transpose(0, 1)


def _expectation_x(state: torch.Tensor, qubits: Sequence[Tuple[int, ...]], n_qubits: int) -> torch.Tensor:
    values: List[torch.Tensor] = []
    basis = torch.arange(state.shape[1], device=state.device)
    for wires in qubits:
        mask = 0
        for wire in wires:
            mask |= _bit_mask(wire, n_qubits)
        flipped = state.index_select(1, basis ^ mask)
        values.append((state.conj() * flipped).sum(dim=1).real)
    return torch.stack(values, dim=1)


def _expectation_pauli_strings(
    state: torch.Tensor,
    strings: Sequence[Sequence[Tuple[int, Literal["X", "Y"]]]],
    n_qubits: int,
    z_signs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate X/Y Pauli strings without constructing dense operators.

    The phase for a Y operator is expressed in the output computational-basis
    index.  This keeps the operation differentiable with respect to the
    statevector and permits the equatorial readout to expose the imaginary
    quadrature generated by the circuit's RZ rotations.
    """

    values: List[torch.Tensor] = []
    basis = torch.arange(state.shape[1], device=state.device)
    for terms in strings:
        mask = 0
        phase = torch.ones(
            state.shape[1], dtype=state.dtype, device=state.device
        )
        for wire, pauli in terms:
            if not 0 <= wire < n_qubits:
                raise ValueError(f"Invalid Pauli wire: {wire}")
            mask |= _bit_mask(wire, n_qubits)
            if pauli == "Y":
                phase = phase * (
                    -1j * z_signs[wire].to(device=state.device, dtype=state.dtype)
                )
            elif pauli != "X":
                raise ValueError(f"Unsupported equatorial Pauli: {pauli}")
        transformed = state.index_select(1, basis ^ mask) * phase.view(1, -1)
        values.append((state.conj() * transformed).sum(dim=1).real)
    return torch.stack(values, dim=1)


def _expectation_symmetric_xz(
    state: torch.Tensor,
    edges: Sequence[Tuple[int, int]],
    n_qubits: int,
    z_signs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``<X_a Z_b + Z_a X_b>`` on undirected edges.

    The symmetric combination is the first derivative at zero of the tied
    meridional pair observable

    ``P(phi)_a P(phi)_b``, where ``P(phi)=cos(phi)X+sin(phi)Z``.

    Expressing each term in the output computational-basis index avoids dense
    operators and keeps the result differentiable with respect to the
    statevector.  Symmetrization is essential because the Cayley edge families
    are stored as undirected pairs.
    """

    values: List[torch.Tensor] = []
    basis = torch.arange(state.shape[1], device=state.device)
    for a, b in edges:
        if not (0 <= a < n_qubits and 0 <= b < n_qubits) or a == b:
            raise ValueError(f"Invalid mixed-Pauli edge: {(a, b)}")
        flip_a = state.index_select(1, basis ^ _bit_mask(a, n_qubits))
        flip_b = state.index_select(1, basis ^ _bit_mask(b, n_qubits))
        x_a_z_b = flip_a * z_signs[b].to(state.dtype).view(1, -1)
        z_a_x_b = flip_b * z_signs[a].to(state.dtype).view(1, -1)
        values.append((state.conj() * (x_a_z_b + z_a_x_b)).sum(dim=1).real)
    return torch.stack(values, dim=1)


class D4OrbitQuantumBottleneck(nn.Module):
    """Several reusable eight-qubit D4-equivariant circuits.

    Input shape is ``(batch, heads, 2, 8)``. The two features per qubit are
    densely angle encoded with RY/RZ and re-uploaded at every layer. The
    returned features are exact D4 invariants assembled from local Z/X and
    Cayley-edge ZZ/XX expectations.
    """

    parameters_per_layer = 11
    invariants_per_head = 12

    def __init__(
        self,
        heads: int = 4,
        reuploads: int = 2,
        n_qubits: int = 8,
        input_encoding: Literal["angle", "boltzmann", "gibbs"] = "angle",
        observable_readout: Literal[
            "pair", "plaquette", "cayley-complete"
        ] = "pair",
        r2_entanglers: bool = False,
        equatorial_readout: bool = False,
        meridional_readout: bool = False,
    ) -> None:
        super().__init__()
        if n_qubits != len(D4_ELEMENTS):
            raise ValueError("The D4 regular register requires exactly 8 qubits")
        if input_encoding not in ("angle", "boltzmann", "gibbs"):
            raise ValueError(f"Unknown quantum input encoding: {input_encoding}")
        if observable_readout not in ("pair", "plaquette", "cayley-complete"):
            raise ValueError(f"Unknown observable readout: {observable_readout}")
        if equatorial_readout and observable_readout != "pair":
            raise ValueError("Equatorial readout requires the 12-feature pair readout")
        if meridional_readout and observable_readout != "pair":
            raise ValueError("Meridional readout requires the 12-feature pair readout")
        if sum((bool(r2_entanglers), bool(equatorial_readout), bool(meridional_readout))) > 1:
            raise ValueError(
                "R2 entanglers, equatorial readout, and meridional readout "
                "are separate experiments"
            )
        self.heads = heads
        self.reuploads = reuploads
        self.n_qubits = n_qubits
        self.input_encoding = input_encoding
        self.observable_readout = observable_readout
        self.r2_entanglers = bool(r2_entanglers)
        self.equatorial_readout = bool(equatorial_readout)
        self.meridional_readout = bool(meridional_readout)
        self.invariants_per_head = {
            "pair": 12,
            "plaquette": 16,
            "cayley-complete": 28,
        }[observable_readout]

        params = torch.zeros(heads, reuploads, self.parameters_per_layer)
        params[..., 0] = 1.0
        params[..., 2] = 1.0
        params[..., 4:] = 0.02 * torch.randn_like(params[..., 4:])
        self.params = nn.Parameter(params)
        if self.r2_entanglers:
            # Keep the half-turn edge family separate from the established
            # eleven-parameter layer so legacy checkpoints and default state
            # dictionaries remain unchanged.  Zero angles exactly recover the
            # original R/S-only circuit.
            self.r2_params = nn.Parameter(torch.zeros(heads, reuploads, 2))
        else:
            self.register_parameter("r2_params", None)
        if self.equatorial_readout:
            # Per head: local, R-edge, R2-edge, and S-edge measurement-basis
            # phases.  P(phi)=cos(phi)X+sin(phi)Y is physically a tied virtual
            # RZ rotation followed by X measurement.  Exact zeros recover all
            # six established X-sector invariants without changing their
            # width or the downstream classifier.
            self.readout_phases = nn.Parameter(torch.zeros(heads, 4))
        else:
            self.register_parameter("readout_phases", None)
        if self.meridional_readout:
            # Per head: local, R-edge, R2-edge, and S-edge measurement-basis
            # phases.  P(phi)=cos(phi)X+sin(phi)Z is a tied virtual RY rotation
            # followed by X measurement.  Exact zeros recover the established
            # X-sector invariants without changing their width or classifier.
            self.meridional_phases = nn.Parameter(torch.zeros(heads, 4))
        else:
            self.register_parameter("meridional_phases", None)

        if input_encoding == "gibbs":
            # Per head: five local-field coefficients, three Cayley-family
            # couplings, three input-dependent coupling modulations, and one
            # inverse-temperature parameter.  The resulting positive real
            # amplitudes form a D4-covariant, generally entangled Gibbs state.
            energy_params = torch.zeros(heads, 12)
            energy_params[:, 0:2] = 0.25
            energy_params[:, 5:8] = 0.02
            self.energy_params = nn.Parameter(energy_params)
        else:
            self.register_parameter("energy_params", None)

        basis = torch.arange(1 << n_qubits)
        signs = []
        for q in range(n_qubits):
            bit = (basis & _bit_mask(q, n_qubits)) != 0
            signs.append(torch.where(bit, -torch.ones_like(basis), torch.ones_like(basis)))
        self.register_buffer("z_signs", torch.stack(signs).float(), persistent=False)
        self.register_buffer("r_edges", torch.tensor(R_EDGES, dtype=torch.long), persistent=False)
        self.register_buffer("r2_edges", torch.tensor(R2_EDGES, dtype=torch.long), persistent=False)
        self.register_buffer("s_edges", torch.tensor(S_EDGES, dtype=torch.long), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.heads * self.invariants_per_head

    def _edge_rotation(
        self, state: torch.Tensor, theta: torch.Tensor, edges: torch.Tensor, pauli: str
    ) -> torch.Tensor:
        for a, b in edges.tolist():
            state = _apply_pair_rotation(state, theta, a, b, pauli, self.z_signs)
        return state

    def _run_statevector(self, orbit_features: torch.Tensor) -> torch.Tensor:
        batch = orbit_features.shape[0]
        flat = orbit_features.reshape(batch * self.heads, 2, self.n_qubits).float()
        p = self.params.unsqueeze(0).expand(batch, -1, -1, -1)
        p = p.reshape(batch * self.heads, self.reuploads, self.parameters_per_layer).float()
        r2_p = None
        if self.r2_params is not None:
            r2_p = self.r2_params.unsqueeze(0).expand(batch, -1, -1, -1)
            r2_p = r2_p.reshape(
                batch * self.heads, self.reuploads, 2
            ).float()
        if self.input_encoding == "gibbs":
            state = self._gibbs_state(flat, batch)
        else:
            state = torch.zeros(
                flat.shape[0],
                1 << self.n_qubits,
                dtype=torch.complex64,
                device=flat.device,
            )
            state[:, 0] = 1.0 + 0.0j

        for layer in range(self.reuploads):
            layer_p = p[:, layer]
            if self.input_encoding == "boltzmann":
                inverse_temperature = F.softplus(layer_p[:, 0]).unsqueeze(1) + 1e-4
                distribution = torch.softmax(
                    -inverse_temperature * flat[:, 0], dim=1
                )
                amplitude_angles = 2.0 * torch.asin(
                    distribution.clamp_min(1e-12).sqrt()
                )
            for qubit in range(self.n_qubits):
                if self.input_encoding == "boltzmann":
                    ry_angle = amplitude_angles[:, qubit] + layer_p[:, 1]
                else:
                    ry_angle = layer_p[:, 0] * flat[:, 0, qubit] + layer_p[:, 1]
                rz_angle = layer_p[:, 2] * flat[:, 1, qubit] + layer_p[:, 3]
                state = _apply_ry(state, ry_angle, qubit, self.n_qubits)
                state = _apply_rz(state, rz_angle, qubit, self.n_qubits)
            for qubit in range(self.n_qubits):
                state = _apply_rx(state, layer_p[:, 4], qubit, self.n_qubits)
                state = _apply_ry(state, layer_p[:, 5], qubit, self.n_qubits)
                state = _apply_rz(state, layer_p[:, 6], qubit, self.n_qubits)
            state = self._edge_rotation(state, layer_p[:, 7], self.r_edges, "z")
            state = self._edge_rotation(state, layer_p[:, 8], self.s_edges, "z")
            if r2_p is not None:
                state = self._edge_rotation(
                    state, r2_p[:, layer, 0], self.r2_edges, "z"
                )
            state = self._edge_rotation(state, layer_p[:, 9], self.r_edges, "x")
            state = self._edge_rotation(state, layer_p[:, 10], self.s_edges, "x")
            if r2_p is not None:
                state = self._edge_rotation(
                    state, r2_p[:, layer, 1], self.r2_edges, "x"
                )
        return state

    def _gibbs_state(self, flat: torch.Tensor, batch: int) -> torch.Tensor:
        """Prepare a neural pairwise-Gibbs amplitude state over eight spins.

        Direct amplitude construction is exact and inexpensive at eight qubits,
        but a hardware implementation would require a nontrivial state-
        preparation circuit; this simulator path must not be presented as a
        shallow native NISQ encoding.
        """

        energy = self.energy_params.unsqueeze(0).expand(batch, -1, -1)
        energy = energy.reshape(batch * self.heads, 12).float()
        u, v = flat[:, 0], flat[:, 1]
        local_field = (
            energy[:, 0, None] * u
            + energy[:, 1, None] * v
            + energy[:, 2, None] * u * v
            + energy[:, 3, None] * (u.square() + v.square())
            + energy[:, 4, None] * (u.square() - v.square())
        )
        score = local_field @ self.z_signs

        for family, edges in enumerate((R_EDGES, R2_EDGES, S_EDGES)):
            edge_similarity = torch.stack(
                [u[:, a] * u[:, b] + v[:, a] * v[:, b] for a, b in edges], dim=1
            )
            coupling = energy[:, 5 + family, None] * (
                1.0 + energy[:, 8 + family, None] * torch.tanh(edge_similarity)
            )
            spin_products = torch.stack(
                [self.z_signs[a] * self.z_signs[b] for a, b in edges], dim=0
            )
            score = score + coupling @ spin_products

        inverse_temperature = F.softplus(energy[:, 11]) + 1e-4
        log_weight = inverse_temperature[:, None] * score
        log_normalizer = torch.logsumexp(log_weight, dim=1, keepdim=True)
        amplitudes = torch.exp(0.5 * (log_weight - log_normalizer))
        return amplitudes.to(torch.complex64)

    def forward(
        self, orbit_features: torch.Tensor, return_equivariant: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if orbit_features.ndim != 4 or orbit_features.shape[1:] != (
            self.heads,
            2,
            self.n_qubits,
        ):
            raise ValueError(
                f"Expected (B,{self.heads},2,{self.n_qubits}), got {tuple(orbit_features.shape)}"
            )
        with torch.autocast(device_type=orbit_features.device.type, enabled=False):
            state = self._run_statevector(orbit_features)
            probabilities = state.abs().square()
            z = probabilities @ self.z_signs.transpose(0, 1)
            x = _expectation_x(state, [(q,) for q in range(self.n_qubits)], self.n_qubits)
            pair_families = (R_EDGES, R2_EDGES, S_EDGES)
            readout_families = (
                CAYLEY_EDGE_FAMILIES
                if self.observable_readout == "cayley-complete"
                else pair_families
            )
            zz_readout = tuple(
                _edge_expectation_z(probabilities, self.z_signs, edges)
                for edges in readout_families
            )
            xx_readout = tuple(
                _expectation_x(state, edges, self.n_qubits)
                for edges in readout_families
            )
            zz_pair, xx_pair = zz_readout[:3], xx_readout[:3]

            def edge_product(values: torch.Tensor, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
                return torch.stack([values[:, a] * values[:, b] for a, b in edges], dim=1)

            z_mean = z.mean(dim=1)
            x_mean = x.mean(dim=1)
            equatorial_fields = None
            meridional_fields = None
            if self.readout_phases is not None:
                y = _expectation_pauli_strings(
                    state,
                    [[(q, "Y")] for q in range(self.n_qubits)],
                    self.n_qubits,
                    self.z_signs,
                )
                batch = orbit_features.shape[0]
                phases = self.readout_phases.unsqueeze(0).expand(batch, -1, -1)
                phases = phases.reshape(batch * self.heads, 4).float()
                cosine, sine = phases.cos(), phases.sin()
                equatorial_fields = (
                    cosine[:, :, None] * x[:, None, :]
                    + sine[:, :, None] * y[:, None, :]
                )

                rotated_pairs = []
                for family_index, edges in enumerate(pair_families):
                    yy = _expectation_pauli_strings(
                        state,
                        [[(a, "Y"), (b, "Y")] for a, b in edges],
                        self.n_qubits,
                        self.z_signs,
                    )
                    xy_plus_yx = _expectation_pauli_strings(
                        state,
                        [[(a, "X"), (b, "Y")] for a, b in edges],
                        self.n_qubits,
                        self.z_signs,
                    ) + _expectation_pauli_strings(
                        state,
                        [[(a, "Y"), (b, "X")] for a, b in edges],
                        self.n_qubits,
                        self.z_signs,
                    )
                    c = cosine[:, family_index + 1, None]
                    s = sine[:, family_index + 1, None]
                    rotated_pairs.append(
                        c.square() * xx_pair[family_index]
                        + s.square() * yy
                        + c * s * xy_plus_yx
                    )

                local_equatorial = equatorial_fields[:, 0]
                local_mean = local_equatorial.mean(dim=1)
                r_equatorial = equatorial_fields[:, 1]
                invariant_features = [
                    z_mean,
                    z.square().mean(dim=1) - z_mean.square(),
                    local_mean,
                    local_equatorial.square().mean(dim=1)
                    - local_mean.square(),
                    *(values.mean(dim=1) for values in zz_pair),
                    *(values.mean(dim=1) for values in rotated_pairs),
                    (zz_pair[0] - edge_product(z, R_EDGES)).mean(dim=1),
                    (
                        rotated_pairs[0]
                        - edge_product(r_equatorial, R_EDGES)
                    ).mean(dim=1),
                ]
            elif self.meridional_phases is not None:
                batch = orbit_features.shape[0]
                phases = self.meridional_phases.unsqueeze(0).expand(
                    batch, -1, -1
                )
                phases = phases.reshape(batch * self.heads, 4).float()
                cosine, sine = phases.cos(), phases.sin()
                meridional_fields = (
                    cosine[:, :, None] * x[:, None, :]
                    + sine[:, :, None] * z[:, None, :]
                )

                rotated_pairs = []
                for family_index, edges in enumerate(pair_families):
                    xz_plus_zx = _expectation_symmetric_xz(
                        state,
                        edges,
                        self.n_qubits,
                        self.z_signs,
                    )
                    c = cosine[:, family_index + 1, None]
                    s = sine[:, family_index + 1, None]
                    rotated_pairs.append(
                        c.square() * xx_pair[family_index]
                        + s.square() * zz_pair[family_index]
                        + c * s * xz_plus_zx
                    )

                local_meridional = meridional_fields[:, 0]
                local_mean = local_meridional.mean(dim=1)
                r_meridional = meridional_fields[:, 1]
                invariant_features = [
                    z_mean,
                    z.square().mean(dim=1) - z_mean.square(),
                    local_mean,
                    local_meridional.square().mean(dim=1)
                    - local_mean.square(),
                    *(values.mean(dim=1) for values in zz_pair),
                    *(values.mean(dim=1) for values in rotated_pairs),
                    (zz_pair[0] - edge_product(z, R_EDGES)).mean(dim=1),
                    (
                        rotated_pairs[0]
                        - edge_product(r_meridional, R_EDGES)
                    ).mean(dim=1),
                ]
            else:
                invariant_features = [
                    z_mean,
                    z.square().mean(dim=1) - z_mean.square(),
                    x_mean,
                    x.square().mean(dim=1) - x_mean.square(),
                    *(values.mean(dim=1) for values in zz_pair),
                    *(values.mean(dim=1) for values in xx_pair),
                    (zz_pair[0] - edge_product(z, R_EDGES)).mean(dim=1),
                    (xx_pair[0] - edge_product(x, R_EDGES)).mean(dim=1),
                ]
            if self.observable_readout == "cayley-complete":
                # Complete the two-point displacement spectrum of D4.  Each
                # family is a full left-edge orbit and is therefore invariant
                # under the induced right action after orbit averaging.
                zz_all, xx_all = zz_readout, xx_readout
                invariant_features = invariant_features[:4]
                invariant_features.extend(
                    values.mean(dim=1) for values in zz_all
                )
                invariant_features.extend(
                    values.mean(dim=1) for values in xx_all
                )
                invariant_features.extend(
                    (values - edge_product(z, edges)).mean(dim=1)
                    for values, edges in zip(zz_all, CAYLEY_EDGE_FAMILIES)
                )
                invariant_features.extend(
                    (values - edge_product(x, edges)).mean(dim=1)
                    for values, edges in zip(xx_all, CAYLEY_EDGE_FAMILIES)
                )
            elif self.observable_readout == "plaquette":
                # These orbit averages are four-body morphology probes.  They
                # add neither gates nor trainable parameters and reuse the
                # existing all-Z and all-X measurement settings.
                invariant_features.extend(
                    (
                        _expectation_z(
                            probabilities, self.z_signs, RS_PLAQUETTES
                        ).mean(dim=1),
                        _expectation_x(
                            state, RS_PLAQUETTES, self.n_qubits
                        ).mean(dim=1),
                        _expectation_z(
                            probabilities, self.z_signs, R2S_PLAQUETTES
                        ).mean(dim=1),
                        _expectation_x(
                            state, R2S_PLAQUETTES, self.n_qubits
                        ).mean(dim=1),
                    )
                )
            invariant = torch.stack(invariant_features, dim=1)
            invariant = invariant.reshape(orbit_features.shape[0], self.output_dim)

        if not return_equivariant:
            return invariant
        equivariant = {
            "z": z.reshape(orbit_features.shape[0], self.heads, self.n_qubits),
            "x": x.reshape(orbit_features.shape[0], self.heads, self.n_qubits),
        }
        if equatorial_fields is not None:
            equivariant["equatorial"] = equatorial_fields.reshape(
                orbit_features.shape[0], self.heads, 4, self.n_qubits
            )
        if meridional_fields is not None:
            equivariant["meridional"] = meridional_fields.reshape(
                orbit_features.shape[0], self.heads, 4, self.n_qubits
            )
        return invariant, equivariant

    def parameter_report(self) -> Dict[str, int | str]:
        return {
            "qubits": self.n_qubits,
            "heads": self.heads,
            "reuploads": self.reuploads,
            "quantum_trainable": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "input_encoding": self.input_encoding,
            "observable_readout": self.observable_readout,
            "r2_entanglers": self.r2_entanglers,
            "r2_entangler_trainable": (
                0 if self.r2_params is None else self.r2_params.numel()
            ),
            "equatorial_readout": self.equatorial_readout,
            "equatorial_readout_trainable": (
                0 if self.readout_phases is None else self.readout_phases.numel()
            ),
            "meridional_readout": self.meridional_readout,
            "meridional_readout_trainable": (
                0
                if self.meridional_phases is None
                else self.meridional_phases.numel()
            ),
            "state_preparation_trainable": sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if name not in ("readout_phases", "meridional_phases")
            ),
            "energy_trainable": 0 if self.energy_params is None else self.energy_params.numel(),
            "statevector_dimension": 1 << self.n_qubits,
            "invariants": self.output_dim,
        }
