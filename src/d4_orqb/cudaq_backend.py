"""CUDA-Q inference/parity backend for the base D4 quantum bottleneck.

This file deliberately has no top-level PyTorch or CUDA-Q import.  It can be
run directly in NVIDIA's CUDA-Q container (which need not contain PyTorch):

    python cudaq_backend.py verify-fixture --fixture parity.npz --target nvidia

Create the fixture in the training environment, where PyTorch is available:

    python cudaq_backend.py export-fixture --output parity.npz \
        --run-dir /path/to/run --checkpoint /path/to/run/best.pt \
        --development-cache /path/to/cache/development --split validation

Only the established base configuration is implemented: four heads, three
data-reupload layers, eight D4 qubits, angle encoding, pair readout, and no
experimental R2 entanglers or rotated readout bases.  The CUDA-Q path is for
analytic inference/parity, not gradient-based training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import linecache
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


HEADS = 4
REUPLOADS = 3
N_QUBITS = 8
PARAMETERS_PER_LAYER = 11
INVARIANTS_PER_HEAD = 12
FEATURES_PER_HEAD = 2 * N_QUBITS
PARAMETERS_PER_HEAD = REUPLOADS * PARAMETERS_PER_LAYER
FIXTURE_SCHEMA = "d4-orqb-cudaq-parity-v1"

# These are copied from d4_orqb.quantum rather than imported so that the
# CUDA-Q-only verification image does not need PyTorch.
R_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 3),
    (1, 2),
    (2, 3),
    (4, 5),
    (4, 7),
    (5, 6),
    (6, 7),
)
R2_EDGES: tuple[tuple[int, int], ...] = ((0, 2), (1, 3), (4, 6), (5, 7))
S_EDGES: tuple[tuple[int, int], ...] = ((0, 4), (1, 7), (2, 6), (3, 5))
PAIR_FAMILIES = (R_EDGES, R2_EDGES, S_EDGES)


class CudaQUnavailableError(RuntimeError):
    """Raised when CUDA-Q is requested but cannot be imported."""


class CudaQConfigurationError(ValueError):
    """Raised for a circuit configuration that this parity backend cannot run."""


@dataclass(frozen=True)
class ObservableSpec:
    """One coefficient-one Pauli observable in the analytic readout."""

    label: str
    pauli: str
    wires: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.pauli not in ("X", "Z"):
            raise ValueError(f"Unsupported Pauli family: {self.pauli}")
        if not self.wires or len(set(self.wires)) != len(self.wires):
            raise ValueError(f"Invalid wires: {self.wires}")
        if any(wire < 0 or wire >= N_QUBITS for wire in self.wires):
            raise ValueError(f"Wire outside the D4 register: {self.wires}")

    @property
    def word(self) -> str:
        values = ["I"] * N_QUBITS
        for wire in self.wires:
            values[wire] = self.pauli
        return "".join(values)


def _observable_specs() -> tuple[ObservableSpec, ...]:
    specs: list[ObservableSpec] = []
    specs.extend(ObservableSpec(f"Z{q}", "Z", (q,)) for q in range(N_QUBITS))
    specs.extend(ObservableSpec(f"X{q}", "X", (q,)) for q in range(N_QUBITS))
    for pauli in ("Z", "X"):
        for family_name, edges in zip(("R", "R2", "S"), PAIR_FAMILIES):
            specs.extend(
                ObservableSpec(f"{pauli}{pauli}_{family_name}_{a}_{b}", pauli, (a, b))
                for a, b in edges
            )
    return tuple(specs)


OBSERVABLE_SPECS = _observable_specs()
OBSERVABLE_COUNT = len(OBSERVABLE_SPECS)


def validate_base_configuration(
    *,
    heads: int = HEADS,
    reuploads: int = REUPLOADS,
    n_qubits: int = N_QUBITS,
    input_encoding: str = "angle",
    observable_readout: str = "pair",
    r2_entanglers: bool = False,
    equatorial_readout: bool = False,
    meridional_readout: bool = False,
) -> None:
    """Reject anything that is not the checkpoint-compatible base circuit."""

    actual = {
        "heads": heads,
        "reuploads": reuploads,
        "n_qubits": n_qubits,
        "input_encoding": input_encoding,
        "observable_readout": observable_readout,
        "r2_entanglers": bool(r2_entanglers),
        "equatorial_readout": bool(equatorial_readout),
        "meridional_readout": bool(meridional_readout),
    }
    expected = {
        "heads": HEADS,
        "reuploads": REUPLOADS,
        "n_qubits": N_QUBITS,
        "input_encoding": "angle",
        "observable_readout": "pair",
        "r2_entanglers": False,
        "equatorial_readout": False,
        "meridional_readout": False,
    }
    if actual != expected:
        differences = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        raise CudaQConfigurationError(
            "CUDA-Q parity only supports the base 4-head/3-layer angle/pair "
            f"circuit; mismatches={differences}"
        )


def validate_arrays(
    orbit_features: np.ndarray, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize host arrays without silently changing layout."""

    features = np.asarray(orbit_features)
    params = np.asarray(parameters)
    if features.ndim != 4 or features.shape[1:] != (HEADS, 2, N_QUBITS):
        raise ValueError(
            f"Expected orbit features (B,{HEADS},2,{N_QUBITS}), got {features.shape}"
        )
    if params.shape != (HEADS, REUPLOADS, PARAMETERS_PER_LAYER):
        raise ValueError(
            "Expected circuit parameters "
            f"({HEADS},{REUPLOADS},{PARAMETERS_PER_LAYER}), got {params.shape}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise TypeError(f"Orbit features must be floating point, got {features.dtype}")
    if not np.issubdtype(params.dtype, np.floating):
        raise TypeError(f"Circuit parameters must be floating point, got {params.dtype}")
    if not np.isfinite(features).all() or not np.isfinite(params).all():
        raise ValueError("Orbit features and circuit parameters must be finite")
    # CUDA-Q list[float] arguments are host doubles.  Preserving the source
    # float32 values exactly in float64 avoids an extra rounding step.
    return (
        np.ascontiguousarray(features, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
    )


def pack_head_arguments(
    orbit_features: np.ndarray, parameters: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Pack each (sample, head) exactly as consumed by the CUDA-Q kernel.

    Features are channel-major: eight RY features followed by eight RZ
    features.  Parameters are layer-major with eleven scalars per layer.
    """

    features, params = validate_arrays(orbit_features, parameters)
    packed: list[tuple[np.ndarray, np.ndarray]] = []
    for sample in range(features.shape[0]):
        for head in range(HEADS):
            packed.append(
                (
                    features[sample, head].reshape(FEATURES_PER_HEAD).copy(),
                    params[head].reshape(PARAMETERS_PER_HEAD).copy(),
                )
            )
    return tuple(packed)


def cudaq_exp_pauli_coefficient(torch_pair_angle: float | np.ndarray) -> Any:
    """Map ``exp(-i theta P/2)`` to CUDA-Q's ``exp_pauli`` coefficient.

    CUDA-Q defines ``exp_pauli(alpha, ..., P)`` as ``exp(+i alpha P)``.
    Consequently the faithful coefficient is ``alpha = -theta / 2``.
    """

    return -0.5 * np.asarray(torch_pair_angle)


def invariants_from_expectations(expectations: np.ndarray) -> np.ndarray:
    """Assemble the exact 12-feature pair readout from 48 expectations.

    The trailing expectation layout is given by ``OBSERVABLE_SPECS``:
    local Z, local X, ZZ over R/R2/S, then XX over R/R2/S.
    """

    values = np.asarray(expectations, dtype=np.float64)
    if values.shape[-1] != OBSERVABLE_COUNT:
        raise ValueError(
            f"Expected {OBSERVABLE_COUNT} observables, got shape {values.shape}"
        )
    cursor = 0
    z = values[..., cursor : cursor + N_QUBITS]
    cursor += N_QUBITS
    x = values[..., cursor : cursor + N_QUBITS]
    cursor += N_QUBITS
    zz: list[np.ndarray] = []
    xx: list[np.ndarray] = []
    for destination in (zz, xx):
        for edges in PAIR_FAMILIES:
            destination.append(values[..., cursor : cursor + len(edges)])
            cursor += len(edges)
    if cursor != OBSERVABLE_COUNT:
        raise AssertionError("Internal observable layout mismatch")

    z_mean = z.mean(axis=-1)
    x_mean = x.mean(axis=-1)
    r_z_product = np.stack([z[..., a] * z[..., b] for a, b in R_EDGES], axis=-1)
    r_x_product = np.stack([x[..., a] * x[..., b] for a, b in R_EDGES], axis=-1)
    result = np.stack(
        (
            z_mean,
            np.square(z).mean(axis=-1) - np.square(z_mean),
            x_mean,
            np.square(x).mean(axis=-1) - np.square(x_mean),
            *(family.mean(axis=-1) for family in zz),
            *(family.mean(axis=-1) for family in xx),
            (zz[0] - r_z_product).mean(axis=-1),
            (xx[0] - r_x_product).mean(axis=-1),
        ),
        axis=-1,
    )
    if result.shape[-1] != INVARIANTS_PER_HEAD:
        raise AssertionError("Internal invariant width mismatch")
    return result


def _require_cudaq() -> Any:
    try:
        return importlib.import_module("cudaq")
    except (ImportError, ModuleNotFoundError) as exc:
        raise CudaQUnavailableError(
            "CUDA-Q is not installed. Run verify-fixture in an NVIDIA CUDA-Q "
            "container (the parity job uses the official CUDA-Q 0.12 image)."
        ) from exc


def _kernel_source() -> str:
    """Generate the fixed CUDA-Q kernel using native gate decompositions.

    ``CX-RZ(theta)-CX`` is exactly ``exp(-i theta ZZ / 2)``.  Conjugating
    that sequence by Hadamards on both wires gives ``exp(-i theta XX / 2)``.
    This is equivalent to passing ``-theta/2`` to CUDA-Q ``exp_pauli`` (whose
    convention is ``exp(+i alpha P)``), but avoids a sign/convention dependency
    across CUDA-Q releases.  The feature and parameter loops have fixed bounds.
    """

    edge_lines: list[str] = []
    for parameter_index, pauli, edges in (
        (7, "Z", R_EDGES),
        (8, "Z", S_EDGES),
        (9, "X", R_EDGES),
        (10, "X", S_EDGES),
    ):
        for a, b in edges:
            if pauli == "X":
                edge_lines.extend(
                    (f"        h(qubits[{a}])", f"        h(qubits[{b}])")
                )
            edge_lines.extend(
                (
                    f"        x.ctrl(qubits[{a}], qubits[{b}])",
                    f"        rz(parameters[offset + {parameter_index}], qubits[{b}])",
                    f"        x.ctrl(qubits[{a}], qubits[{b}])",
                )
            )
            if pauli == "X":
                edge_lines.extend(
                    (f"        h(qubits[{a}])", f"        h(qubits[{b}])")
                )
    entanglers = "\n".join(edge_lines)
    return f'''@cudaq.kernel
def _d4_orbit_cudaq_kernel(features: list[float], parameters: list[float]):
    qubits = cudaq.qvector({N_QUBITS})
    for layer in range({REUPLOADS}):
        offset = layer * {PARAMETERS_PER_LAYER}
        for qubit in range({N_QUBITS}):
            ry(parameters[offset] * features[qubit] + parameters[offset + 1], qubits[qubit])
            rz(parameters[offset + 2] * features[{N_QUBITS} + qubit] + parameters[offset + 3], qubits[qubit])
        for qubit in range({N_QUBITS}):
            rx(parameters[offset + 4], qubits[qubit])
            ry(parameters[offset + 5], qubits[qubit])
            rz(parameters[offset + 6], qubits[qubit])
{entanglers}
'''


def _build_kernel(cudaq: Any) -> Any:
    # exec is intentional: a top-level @cudaq.kernel would make importing this
    # utility impossible in the PyTorch-only training environment.
    source = _kernel_source()
    filename = "<d4_orqb_cudaq_kernel>"
    # CUDA-Q's decorator retrieves the Python AST through inspect/linecache.
    # Register generated source so the decorator sees the same text we compile.
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {"cudaq": cudaq, "__name__": __name__}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["_d4_orbit_cudaq_kernel"]


def _spin_observable(cudaq: Any, spec: ObservableSpec) -> Any:
    constructor = cudaq.spin.x if spec.pauli == "X" else cudaq.spin.z
    operator = constructor(spec.wires[0])
    for wire in spec.wires[1:]:
        operator = operator * constructor(wire)
    return operator


def _value_or_call(value: Any) -> Any:
    try:
        return value() if callable(value) else value
    except TypeError:
        return str(value)


def _target_metadata(cudaq: Any, requested: str) -> dict[str, Any]:
    has_target = getattr(cudaq, "has_target", None)
    if callable(has_target) and not bool(has_target(requested)):
        names = []
        get_targets = getattr(cudaq, "get_targets", None)
        if callable(get_targets):
            for target in get_targets():
                names.append(str(_value_or_call(getattr(target, "name", target))))
        raise RuntimeError(
            f"CUDA-Q target {requested!r} is unavailable; available={names}"
        )
    cudaq.set_target(requested)
    selected = cudaq.get_target()
    report: dict[str, Any] = {"requested": requested}
    for field in (
        "name",
        "description",
        "platform",
        "simulator",
        "num_qpus",
        "is_emulated",
        "is_remote",
        "get_precision",
    ):
        if hasattr(selected, field):
            value = _value_or_call(getattr(selected, field))
            if not isinstance(value, (str, int, float, bool, type(None))):
                value = str(value)
            report[field] = value
    return report


def _nvidia_smi_report() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "available": result.returncode == 0 and bool(lines),
        "returncode": result.returncode,
        "gpus": lines,
        "stderr": result.stderr.strip(),
    }


class CudaQD4OrbitBackend:
    """Analytic CUDA-Q inference for the fixed base D4 bottleneck."""

    def __init__(self, target: str = "nvidia", *, require_nvidia: bool = True) -> None:
        validate_base_configuration()
        if require_nvidia and not target.startswith("nvidia"):
            raise CudaQConfigurationError(
                f"A CUDA-Q NVIDIA target is required, got {target!r}"
            )
        self.target = target
        self.require_nvidia = require_nvidia
        self._cudaq: Any | None = None
        self._kernel: Any | None = None
        self._observables: list[Any] | None = None
        self.target_report: dict[str, Any] | None = None

    def initialize(self) -> dict[str, Any]:
        if self._cudaq is not None:
            assert self.target_report is not None
            return self.target_report
        cudaq = _require_cudaq()
        target = _target_metadata(cudaq, self.target)
        self._cudaq = cudaq
        self._kernel = _build_kernel(cudaq)
        self._observables = [_spin_observable(cudaq, spec) for spec in OBSERVABLE_SPECS]
        self.target_report = {
            "cudaq_version": str(getattr(cudaq, "__version__", "unknown")),
            "target": target,
            "nvidia_smi": _nvidia_smi_report(),
        }
        return self.target_report

    def _expectations_for_head(
        self, features: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        assert self._cudaq is not None
        assert self._kernel is not None
        assert self._observables is not None
        # Use one primitive observe call per operator.  This API is supported by
        # the cluster-compatible CUDA-Q 0.12 image; list-of-operator observe was
        # added/changed across later releases.  No shots_count is supplied, so
        # simulator expectations are analytic rather than finite-shot samples.
        arguments = (features.tolist(), parameters.tolist())
        values = [
            self._cudaq.observe(
                self._kernel, observable, *arguments
            ).expectation()
            for observable in self._observables
        ]
        return np.asarray(values, dtype=np.float64)

    def __call__(self, orbit_features: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        features, params = validate_arrays(orbit_features, parameters)
        self.initialize()
        head_expectations = np.empty(
            (features.shape[0], HEADS, OBSERVABLE_COUNT), dtype=np.float64
        )
        for sample in range(features.shape[0]):
            for head in range(HEADS):
                head_expectations[sample, head] = self._expectations_for_head(
                    features[sample, head].reshape(FEATURES_PER_HEAD),
                    params[head].reshape(PARAMETERS_PER_HEAD),
                )
        invariants = invariants_from_expectations(head_expectations)
        return invariants.reshape(features.shape[0], HEADS * INVARIANTS_PER_HEAD)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    for key in ("model", "model_state", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return checkpoint


def _checkpoint_parameters(
    checkpoint_path: Path, parameter_key: str | None = None
) -> tuple[np.ndarray, str]:
    # Lazy import keeps verify-fixture usable in CUDA-Q images without PyTorch.
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    if parameter_key is not None:
        if parameter_key not in state:
            raise KeyError(f"Parameter key {parameter_key!r} not found in checkpoint")
        candidates = [(parameter_key, state[parameter_key])]
    else:
        candidates = [
            (key, value)
            for key, value in state.items()
            if key == "params" or key.endswith(".params")
        ]
    shaped = [
        (key, value)
        for key, value in candidates
        if hasattr(value, "shape")
        and tuple(value.shape) == (HEADS, REUPLOADS, PARAMETERS_PER_LAYER)
    ]
    if len(shaped) != 1:
        found = [(key, tuple(value.shape)) for key, value in candidates if hasattr(value, "shape")]
        raise ValueError(
            "Expected exactly one base quantum parameter tensor with shape "
            f"({HEADS},{REUPLOADS},{PARAMETERS_PER_LAYER}); found={found}. "
            "Use --parameter-key if the checkpoint has multiple quantum branches."
        )
    key, tensor = shaped[0]
    # ``tolist`` is a robust bridge even in minimal containers whose PyTorch
    # and NumPy wheels were built against different NumPy C-API revisions.
    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32), key


def export_fixture(
    output: Path,
    *,
    run_dir: Path,
    checkpoint: Path,
    development_cache: Path,
    split: str,
    parameter_key: str | None,
    seed: int,
    max_samples: int,
) -> dict[str, Any]:
    """Export real split inputs, projected angles, and PyTorch invariants."""

    import torch

    # When this file is executed directly, add src/ so the package is visible.
    src_root = str(Path(__file__).resolve().parents[1])
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from d4_orqb.model import D4OrbitClassifier

    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    run_dir = run_dir.resolve()
    checkpoint = checkpoint.resolve()
    development_cache = development_cache.resolve()
    config_path = run_dir / "config.json"
    split_path = run_dir / "split_indices.npz"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for name in ("images.npy", "labels.npy"):
        if not (development_cache / name).is_file():
            raise FileNotFoundError(development_cache / name)

    config = json.loads(config_path.read_text())
    if config.get("core") != "quantum":
        raise CudaQConfigurationError(
            f"Checkpoint core must be 'quantum', got {config.get('core')!r}"
        )
    validate_base_configuration(
        heads=int(config.get("heads", HEADS)),
        reuploads=int(config.get("reuploads", REUPLOADS)),
        n_qubits=N_QUBITS,
        input_encoding=str(config.get("quantum_encoding", "angle")),
        observable_readout=str(config.get("observable_readout", "pair")),
        r2_entanglers=bool(config.get("r2_entanglers", False)),
        equatorial_readout=bool(config.get("equatorial_readout", False)),
        meridional_readout=bool(config.get("meridional_readout", False)),
    )
    if bool(config.get("cross_scale_reupload", False)):
        raise CudaQConfigurationError(
            "Cross-scale reupload changes the circuit inputs and is not the base backend"
        )

    split_key = {"validation": "val", "train": "train"}.get(split)
    if split_key is None:
        raise ValueError("split must be 'validation' or 'train'")
    with np.load(split_path, allow_pickle=False) as split_file:
        if split_key not in split_file.files:
            raise ValueError(f"{split_path} does not contain split {split_key!r}")
        candidate_indices = np.asarray(split_file[split_key], dtype=np.int64)
    if candidate_indices.ndim != 1 or len(np.unique(candidate_indices)) != len(
        candidate_indices
    ):
        raise ValueError(f"Split {split_key!r} must contain unique one-dimensional indices")
    rng = np.random.default_rng(seed)
    if len(candidate_indices) > max_samples:
        positions = rng.choice(len(candidate_indices), size=max_samples, replace=False)
        selected_indices = candidate_indices[positions]
    else:
        selected_indices = candidate_indices.copy()

    labels_all = np.load(development_cache / "labels.npy", mmap_mode="r")
    images_all = np.load(development_cache / "images.npy", mmap_mode="r")
    if images_all.ndim != 3 or len(images_all) != len(labels_all):
        raise ValueError(
            f"Invalid cache arrays: images={images_all.shape}, labels={labels_all.shape}"
        )
    if len(selected_indices) == 0 or selected_indices.min() < 0 or selected_indices.max() >= len(
        labels_all
    ):
        raise ValueError("Selected split indices are empty or outside the development cache")
    image_array = np.ascontiguousarray(
        images_all[selected_indices], dtype=np.float32
    )
    # ``frombuffer`` avoids requiring PyTorch's optional NumPy C-API bridge;
    # clone makes the tensor own its storage after this local array is released.
    images = torch.frombuffer(
        memoryview(image_array), dtype=torch.float32
    ).reshape(image_array.shape).clone().unsqueeze(1)
    labels = np.asarray(labels_all[selected_indices], dtype=np.int64).copy()

    model_kwargs = {
        "num_classes": int(len(np.unique(np.asarray(labels_all)))),
        "heads": int(config.get("heads", HEADS)),
        "reuploads": int(config.get("reuploads", REUPLOADS)),
        "core": str(config.get("core", "quantum")),
        "include_context": bool(config.get("include_context", False)),
        "dropout": float(config.get("dropout", 0.1)),
        "encoder_variant": str(config.get("encoder_variant", "tiny")),
        "physics_variant": str(config.get("physics_variant", "base")),
        "physics_summary": str(config.get("physics_summary", "none")),
        "quantum_encoding": str(config.get("quantum_encoding", "angle")),
        "observable_readout": str(config.get("observable_readout", "pair")),
        "tied_mean_dispersion": bool(config.get("tied_mean_dispersion", False)),
        "haar_subtype_residual": bool(config.get("haar_subtype_residual", False)),
        "shared_late_refinement": bool(config.get("shared_late_refinement", False)),
        "haar_subtype_max_envelope": bool(
            config.get("haar_subtype_max_envelope", False)
        ),
        "r2_entanglers": bool(config.get("r2_entanglers", False)),
        "equatorial_readout": bool(config.get("equatorial_readout", False)),
        "meridional_readout": bool(config.get("meridional_readout", False)),
        "cross_scale_reupload": bool(config.get("cross_scale_reupload", False)),
    }
    torch.manual_seed(seed)
    model = D4OrbitClassifier(**model_kwargs).eval()
    try:
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
    except TypeError:  # PyTorch < 2.0
        checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    state = _extract_state_dict(checkpoint_payload)
    model.load_state_dict(state, strict=True)
    if parameter_key is not None and parameter_key != "core.params":
        raise ValueError(
            "A full-model base checkpoint exposes parameters as 'core.params'; "
            f"got --parameter-key={parameter_key!r}"
        )
    with torch.no_grad():
        _, angles = model.orbit_encode(images)
        expected_tensor = model.core(angles).cpu()
        angles_array = np.asarray(angles.cpu().tolist(), dtype=np.float32)
        expected = np.asarray(expected_tensor.tolist(), dtype=np.float32)
        parameters = np.asarray(model.core.params.cpu().tolist(), dtype=np.float32)
    batch_size = len(selected_indices)
    cache_metadata_path = development_cache / "metadata.json"
    cache_metadata = (
        json.loads(cache_metadata_path.read_text())
        if cache_metadata_path.is_file()
        else None
    )
    metadata = {
        "schema": FIXTURE_SCHEMA,
        "seed": seed,
        "batch_size": batch_size,
        "split": split,
        "split_key": split_key,
        "split_size": int(len(candidate_indices)),
        "max_samples": max_samples,
        "heads": HEADS,
        "reuploads": REUPLOADS,
        "n_qubits": N_QUBITS,
        "parameters_per_layer": PARAMETERS_PER_LAYER,
        "invariants_per_head": INVARIANTS_PER_HEAD,
        "run_dir": str(run_dir),
        "development_cache": str(development_cache),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_parameter_key": "core.params",
        "model_config": model_kwargs,
        "cache_metadata": cache_metadata,
        "torch_version": str(torch.__version__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(FIXTURE_SCHEMA),
        orbit_features=angles_array,
        parameters=parameters,
        expected_invariants=expected,
        sample_indices=selected_indices,
        sample_labels=labels,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = dict(metadata)
    report.update(
        {
            "status": "exported",
            "fixture": str(output.resolve()),
            "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "feature_shape": list(angles_array.shape),
            "parameter_shape": list(parameters.shape),
            "expected_shape": list(expected.shape),
            "sample_indices": selected_indices.tolist(),
            "sample_labels": labels.tolist(),
        }
    )
    return report


def load_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as fixture:
        required = {
            "schema",
            "orbit_features",
            "parameters",
            "expected_invariants",
            "metadata_json",
        }
        missing = sorted(required.difference(fixture.files))
        if missing:
            raise ValueError(f"Fixture is missing arrays: {missing}")
        schema = str(fixture["schema"].item())
        if schema != FIXTURE_SCHEMA:
            raise ValueError(f"Unsupported fixture schema {schema!r}")
        features = np.asarray(fixture["orbit_features"]).copy()
        params = np.asarray(fixture["parameters"]).copy()
        expected = np.asarray(fixture["expected_invariants"]).copy()
        metadata = json.loads(str(fixture["metadata_json"].item()))
    validate_arrays(features, params)
    if expected.shape != (features.shape[0], HEADS * INVARIANTS_PER_HEAD):
        raise ValueError(
            "Expected invariant fixture shape "
            f"({features.shape[0]},{HEADS * INVARIANTS_PER_HEAD}), got {expected.shape}"
        )
    if not np.isfinite(expected).all():
        raise ValueError("Expected invariants must be finite")
    return features, params, expected, metadata


def verify_fixture(
    fixture: Path,
    *,
    target: str,
    rtol: float,
    atol: float,
    require_nvidia: bool,
) -> tuple[dict[str, Any], bool]:
    features, params, expected, metadata = load_fixture(fixture)
    backend = CudaQD4OrbitBackend(target=target, require_nvidia=require_nvidia)
    actual = backend(features, params)
    absolute = np.abs(actual - expected.astype(np.float64))
    scale = np.maximum(np.abs(expected.astype(np.float64)), atol)
    relative = absolute / scale
    close = np.isclose(actual, expected, rtol=rtol, atol=atol)
    passed = bool(close.all())
    report = {
        "status": "passed" if passed else "failed",
        "fixture": str(fixture.resolve()),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "fixture_metadata": metadata,
        "target_check": backend.target_report,
        "rtol": rtol,
        "atol": atol,
        "values_compared": int(expected.size),
        "mismatches": int((~close).sum()),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "max_rel_error": float(relative.max(initial=0.0)),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
    }
    if not passed:
        flat_index = int(np.argmax(absolute))
        index = tuple(int(v) for v in np.unravel_index(flat_index, absolute.shape))
        report["worst_index"] = list(index)
        report["worst_actual"] = float(actual[index])
        report["worst_expected"] = float(expected[index])
    return report, passed


def _write_report(report: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid a partially written report if a pod is interrupted.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(rendered)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export-fixture", help="write a PyTorch reference NPZ (CUDA-Q not required)"
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--development-cache", type=Path, required=True)
    export.add_argument("--split", choices=("validation", "train"), default="validation")
    export.add_argument("--parameter-key")
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--max-samples", type=int, default=16)
    export.add_argument("--report", type=Path)

    verify = subparsers.add_parser(
        "verify-fixture", help="run CUDA-Q and compare all analytic invariants"
    )
    verify.add_argument("--fixture", type=Path, required=True)
    verify.add_argument("--target", default="nvidia")
    verify.add_argument("--rtol", type=float, default=2e-5)
    verify.add_argument("--atol", type=float, default=2e-5)
    verify.add_argument(
        "--allow-cpu-target",
        action="store_true",
        help="allow qpp-cpu for developer parity tests; production should use nvidia",
    )
    verify.add_argument(
        "--output",
        "--report",
        dest="report",
        type=Path,
        help="write the JSON parity report (also printed to stdout)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export-fixture":
            report = export_fixture(
                args.output,
                run_dir=args.run_dir,
                checkpoint=args.checkpoint,
                development_cache=args.development_cache,
                split=args.split,
                parameter_key=args.parameter_key,
                seed=args.seed,
                max_samples=args.max_samples,
            )
            _write_report(report, args.report)
            return 0
        report, passed = verify_fixture(
            args.fixture,
            target=args.target,
            rtol=args.rtol,
            atol=args.atol,
            require_nvidia=not args.allow_cpu_target,
        )
        _write_report(report, args.report)
        return 0 if passed else 1
    except Exception as exc:
        report = {
            "status": "error",
            "command": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_report(report, getattr(args, "report", None))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
