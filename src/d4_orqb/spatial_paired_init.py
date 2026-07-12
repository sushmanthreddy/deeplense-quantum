"""Build auditable paired initializers for the Model-I spatial-stat core study.

The builder deliberately runs on CPU and never accepts a dataset or test-root
argument.  For each requested seed it preserves the quantum and classical
cores exactly as their native constructors initialized them, gives the
quantum arm a separately seeded common head, and copies every non-core state
tensor from that arm to the classical arm.  The resulting full checkpoints
therefore differ only inside ``core.*`` at epoch zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Literal

import torch

from .model import D4OrbitClassifier
from .train import zero_extend_input_weight


PROTOCOL = "model-i-spatial-stat-paired-initializer-v1"
EXPECTED_BACKBONE_SHA256 = (
    "2d3c49b94f60855279c878af925a34b6e274ee00c81f0d5ae758ceaadb8af200"
)
FORBIDDEN_TEST_MARKER = "model_i_test"
COMMON_HEAD_SEED_OFFSET = 10_000
EXPECTED_PARAMETER_TOTAL = 122_573
EXPECTED_CORE_PARAMETERS = 132
ARCHITECTURE = {
    "num_classes": 3,
    "encoder_variant": "micro-stat",
    "physics_variant": "base",
    "physics_summary": "moments",
    "heads": 4,
    "reuploads": 3,
    "quantum_encoding": "angle",
    "observable_readout": "pair",
    "include_context": False,
    "dropout": 0.10,
    "total_parameters_per_arm": EXPECTED_PARAMETER_TOTAL,
    "core_parameters_per_arm": EXPECTED_CORE_PARAMETERS,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU bytes canonically."""

    digest = hashlib.sha256()
    digest.update(b"d4-orqb-tensor-state-v1\0")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State value is not a tensor: {name}")
        value = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(struct.pack(">Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">Q", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack(">q", int(dimension)))
        # Some production checkpoints preserve channels-last strides.  Hash a
        # canonical flat copy so identical values never depend on serialization
        # memory format and dtype reinterpretation always has unit last stride.
        flat = value.reshape(-1).contiguous()
        raw = flat.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
    }


def _component_state(
    state: Mapping[str, torch.Tensor], *, core: bool
) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if name.startswith("core.") is core
    }


def _module_core_state(model: D4OrbitClassifier) -> Dict[str, torch.Tensor]:
    return {
        f"core.{name}": value.detach().cpu().clone()
        for name, value in model.core.state_dict().items()
    }


def _assert_equal_states(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    description: str,
) -> None:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{description} tensor keys differ: "
            f"actual_only={sorted(set(actual) - set(expected))} "
            f"expected_only={sorted(set(expected) - set(actual))}"
        )
    unequal = [
        name
        for name in sorted(actual)
        if not torch.equal(actual[name].detach().cpu(), expected[name])
    ]
    if unequal:
        raise RuntimeError(f"{description} is not bitwise equal: {unequal[:8]}")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Seeds must be integers")
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("Seeds must be in [0, 2**31 - 1]")


def _path_has_forbidden_marker(value: str | os.PathLike[str]) -> bool:
    path = Path(value)
    candidates = (str(value), str(path.absolute()), str(path.resolve(strict=False)))
    return any(FORBIDDEN_TEST_MARKER in item.casefold() for item in candidates)


def _reject_forbidden_paths(*values: str | os.PathLike[str]) -> None:
    bad = [str(value) for value in values if _path_has_forbidden_marker(value)]
    if bad:
        raise ValueError(f"Official-test references are forbidden: {bad}")


def _find_forbidden_string(value: Any, location: str = "checkpoint") -> str | None:
    """Return the first nested checkpoint string that names the official test."""

    if isinstance(value, (str, os.PathLike)):
        return location if FORBIDDEN_TEST_MARKER in str(value).casefold() else None
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _find_forbidden_string(key, f"{location}.<key>")
            if found is not None:
                return found
            found = _find_forbidden_string(item, f"{location}.{key}")
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            found = _find_forbidden_string(item, f"{location}[{index}]")
            if found is not None:
                return found
    return None


def construct_native_spatial_model(
    core: Literal["quantum", "classical"], construction_seed: int
) -> D4OrbitClassifier:
    """Construct the exact spatial-stat model without perturbing caller RNG."""

    if core not in ("quantum", "classical"):
        raise ValueError(f"Unsupported paired core: {core}")
    _validate_seed(construction_seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(construction_seed)
        model = D4OrbitClassifier(
            num_classes=3,
            heads=4,
            reuploads=3,
            core=core,
            include_context=False,
            encoder_variant="micro-stat",
            physics_variant="base",
            physics_summary="moments",
            quantum_encoding="angle",
            observable_readout="pair",
            dropout=0.10,
        ).cpu()
    return model


def _validate_parameter_report(
    core: Literal["quantum", "classical"], report: Mapping[str, Any]
) -> None:
    expected = {
        "total": EXPECTED_PARAMETER_TOTAL,
        "encoder": 119_682,
        "orbit_projection": 1_672,
        "head_and_context": 1_087,
        "core": EXPECTED_CORE_PARAMETERS,
        "core_architecture": core,
        "quantum": EXPECTED_CORE_PARAMETERS if core == "quantum" else 0,
    }
    drift = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if core == "classical" and report.get("parallel_classical") != 132:
        drift["parallel_classical"] = (report.get("parallel_classical"), 132)
    if drift:
        raise RuntimeError(f"Spatial-stat parameter contract drifted: {drift}")


def _load_backbone_checkpoint(
    checkpoint_path: Path, expected_sha256: str
) -> tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError(f"Backbone checkpoint is missing or unsafe: {checkpoint_path}")
    actual_sha256 = file_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Backbone checkpoint SHA256 drifted: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    forbidden = _find_forbidden_string(checkpoint)
    if forbidden is not None:
        raise ValueError(
            f"Backbone checkpoint contains an official-test reference at {forbidden}"
        )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Backbone checkpoint must contain a mapping")
    source_state = checkpoint.get("model", checkpoint)
    if not isinstance(source_state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in source_state.items()
    ):
        raise TypeError("Backbone model state must map string names to tensors")
    metadata = {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "bytes": checkpoint_path.stat().st_size,
        "source_epoch": checkpoint.get("epoch"),
    }
    return source_state, metadata


def _apply_train_backbone_initialization(
    model: D4OrbitClassifier, source_state: Mapping[str, torch.Tensor]
) -> Dict[str, Any]:
    """Mirror the ``train --init-backbone-checkpoint`` branch exactly."""

    allowed_prefixes = ("physics.", "encoder.", "orbit_projection.")
    backbone_state = {
        key: value.detach().cpu().clone()
        for key, value in source_state.items()
        if key.startswith(allowed_prefixes)
    }
    adapted_tensors: list[Dict[str, Any]] = []
    target_state = model.state_dict()
    for expandable_key in ("encoder.stem.0.weight", "orbit_projection.weight"):
        zero_extend_input_weight(
            backbone_state,
            target_state,
            expandable_key,
            adapted_tensors,
            insert_before_tail=(
                model.physics_summary_dim
                if expandable_key == "orbit_projection.weight"
                else 0
            ),
        )
    if not any(key.startswith("encoder.") for key in backbone_state):
        raise RuntimeError("Backbone checkpoint contains no encoder state")
    incompatible = model.load_state_dict(backbone_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected initialized keys: {list(incompatible.unexpected_keys)}"
        )
    return {
        "loaded_tensors": len(backbone_state),
        "loaded_prefixes": list(allowed_prefixes),
        "missing_target_tensors": len(incompatible.missing_keys),
        "adapted_tensors": adapted_tensors,
        "loaded_state_sha256": state_sha256(backbone_state),
    }


def _reset_quantum_head(model: D4OrbitClassifier, common_seed: int) -> list[str]:
    _validate_seed(common_seed)
    reset_modules: list[str] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(common_seed)
        for name, module in model.head.named_modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
                reset_modules.append(f"head{'.' + name if name else ''}:{type(module).__name__}")
    if not reset_modules:
        raise RuntimeError("Spatial-stat head exposed no resettable modules")
    return reset_modules


def _clone_all_noncore_state(
    source: D4OrbitClassifier, target: D4OrbitClassifier
) -> Dict[str, Any]:
    source_noncore = _component_state(source.state_dict(), core=False)
    target_noncore = _component_state(target.state_dict(), core=False)
    if set(source_noncore) != set(target_noncore):
        raise RuntimeError("Quantum/classical non-core state schemas differ")
    for name in source_noncore:
        if source_noncore[name].shape != target_noncore[name].shape:
            raise RuntimeError(f"Non-core tensor shape differs for {name}")
    incompatible = target.load_state_dict(source_noncore, strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("core.") for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "Non-core clone had an invalid state result: "
            f"missing={list(incompatible.missing_keys)} "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    final_target = _component_state(target.state_dict(), core=False)
    _assert_equal_states(final_target, source_noncore, "paired non-core state")
    return {
        "tensor_count": len(source_noncore),
        "value_count": sum(value.numel() for value in source_noncore.values()),
        "sha256": state_sha256(source_noncore),
        "bitwise_equal": True,
        "source_arm": "quantum",
        "target_arm": "classical",
    }


def _checkpoint_payload(
    model: D4OrbitClassifier,
    core: Literal["quantum", "classical"],
    seed: int,
    common_head_seed: int,
    backbone_sha256: str,
    common_noncore_sha256: str,
    native_core_sha256: str,
) -> Dict[str, Any]:
    state = _clone_state(model.state_dict())
    parameters = model.parameter_report()
    _validate_parameter_report(core, parameters)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "epoch": 0,
        "seed": seed,
        "core_name": core,
        "architecture": dict(ARCHITECTURE),
        "common_head_seed": common_head_seed,
        "backbone_fingerprint": {"sha256": backbone_sha256},
        "common_noncore_state_sha256": common_noncore_sha256,
        "noncore_state_sha256": common_noncore_sha256,
        "native_core_state_sha256": native_core_sha256,
        "core_state_sha256": native_core_sha256,
        "full_state_sha256": state_sha256(state),
        "parameters": parameters,
        "model": state,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    torch.save(dict(payload), path)
    return {
        "path": str(path.name),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "full_state_sha256": payload["full_state_sha256"],
    }


def _build_seed_pair(
    source_state: Mapping[str, torch.Tensor],
    backbone: Mapping[str, Any],
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    _validate_seed(seed)
    common_head_seed = COMMON_HEAD_SEED_OFFSET + seed
    _validate_seed(common_head_seed)

    quantum = construct_native_spatial_model("quantum", seed)
    classical = construct_native_spatial_model("classical", seed)
    quantum_native_core = _module_core_state(quantum)
    classical_native_core = _module_core_state(classical)

    quantum_backbone = _apply_train_backbone_initialization(quantum, source_state)
    classical_backbone = _apply_train_backbone_initialization(classical, source_state)
    if quantum_backbone != classical_backbone:
        raise RuntimeError("Paired arms produced different backbone adaptation reports")

    reset_modules = _reset_quantum_head(quantum, common_head_seed)
    common_noncore = _clone_all_noncore_state(quantum, classical)

    quantum_final_core = _module_core_state(quantum)
    classical_final_core = _module_core_state(classical)
    _assert_equal_states(
        quantum_final_core, quantum_native_core, "quantum native core preservation"
    )
    _assert_equal_states(
        classical_final_core,
        classical_native_core,
        "classical native core preservation",
    )
    if sum(parameter.numel() for parameter in quantum.core.parameters()) != 132:
        raise RuntimeError("Quantum native core is not the 132-parameter core")
    if sum(parameter.numel() for parameter in classical.core.parameters()) != 132:
        raise RuntimeError("Classical native core is not the 132-parameter core")

    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir()
    quantum_payload = _checkpoint_payload(
        quantum,
        "quantum",
        seed,
        common_head_seed,
        str(backbone["sha256"]),
        str(common_noncore["sha256"]),
        state_sha256(quantum_native_core),
    )
    classical_payload = _checkpoint_payload(
        classical,
        "classical",
        seed,
        common_head_seed,
        str(backbone["sha256"]),
        str(common_noncore["sha256"]),
        state_sha256(classical_native_core),
    )
    quantum_artifact = _save_checkpoint(seed_dir / "quantum-init.pt", quantum_payload)
    classical_artifact = _save_checkpoint(
        seed_dir / "classical-init.pt", classical_payload
    )
    quantum_artifact["path"] = f"seed-{seed}/{quantum_artifact['path']}"
    classical_artifact["path"] = f"seed-{seed}/{classical_artifact['path']}"

    arm_reports = {
        "quantum": {
            "core_name": "quantum",
            "checkpoint": quantum_artifact["path"],
            "checkpoint_path": quantum_artifact["path"],
            "checkpoint_sha256": quantum_artifact["sha256"],
            "checkpoint_bytes": quantum_artifact["bytes"],
            "full_state_sha256": quantum_artifact["full_state_sha256"],
            "core_state_sha256": state_sha256(quantum_native_core),
            "native_core_state_sha256": state_sha256(quantum_native_core),
            "noncore_state_sha256": common_noncore["sha256"],
            "parameters": quantum_payload["parameters"],
        },
        "classical": {
            "core_name": "classical",
            "checkpoint": classical_artifact["path"],
            "checkpoint_path": classical_artifact["path"],
            "checkpoint_sha256": classical_artifact["sha256"],
            "checkpoint_bytes": classical_artifact["bytes"],
            "full_state_sha256": classical_artifact["full_state_sha256"],
            "core_state_sha256": state_sha256(classical_native_core),
            "native_core_state_sha256": state_sha256(classical_native_core),
            "noncore_state_sha256": common_noncore["sha256"],
            "parameters": classical_payload["parameters"],
        },
    }
    binding_payload = {
        "protocol_id": PROTOCOL,
        "seed": seed,
        "backbone_sha256": backbone["sha256"],
        "common_noncore_state_sha256": common_noncore["sha256"],
        "quantum": {
            key: arm_reports["quantum"][key]
            for key in (
                "core_name",
                "checkpoint_sha256",
                "full_state_sha256",
                "core_state_sha256",
                "noncore_state_sha256",
            )
        },
        "classical": {
            key: arm_reports["classical"][key]
            for key in (
                "core_name",
                "checkpoint_sha256",
                "full_state_sha256",
                "core_state_sha256",
                "noncore_state_sha256",
            )
        },
    }
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "seed": seed,
        "architecture": dict(ARCHITECTURE),
        "construction_seed": seed,
        "common_head_seed": common_head_seed,
        "common_head_seed_rule": "10000 + construction_seed",
        "head_reset_modules": reset_modules,
        "backbone_fingerprint": {
            **dict(backbone),
            "loaded_state_sha256": quantum_backbone["loaded_state_sha256"],
        },
        "common_noncore_state_sha256": common_noncore["sha256"],
        "pair_binding_sha256": hashlib.sha256(_json_bytes(binding_payload)).hexdigest(),
        "arms": arm_reports,
        "common_noncore": common_noncore,
        "native_core": {
            "quantum": {
                "sha256": state_sha256(quantum_native_core),
                "tensor_count": len(quantum_native_core),
                "preserved_bitwise": True,
            },
            "classical": {
                "sha256": state_sha256(classical_native_core),
                "tensor_count": len(classical_native_core),
                "preserved_bitwise": True,
            },
        },
        "backbone_initialization": quantum_backbone,
    }


def build_paired_initializers(
    backbone_checkpoint: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    seeds: Sequence[int],
    *,
    expected_backbone_sha256: str = EXPECTED_BACKBONE_SHA256,
) -> Dict[str, Any]:
    """Build all seed pairs atomically and return the persisted report."""

    backbone_path = Path(backbone_checkpoint)
    destination = Path(output_dir)
    _reject_forbidden_paths(backbone_path, destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite paired initializer output: {destination}")
    requested_seeds = list(seeds)
    if not requested_seeds:
        raise ValueError("At least one seed is required")
    for seed in requested_seeds:
        _validate_seed(seed)
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("Seeds must be unique")
    if len(expected_backbone_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_backbone_sha256
    ):
        raise ValueError("Expected backbone SHA256 must be lowercase hexadecimal")

    source_state, backbone = _load_backbone_checkpoint(
        backbone_path, expected_backbone_sha256
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Unexpected initializer staging path exists: {staging}")
    staging.mkdir()
    try:
        seed_reports = {
            str(seed): _build_seed_pair(source_state, backbone, seed, staging)
            for seed in requested_seeds
        }
        report: Dict[str, Any] = {
            "schema_version": 1,
            "protocol_id": PROTOCOL,
            "architecture": dict(ARCHITECTURE),
            "backbone_fingerprint": dict(backbone),
            "seeds": requested_seeds,
            "per_seed": seed_reports,
            "official_test_opened": False,
            "official_test_reference_accepted": False,
        }
        # Normalize tuples from the shared train-time adaptation helper to the
        # exact JSON representation returned to callers and consumed by jobs.
        report = json.loads(json.dumps(report, allow_nan=False))
        report["report_payload_sha256"] = hashlib.sha256(_json_bytes(report)).hexdigest()
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(staging, destination)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SHA-audited Model-I spatial-stat paired initializers"
    )
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    _reject_forbidden_paths(args.backbone_checkpoint, args.output_dir)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_paired_initializers(
        args.backbone_checkpoint,
        args.output_dir,
        args.seeds,
    )
    report_path = Path(args.output_dir) / "report.json"
    print(
        "SPATIAL_PAIRED_INITIALIZERS "
        + json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "report_sha256": file_sha256(report_path),
                "report_payload_sha256": report["report_payload_sha256"],
                "seeds": report["seeds"],
                "official_test_opened": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
