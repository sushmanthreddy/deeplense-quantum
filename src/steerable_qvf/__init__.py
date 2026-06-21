"""Steerable-QVF Equivariant-QCNN for DeepLense gravitational lensing.

Package layout::

    config.py    -> Config dataclass + dataset paths
    data.py      -> .npy dataset, transforms, dataloaders
    encoder.py   -> steerable C8/D8 CNN + QVF amplitude/angle head
    quantum.py   -> p4m equivariant QCNN (TorchQuantum port of EQNN_for_HEP)
    model.py     -> full encoder -> QCNN -> head model
    engine.py    -> train / evaluate loops
    main.py      -> CLI entrypoint

This is a refactor of the original ``steerable_qvf_quantum_lensing.py`` script
into an importable package. The original script is left untouched.
"""

from .config import Config

__all__ = [
    "Config",
    "SteerableAmplitudeEncoder",
    "EquivQCNN_TQ",
    "SteerableQVFQuantumModel",
    "build_model",
    "parameter_summary",
]

# Lazy attribute access so the lightweight, dependency-free pieces (e.g. ``Config``)
# stay importable even when the heavy runtime deps (torchquantum / e2cnn) are absent.
# The model/encoder/quantum symbols are resolved on first access instead of at import.
_LAZY = {
    "SteerableAmplitudeEncoder": ("encoder", "SteerableAmplitudeEncoder"),
    "EquivQCNN_TQ": ("quantum", "EquivQCNN_TQ"),
    "SteerableQVFQuantumModel": ("model", "SteerableQVFQuantumModel"),
    "build_model": ("model", "build_model"),
    "parameter_summary": ("model", "parameter_summary"),
}


def __getattr__(name):  # PEP 562 module-level lazy import
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
