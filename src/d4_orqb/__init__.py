"""Clean D4 orbit-reuploading quantum bottleneck package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .model import D4OrbitClassifier
    from .quantum import D4OrbitQuantumBottleneck

__all__ = ["Config", "D4OrbitClassifier", "D4OrbitQuantumBottleneck"]


def __getattr__(name: str):
    if name == "Config":
        from .config import Config

        return Config
    if name == "D4OrbitClassifier":
        from .model import D4OrbitClassifier

        return D4OrbitClassifier
    if name == "D4OrbitQuantumBottleneck":
        from .quantum import D4OrbitQuantumBottleneck

        return D4OrbitQuantumBottleneck
    raise AttributeError(name)
