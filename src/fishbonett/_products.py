"""Internal structured tensor-network operator products."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaledTreeIdentity:
    """A scalar identity on a particular tree tensor-network topology."""

    coefficient: complex
    dimensions: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    root: int
