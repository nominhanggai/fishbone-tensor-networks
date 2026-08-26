"""What every ``run`` returns.

One container for both shapes of result, because the *model* decides the shape:

.. list-table::
   :header-rows: 1

   * - field
     - single-system models
     - multi-site models
   * - ``expect[name]``
     - ``(n_records,)``; ``ExcitonBath`` also records all site populations as
       ``(n_records, n_levels)``
     - ``(n_records, n_sites)`` for a per-site specification; otherwise
       ``(n_records,)``
   * - ``rdm``
     - ``(n_records, d, d)``
     - ``(n_records, n_sites, d, d)``, or an object array for unequal dimensions
   * - ``max_bond``
     - peak bond per record
     - peak bond per record
   * - ``method`` and ``meta``
     - selected method and settings
     - selected method, settings, and site information
   * - ``checkpoint``
     - resumable state when the method supports it
     - resumable state when the method supports it

``system-bath``, ``multichannel``, and ``exciton-bath`` are single-system;
``comb`` and ``site-tree`` are multi-site. ``n_records`` equals ``n_steps`` unless ``observe_every`` is
greater than one. See :mod:`fishbonett.models.registry`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from fishbonett.representations.schrodinger import LocalTerms
    from fishbonett.states.tree import TreeTensorNetwork

__all__ = ["Result", "SimulationCheckpoint", "plan_signature"]


def _update_array(digest, value):
    array = np.ascontiguousarray(np.asarray(value, complex))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))


def _hamiltonian_signature(terms):
    """Stable digest of resolved local terms, including their ordered topology."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(map(int, terms.dims))).encode("ascii"))
    digest.update(json.dumps([list(map(int, edge)) for edge in terms.edges]).encode("ascii"))
    for value in terms.site:
        _update_array(digest, value)
    for edge in terms.edges:
        _update_array(digest, terms.bond[tuple(edge)])
    for edge, value in sorted(getattr(terms, "graph_bond", {}).items()):
        digest.update(json.dumps(list(map(int, edge))).encode("ascii"))
        _update_array(digest, value)
    return digest.hexdigest()


def plan_signature(
    dims: Sequence[int],
    edges: Sequence[tuple[int, int]],
    arrays: Sequence[ArrayLike] = (),
    scalars: Sequence[object] = (),
) -> str:
    """Digest for a plan that builds its own topology rather than :class:`LocalTerms`.

    The interaction-picture comb assembles ``dims``/``edges`` itself from the
    electronic graph plus one bath branch per site, so it has no ``LocalTerms`` to
    hash.  It supplies the same information directly: the topology, the arrays that
    define the Hamiltonian (site terms, graph couplings, coupling operators) and
    the scalars that fix how each bath was resolved.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(list(map(int, dims))).encode("ascii"))
    digest.update(json.dumps([list(map(int, edge)) for edge in edges]).encode("ascii"))
    for value in arrays:
        _update_array(digest, value)
    digest.update(json.dumps([str(value) for value in scalars]).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SimulationCheckpoint:
    """A resumable tensor-network state and its validation metadata.

    Checkpoints are valid only for the same fully resolved Hamiltonian.  They can
    be kept in memory or stored as a pickle-free NPZ archive with :meth:`save`.

    Tree plans and conventional exciton MPS plans produce one. Time-dependent
    representations store ``elapsed`` because resetting their clock would
    change the continued Hamiltonian.
    """

    tensors: tuple[np.ndarray, ...]
    dims: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    oc: int
    method: str
    elapsed: float
    bath_horizon: float
    signature: str

    @classmethod
    def from_state(
        cls,
        state: TreeTensorNetwork,
        terms: LocalTerms,
        *,
        method: str,
        elapsed: float,
        bath_horizon: float,
    ) -> SimulationCheckpoint:
        """Capture a tree state using a represented ``LocalTerms`` signature."""
        return cls(
            tensors=tuple(np.array(value, complex, copy=True) for value in state.T),
            dims=tuple(map(int, terms.dims)),
            edges=tuple(tuple(map(int, edge)) for edge in terms.edges),
            oc=int(state.oc), method=str(method), elapsed=float(elapsed),
            bath_horizon=float(bath_horizon),
            signature=_hamiltonian_signature(terms),
        )

    @classmethod
    def from_tree(
        cls,
        state: TreeTensorNetwork,
        dims: Sequence[int],
        edges: Sequence[tuple[int, int]],
        *,
        signature: str,
        method: str,
        elapsed: float,
        bath_horizon: float,
    ) -> SimulationCheckpoint:
        """Checkpoint a tree state whose topology was assembled by the plan."""
        return cls(
            tensors=tuple(np.array(value, complex, copy=True) for value in state.T),
            dims=tuple(map(int, dims)),
            edges=tuple(tuple(map(int, edge)) for edge in edges),
            oc=int(state.oc), method=str(method), elapsed=float(elapsed),
            bath_horizon=float(bath_horizon), signature=str(signature))

    def restore(self, terms: LocalTerms) -> TreeTensorNetwork:
        """Return a fresh tree state after validating ``terms``."""
        return self.restore_tree(terms.dims, terms.edges,
                                 _hamiltonian_signature(terms))

    def restore_tree(
        self,
        dims: Sequence[int],
        edges: Sequence[tuple[int, int]],
        signature: str,
    ) -> TreeTensorNetwork:
        """Return a fresh tree state after validating an explicit topology."""
        if self.signature != str(signature):
            raise ValueError(
                "checkpoint Hamiltonian does not match this resolved model; "
                "bath temperatures, couplings, discretization and topology must "
                "remain unchanged")
        from fishbonett.states.tree import TreeTensorNetwork
        state = TreeTensorNetwork(dims, edges, root=0)
        if tuple(map(int, dims)) != self.dims:
            raise ValueError("checkpoint physical dimensions do not match the model")
        normalized_edges = tuple(tuple(map(int, edge)) for edge in edges)
        if normalized_edges != self.edges:
            raise ValueError("checkpoint topology does not match the model")
        if len(self.tensors) != state.n:
            raise ValueError("checkpoint tensor count does not match model topology")
        if self.oc < 0 or self.oc >= state.n:
            raise ValueError("checkpoint orthogonality centre is outside the state")
        tensors = [np.array(value, complex, copy=True) for value in self.tensors]
        for node, tensor in enumerate(tensors):
            expected_rank = len(state.neighbours(node)) + 1
            if tensor.ndim != expected_rank:
                raise ValueError(
                    f"checkpoint tensor {node} has rank {tensor.ndim}, "
                    f"expected {expected_rank}"
                )
            if tensor.shape[-1] != state.dims[node]:
                raise ValueError(
                    f"checkpoint tensor {node} has physical dimension "
                    f"{tensor.shape[-1]}, expected {state.dims[node]}"
                )
            if not np.all(np.isfinite(tensor)):
                raise ValueError(f"checkpoint tensor {node} contains non-finite values")
        for left, right in normalized_edges:
            left_leg = state.neighbours(left).index(right)
            right_leg = state.neighbours(right).index(left)
            if tensors[left].shape[left_leg] != tensors[right].shape[right_leg]:
                raise ValueError(
                    f"checkpoint bond {(left, right)} has incompatible dimensions"
                )
        state.T = tensors
        state.oc = int(self.oc)
        return state

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write this checkpoint as an NPZ archive without Python pickles."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        metadata = {
            "version": 1, "dims": self.dims, "edges": self.edges,
            "oc": self.oc, "method": self.method, "elapsed": self.elapsed,
            "bath_horizon": self.bath_horizon, "signature": self.signature,
            "n_tensors": len(self.tensors),
        }
        arrays = {f"tensor_{i}": value for i, value in enumerate(self.tensors)}
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".npz",
            delete=False,
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(
                temporary, metadata=np.array(json.dumps(metadata)), **arrays
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SimulationCheckpoint:
        """Load a checkpoint written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("version") != 1:
                raise ValueError("unsupported checkpoint format version")
            tensors = tuple(np.array(archive[f"tensor_{i}"], copy=True)
                            for i in range(int(metadata["n_tensors"])))
        return cls(
            tensors=tensors, dims=tuple(metadata["dims"]),
            edges=tuple(tuple(edge) for edge in metadata["edges"]),
            oc=int(metadata["oc"]), method=metadata["method"],
            elapsed=float(metadata["elapsed"]),
            bath_horizon=float(metadata["bath_horizon"]),
            signature=metadata["signature"],
        )


@dataclass
class Result:
    """Uniform result returned by every high-level ``run`` method.

    The leading dimension of ``t``, each observable, ``max_bond`` and ``rdm`` is
    the number of recorded samples. It equals ``n_steps`` by default and is
    smaller when ``observe_every > 1``. See the module table for the remaining
    single-system and multi-site dimensions.
    """
    t: np.ndarray
    expect: dict[str, np.ndarray]
    max_bond: np.ndarray | None = None
    rdm: np.ndarray | None = None
    method: str = ""
    meta: dict[str, object] = field(default_factory=dict)
    checkpoint: SimulationCheckpoint | None = None
