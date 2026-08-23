"""What every ``run`` returns.

One container for both shapes of result, because the *model* decides the shape:

===================  ==========================  ==============================
field                single-system models         multi-site models
===================  ==========================  ==============================
``expect[name]``     ``(n_steps,)``              ``(n_steps, n_sites)`` for a
                                                 per-site spec, ``(n_steps,)``
                                                 for a single- or multi-site one
``rdm``              ``(n_steps, d, d)``         ``(n_steps, n_sites, d, d)``,
                                                 or an object array when site
                                                 dimensions differ
``max_bond``         peak bond per step          same
``method``           the method that ran         same
``meta``             ``{}``                      ``{"n_sites": n}``
``checkpoint``       ``None``                    resumable static-tree state
===================  ==========================  ==============================

``system-bath`` and ``multichannel`` are single-system; ``comb`` and ``site-tree``
are multi-site.  See :mod:`fishbonett.models.registry`.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["Result", "SimulationCheckpoint"]


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


@dataclass(frozen=True)
class SimulationCheckpoint:
    """A resumable static-tree tensor state and its compatibility metadata.

    Checkpoints are valid only for the same fully resolved Hamiltonian.  They can
    be kept in memory or stored as a pickle-free NPZ archive with :meth:`save`.
    """

    tensors: tuple
    dims: tuple
    edges: tuple
    oc: int
    method: str
    elapsed: float
    bath_horizon: float
    signature: str

    @classmethod
    def from_state(cls, state, terms, *, method, elapsed, bath_horizon):
        return cls(
            tensors=tuple(np.array(value, complex, copy=True) for value in state.T),
            dims=tuple(map(int, terms.dims)),
            edges=tuple(tuple(map(int, edge)) for edge in terms.edges),
            oc=int(state.oc), method=str(method), elapsed=float(elapsed),
            bath_horizon=float(bath_horizon),
            signature=_hamiltonian_signature(terms),
        )

    def restore(self, terms):
        """Return a fresh tree state after validating ``terms``."""
        if self.signature != _hamiltonian_signature(terms):
            raise ValueError(
                "checkpoint Hamiltonian does not match this resolved model; "
                "bath temperatures, couplings, discretization and topology must "
                "remain unchanged")
        from fishbonett.states.tree import TreeTensorNetwork
        state = TreeTensorNetwork(terms.dims, terms.edges, root=0)
        if len(self.tensors) != state.n:
            raise ValueError("checkpoint tensor count does not match model topology")
        state.T = [np.array(value, complex, copy=True) for value in self.tensors]
        state.oc = int(self.oc)
        return state

    def save(self, path):
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
        np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)
        return path

    @classmethod
    def load(cls, path):
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
    """Result of a propagation."""
    t: np.ndarray
    expect: dict                      # observable name -> array over time
    max_bond: np.ndarray = None       # peak bond dimension per step (adaptive)
    rdm: np.ndarray = None            # spin reduced density matrix per step (T,2,2)
    method: str = ""
    meta: dict = field(default_factory=dict)
    checkpoint: SimulationCheckpoint = None
