"""Compare propagators on conventional MPSs for seven-site FMO dynamics.

The system-first and interleaved layouts each run with swap-network TEBD,
conditional-displacement Trotter-MPO, TDVP1, TDVP2, or dTDVP.  The ``smoke``
profile checks the complete workflow.  The ``200fs`` profile uses automatic
TEDOPA bath resolution, a six-level oscillator basis, and an SVD threshold of
``1e-4``; it is a production calculation and writes a checkpoint after every
segment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fishbonett import Bath, ExcitonBath, SimulationCheckpoint

from fmo_state_layouts import (
    BETA,
    CUTOFF,
    FMO_HAMILTONIAN,
    REORGANIZATION,
    TIME_UNIT_FS,
)


@dataclass(frozen=True)
class Profile:
    dt: float
    n_steps: int
    n_modes: int | None
    phys_dim: int
    trunc_eps: float
    domain: tuple[float, float]
    segment_steps: int
    bond_dim: int

    @property
    def horizon(self):
        return self.dt * self.n_steps


PROFILES = {
    "smoke": Profile(0.01, 2, 1, 3, 1e-8, (-8.0, 8.0), 2, 64),
    "200fs": Profile(0.025, 151, None, 6, 1e-4, (-10.0, 10.0), 5, 512),
}

METHODS = {
    f"{layout}-{integrator}": f"interaction-chain-{layout}-{integrator}"
    for layout in ("system-first", "interleaved")
    for integrator in ("tebd", "trotter-mpo", "tdvp1", "tdvp2", "dtdvp")
}


def make_model(profile):
    normalization = np.pi / (
        2.0 * np.arctan(profile.domain[1] / CUTOFF)
    )

    def spectral_density(frequency):
        if frequency < 0.0:
            return 0.0
        return (
            normalization
            * 2.0
            * REORGANIZATION
            * CUTOFF
            * frequency
            / (CUTOFF**2 + frequency**2)
        )

    bath = Bath(
        J=spectral_density,
        beta=BETA,
        domain=profile.domain,
        n_modes=profile.n_modes,
        phys_dim=profile.phys_dim,
        discretization="tedopa",
        extra_breaks=(0.0,),
    ).resolved(profile.horizon)
    return ExcitonBath(FMO_HAMILTONIAN, [bath] * 7)


def _paths(output, label):
    directory = output / label
    return (
        directory / "checkpoint.npz",
        directory / "partial.npz",
        directory / "summary.json",
    )


def _load_progress(checkpoint_path, partial_path):
    if not checkpoint_path.exists() or not partial_path.exists():
        return None, [], [], []
    checkpoint = SimulationCheckpoint.load(checkpoint_path)
    with np.load(partial_path) as archive:
        times = list(np.asarray(archive["t"], float))
        populations = list(np.asarray(archive["population"], float))
        bonds = list(np.asarray(archive["max_bond"], int))
    return checkpoint, times, populations, bonds


def _save_progress(
    checkpoint_path, partial_path, checkpoint, times, populations, bonds,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.save(checkpoint_path)
    np.savez_compressed(
        partial_path,
        t=np.asarray(times),
        t_fs=np.asarray(times) * TIME_UNIT_FS,
        population=np.asarray(populations),
        max_bond=np.asarray(bonds, dtype=int),
    )


def run_method(profile, label, output):
    method = METHODS[label]
    checkpoint_path, partial_path, summary_path = _paths(output, label)
    checkpoint, times, populations, bonds = _load_progress(
        checkpoint_path, partial_path
    )
    completed = 0 if checkpoint is None else int(round(checkpoint.elapsed / profile.dt))
    started = perf_counter()
    model = make_model(profile)
    print(
        f"[{label}] {completed}/{profile.n_steps} steps already checkpointed",
        flush=True,
    )
    while completed < profile.n_steps:
        count = min(profile.segment_steps, profile.n_steps - completed)
        options = dict(
            dt=profile.dt,
            n_steps=count,
            bath_horizon=profile.horizon,
            method=method,
            trunc_eps=profile.trunc_eps,
            bond_dim=profile.bond_dim,
            svd_backend="auto",
        )
        if checkpoint is None:
            options["initial"] = 0
        else:
            options["resume"] = checkpoint
        result = model.run(**options)
        checkpoint = result.checkpoint
        times.extend(result.t.tolist())
        populations.extend(result.expect["population"].tolist())
        bonds.extend(result.max_bond.tolist())
        completed += count
        _save_progress(
            checkpoint_path,
            partial_path,
            checkpoint,
            times,
            populations,
            bonds,
        )
        print(
            f"[{label}] {completed:3d}/{profile.n_steps}: "
            f"t={times[-1] * TIME_UNIT_FS:8.3f} fs, bond={bonds[-1]}",
            flush=True,
        )

    population = np.asarray(populations)
    summary = {
        "method": method,
        "state_family": "conventional-mps",
        "state_geometry": (
            "system-first-mps"
            if label.startswith("system-first-")
            else "interleaved-mps"
        ),
        "layout": (
            "system-first" if label.startswith("system-first-") else "interleaved"
        ),
        "elapsed_seconds_this_invocation": perf_counter() - started,
        "final_time_fs": times[-1] * TIME_UNIT_FS,
        "dt_fs": profile.dt * TIME_UNIT_FS,
        "n_steps": profile.n_steps,
        "svd_threshold": profile.trunc_eps,
        "max_bond_cap": profile.bond_dim,
        "physical_dimension": profile.phys_dim,
        "peak_bond": int(max(bonds)),
        "normalization_error": float(
            np.max(np.abs(np.sum(population, axis=1) - 1.0))
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS,
        default=["system-first-tdvp2", "interleaved-tdvp2"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("examples/output/fmo_mps_methods")
    )
    args = parser.parse_args()
    summaries = {
        label: run_method(PROFILES[args.profile], label, args.output)
        for label in args.methods
    }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
