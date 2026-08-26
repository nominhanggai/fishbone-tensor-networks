"""Compare tensor-network layouts for seven-site FMO exciton dynamics.

The electronic Hamiltonian is the Adolphs--Renger seven-site parameter set.
Every pigment has an independent Drude--Lorentz bath coupled to its site
population.  The smoke profile is an API and finite-layout comparison; increase
the bath resolution and time horizon only with independent convergence checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fishbonett import Bath, ExcitonBath


CM_SCALE = 100.0
REORGANIZATION = 35.0 / CM_SCALE
CUTOFF = 106.0 / CM_SCALE
BETA = CM_SCALE / 208.509
DOMAIN = (-8.0, 8.0)
DOMAIN_NORMALIZATION = np.pi / (2.0 * np.arctan(DOMAIN[1] / CUTOFF))

FMO_HAMILTONIAN_CM = np.array(
    [
        [240.0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
        [-87.7, 315.0, 30.8, 8.2, 0.7, 11.8, 4.3],
        [5.5, 30.8, 0.0, -53.5, -2.2, -9.6, 6.0],
        [-5.9, 8.2, -53.5, 130.0, -70.7, -17.0, -63.3],
        [6.7, 0.7, -2.2, -70.7, 285.0, 81.1, -1.3],
        [-13.7, 11.8, -9.6, -17.0, 81.1, 435.0, 39.7],
        [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 245.0],
    ]
)
FMO_HAMILTONIAN = FMO_HAMILTONIAN_CM / CM_SCALE


@dataclass(frozen=True)
class Profile:
    t_max: float
    dt: float
    n_modes: int
    phys_dim: int
    trunc_eps: float


PROFILES = {
    "smoke": Profile(0.02, 0.02, 1, 3, 1e-8),
    "quick": Profile(0.20, 0.02, 6, 4, 1e-4),
}

METHODS = {
    "system-first": "interaction-chain-system-first-tdvp2",
    "interleaved": "interaction-chain-interleaved-tdvp2",
    "multi-set": "interaction-chain-multi-set-tdvp2",
    "multi-set-tree": "interaction-chain-multi-set-tree-tdvp2",
}


def spectral_density(frequency):
    if frequency < 0:
        return 0.0
    return (
        DOMAIN_NORMALIZATION
        * 2.0
        * REORGANIZATION
        * CUTOFF
        * frequency
        / (CUTOFF**2 + frequency**2)
    )


def make_model(profile):
    baths = [
        Bath(
            J=spectral_density,
            beta=BETA,
            domain=DOMAIN,
            n_modes=profile.n_modes,
            phys_dim=profile.phys_dim,
            discretization="tedopa",
            extra_breaks=(0.0,),
        )
        for _ in range(7)
    ]
    return ExcitonBath(FMO_HAMILTONIAN, baths)


def run(profile, layouts):
    model = make_model(profile)
    steps = int(round(profile.t_max / profile.dt))
    results = {}
    summary = {}
    for layout in layouts:
        started = perf_counter()
        result = model.run(
            dt=profile.dt,
            n_steps=steps,
            bath_horizon=profile.t_max,
            method=METHODS[layout],
            initial=0,
            trunc_eps=profile.trunc_eps,
            bond_dim=None,
        )
        seconds = perf_counter() - started
        results[layout] = result
        summary[layout] = {
            "method": result.method,
            "seconds": seconds,
            "peak_bond": int(np.max(result.max_bond)),
            "final_population": [float(value) for value in result.expect["population"][-1]],
        }
    reference = results[layouts[0]].expect["population"]
    for layout in layouts[1:]:
        summary[layout]["maximum_population_difference"] = float(
            np.max(np.abs(results[layout].expect["population"] - reference))
        )
    arrays = {"t": results[layouts[0]].t}
    for layout, result in results.items():
        key = layout.replace("-", "_")
        arrays[f"{key}_population"] = result.expect["population"]
        arrays[f"{key}_max_bond"] = result.max_bond
    return arrays, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=METHODS,
        default=["system-first", "interleaved", "multi-set"],
        help="include multi-set-tree explicitly; it has the largest per-step cost",
    )
    parser.add_argument("--output", type=Path, default=Path("fmo_state_layouts.npz"))
    args = parser.parse_args()
    arrays, summary = run(PROFILES[args.profile], args.layouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
