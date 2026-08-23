"""Vibrationally assisted transfer in a biased molecular dimer.

The model is the Brownian-oscillator dimer used by Dijkstra et al.
(arXiv:1309.4910). Run the inexpensive engine check with
``python examples/vibronic_dimer.py``. The ``docs`` and ``reference`` profiles
increase only numerical resolution and propagation time.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fishbonett import Bath, Fishbone
from fishbonett.spectral_densities import brownian


OCCUPIED = np.diag([0.0, 1.0])
UNOCCUPIED = np.array([1.0, 0.0])
EXCITED = np.array([0.0, 1.0])


@dataclass(frozen=True)
class Profile:
    name: str
    t_max: float
    dt: float
    frequencies: tuple
    phys_dim: int
    n_modes: int | None
    trunc_eps: float


PROFILES = {
    "smoke": Profile("smoke", 0.04, 0.01, (8.0,), 3, 4, 1e-3),
    "docs": Profile("docs", 10.0, 0.05, (4.0, 8.0), 8, 24, 1e-3),
    "reference": Profile(
        "reference", 20.0, 0.001,
        tuple(np.arange(2.0, 10.0 + 0.25, 0.5)), 20, None, 5e-4,
    ),
}


def make_bath(vibration, profile):
    """Return one independent Brownian bath in units where the coupling is one."""
    density = lambda omega: brownian(
        omega, lam=0.2, gam=2.0 / 3.0, w0=vibration,
    )
    return Bath(
        J=density,
        beta=0.1,
        phys_dim=profile.phys_dim,
        n_modes=profile.n_modes,
        discretization="tedopa",
    )


def make_model(vibration, profile):
    """Biased dimer with an independent occupancy bath on each molecule."""
    electronic = np.array([[8.0, -1.0], [-1.0, 0.0]])
    baths = [make_bath(vibration, profile).bind(OCCUPIED) for _ in range(2)]
    return Fishbone.from_single_excitation(electronic, baths=baths)


def run_profile(profile="smoke", *, announce=False):
    """Propagate every vibrational frequency selected by a profile."""
    config = PROFILES[profile] if isinstance(profile, str) else profile
    results = {}
    for vibration in config.frequencies:
        if announce:
            print(f"[{config.name}] vibration {vibration:g}: starting")
        model = make_model(vibration, config)
        results[float(vibration)] = model.run(
            dt=config.dt,
            t_max=config.t_max,
            method="interaction-chain-fishbone-trotter-mpo",
            trunc_eps=config.trunc_eps,
            bond_dim=None,
            initial=[EXCITED, UNOCCUPIED],
            observables={"population": OCCUPIED},
        )
    return {"profile": config, "results": results}


def summarize(suite):
    """Numerical diagnostics and the final acceptor population scan."""
    results = suite["results"]
    final_population = {}
    normalization_error = 0.0
    max_bond = 1
    resolved_modes = {}
    for vibration, result in results.items():
        population = np.asarray(result.expect["population"], float)
        final_population[vibration] = float(population[-1, 1])
        normalization_error = max(
            normalization_error,
            float(np.max(np.abs(population.sum(axis=1) - 1.0))),
        )
        max_bond = max(max_bond, int(np.max(result.max_bond)))
        resolved_modes[vibration] = tuple(
            branch["n_modes"] for branch in result.meta["bath_branches"]
        )
    return {
        "final_acceptor_population": final_population,
        "normalization_error": normalization_error,
        "max_bond": max_bond,
        "resolved_modes": resolved_modes,
    }


def save_suite(suite, path):
    """Save plain numerical arrays; figures remain documentation build artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for vibration, result in suite["results"].items():
        label = f"omega_{vibration:g}".replace(".", "p")
        payload[f"{label}_t"] = result.t
        payload[f"{label}_population"] = result.expect["population"]
        payload[f"{label}_max_bond"] = result.max_bond
    np.savez_compressed(path, **payload)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    suite = run_profile(args.profile, announce=True)
    if args.output:
        save_suite(suite, args.output)
    print(summarize(suite))
    return suite


if __name__ == "__main__":
    main()
