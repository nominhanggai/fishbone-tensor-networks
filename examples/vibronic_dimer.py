"""Vibrationally assisted transfer in a biased molecular dimer.

The model is the Brownian-oscillator dimer used by Dijkstra et al.
(arXiv:1309.4910). Run the inexpensive engine check with
``python examples/vibronic_dimer.py``. The ``docs`` profile reconstructs the
two quantum benchmark trajectories at omega/J = 4 and 8; the manual
``reference`` profile scans the vibrational frequency more finely.
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
    # Figure 5 of Dijkstra et al. reports these trajectories through tJ=20.
    # Automatic mode resolution is essential here: short fixed chains reflect
    # into the system before the benchmark endpoint.
    "docs": Profile("docs", 20.0, 0.025, (4.0, 8.0), 12, None, 1e-3),
    "reference": Profile(
        "reference", 20.0, 0.0125,
        tuple(np.arange(1.0, 10.0 + 0.5, 1.0)), 16, None, 5e-4,
    ),
}

PAPER_FIGURE_5 = (
    Path(__file__).with_name("reference_data")
    / "dijkstra_2015_fig5_quantum_dynamics.csv"
)


def load_paper_figure5(path=PAPER_FIGURE_5):
    """Load samples derived from the quantum-dynamics curves in Figure 5."""
    return np.genfromtxt(path, delimiter=",", names=True)


def make_bath(vibration, profile):
    """Return the Brownian bath for the fluctuating donor-acceptor gap."""
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
    """Biased dimer coupled to one Brownian gap-fluctuation coordinate.

    The paper describes coupling the molecular energy difference to a Brownian
    coordinate but does not print an operator matrix.  Here ``OCCUPIED`` on the
    donor implements E_D -> E_D + X, and hence (E_D - E_A) -> (E_D - E_A) + X.
    """
    electronic = np.array([[8.0, -1.0], [-1.0, 0.0]])
    baths = {0: make_bath(vibration, profile).bind(OCCUPIED)}
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
    """Numerical diagnostics and comparison with the Figure 5 trajectories."""
    results = suite["results"]
    final_population = {}
    paper_curve_rmse = {}
    paper_curve_max_error = {}
    paper = load_paper_figure5()
    normalization_error = 0.0
    max_bond = 1
    resolved_modes = {}
    for vibration, result in results.items():
        population = np.asarray(result.expect["population"], float)
        final_population[vibration] = float(population[-1, 1])
        column = f"omega{int(vibration)}_acceptor"
        if column in (paper.dtype.names or ()):
            calculated = np.interp(paper["tJ"], result.t, population[:, 1])
            difference = calculated - paper[column]
            paper_curve_rmse[vibration] = float(np.sqrt(np.mean(difference**2)))
            paper_curve_max_error[vibration] = float(np.max(np.abs(difference)))
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
        "paper_curve_rmse": paper_curve_rmse,
        "paper_curve_max_error": paper_curve_max_error,
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
