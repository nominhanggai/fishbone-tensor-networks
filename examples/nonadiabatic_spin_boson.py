"""Strong-coupling nonadiabatic spin--boson dynamics.

This is the high-temperature benchmark of Nuomin, Beratan, and Zhang
(arXiv:2111.14308): Delta=1, eta=4, omega_c=4, and T=4. The ``docs`` profile
uses 200 bath modes through t*Delta/pi = 1; the ``reference`` profile uses 600
bath modes through the published endpoint t*Delta/pi = 5.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z


METHODS = {
    "interaction": "interaction-chain-trotter-mpo",
    "schrodinger_chain": "schrodinger-chain-tdvp2",
    "schrodinger_star": "schrodinger-star-tdvp2",
}

REFERENCE_DATA = Path(__file__).resolve().parent / "reference_data"
PAPER_FIG8_DATA = REFERENCE_DATA / "nuomin_2022_fig8_ic10.csv"
SIMULATION_LABEL = "fishbonett"
PAPER_LABEL = "Fig. 8 IC10 (vector-path samples)"


@dataclass(frozen=True)
class Profile:
    name: str
    t_max: float
    dt: float
    phys_dim: int
    n_modes: int | None
    domain: tuple | None
    trunc_eps: float
    methods: tuple
    comparison_t_max: float | None = None


PROFILES = {
    "smoke": Profile(
        "smoke", 0.04, 0.01, 3, 4, (-16.0, 80.0), 1e-3,
        ("interaction",),
    ),
    "docs": Profile(
        "docs", np.pi, 0.0125, 10, 200, (-16.0, 80.0), 1e-3,
        ("interaction",),
    ),
    "reference": Profile(
        # Figure 9 displays 600 bath-chain bonds for the full Figure 8 run.
        "reference", 5.0 * np.pi, 0.0125, 10, 600, (-16.0, 80.0), 1e-3,
        ("interaction",),
    ),
}


def spectral_density(omega):
    """Drude form used in the benchmark, with eta=omega_c=4."""
    eta, cutoff = 4.0, 4.0
    return eta * cutoff * omega / (cutoff**2 + omega**2)


def make_model(profile):
    bath = Bath(
        J=spectral_density,
        beta=0.25,
        domain=profile.domain,
        phys_dim=profile.phys_dim,
        n_modes=profile.n_modes,
        discretization="tedopa",
    )
    return SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)


def run_profile(profile="smoke", *, announce=False):
    """Run the same Hamiltonian in each representation selected by the profile."""
    config = PROFILES[profile] if isinstance(profile, str) else profile
    results = {}
    for label in config.methods:
        if announce:
            print(f"[{config.name}] {label}: starting")
        horizon = (
            config.comparison_t_max
            if label != "interaction" and config.comparison_t_max is not None
            else config.t_max
        )
        results[label] = make_model(config).run(
            dt=config.dt,
            n_steps=int(np.ceil(horizon / config.dt)),
            method=METHODS[label],
            trunc_eps=config.trunc_eps,
            bond_dim=None,
            initial="up",
            observables={"population_up": 0.5 * (np.eye(2) + sigma_z)},
        )
    return {"profile": config, "results": results}


def load_paper_figure8(path=PAPER_FIG8_DATA):
    """Return vector-path samples of the published converged IC10 curve."""
    table = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "t_delta_over_pi": np.asarray(table["t_delta_over_pi"], float),
        "population_up": np.asarray(table["population_up"], float),
    }


def summarize(suite):
    results = suite["results"]
    primary = np.asarray(results["interaction"].expect["population_up"], float)
    comparisons = {}
    for label, result in results.items():
        if label == "interaction":
            continue
        values = np.asarray(result.expect["population_up"], float)
        overlap = min(len(values), len(primary))
        comparisons[label] = float(np.max(np.abs(
            values[:overlap] - primary[:overlap]
        )))
    paper = load_paper_figure8()
    paper_mask = paper["t_delta_over_pi"] <= (
        results["interaction"].t[-1] / np.pi + 1e-6
    )
    paper_t = paper["t_delta_over_pi"][paper_mask]
    paper_population = paper["population_up"][paper_mask]
    simulated_at_paper_times = np.interp(
        paper_t,
        np.r_[0.0, results["interaction"].t / np.pi],
        np.r_[1.0, primary],
    )
    residual = simulated_at_paper_times - paper_population
    return {
        "final_population_up": float(primary[-1]),
        "paper_curve_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "paper_curve_max_error": float(np.max(np.abs(residual))),
        "paper_points_compared": int(len(paper_t)),
        "max_bond": {label: int(np.max(result.max_bond))
                     for label, result in results.items()},
        "max_difference_from_interaction": comparisons,
    }


def save_suite(suite, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for label, result in suite["results"].items():
        payload[f"{label}_t"] = result.t
        payload[f"{label}_population_up"] = result.expect["population_up"]
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
