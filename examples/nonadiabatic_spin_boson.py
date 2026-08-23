"""Strong-coupling nonadiabatic spin--boson dynamics.

This is the high-temperature benchmark of Nuomin, Beratan, and Zhang
(arXiv:2111.14308): Delta=1, eta=4, omega_c=4, and T=4. The default profile is
a four-step engine check; use ``--profile docs`` for the resolved comparison.
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
        "docs", np.pi, 0.025, 6, 24, (-16.0, 80.0), 1e-3,
        ("interaction", "schrodinger_chain"), 0.25 * np.pi,
    ),
    "reference": Profile(
        "reference", 5.0 * np.pi, 0.0125, 20, None, None, 5e-4,
        ("interaction", "schrodinger_chain", "schrodinger_star"),
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
            n_steps=int(round(horizon / config.dt)),
            method=METHODS[label],
            trunc_eps=config.trunc_eps,
            bond_dim=None,
            initial="up",
            observables={"population_up": 0.5 * (np.eye(2) + sigma_z)},
        )
    return {"profile": config, "results": results}


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
    return {
        "final_population_up": float(primary[-1]),
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
