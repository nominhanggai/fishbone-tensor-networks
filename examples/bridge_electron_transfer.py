"""Donor--bridge--acceptor electron transfer with non-Condon fluctuations.

The parameters follow Acharyya, Ovcharenko, and Fingerhut, J. Chem. Phys. 153,
185101 (2020), DOI:10.1063/5.0027976. Energies are specified in inverse
centimetres and converted consistently to angular ps units before propagation.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fishbonett import Bath, SystemBath


CM_TO_RAD_PS = 2.0 * np.pi * 2.99792458e10 * 1e-12
KB_CM_PER_K = 0.6950348009
TEMPERATURE_K = 300.0

PROJECTORS = {
    "donor": np.diag([1.0, 0.0, 0.0]),
    "bridge": np.diag([0.0, 1.0, 0.0]),
    "acceptor": np.diag([0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class Profile:
    name: str
    t_max_ps: float
    dt_ps: float
    n_modes: int | None
    variants: tuple


PROFILES = {
    "smoke": Profile("smoke", 0.008, 0.002, 4, (("primary", 3, 1e-3),)),
    "docs": Profile("docs", 0.2, 0.005, 12, (("primary", 6, 1e-3),)),
    "reference": Profile(
        "reference", 10.0, 0.001, None,
        (("primary", 20, 1e-3), ("fock", 40, 1e-3),
         ("svd", 20, 5e-4)),
    ),
}


def _case(case):
    energies = np.array([0.0, -150.0, -1000.0])
    hamiltonian = np.diag(energies)
    coupling = np.diag([2.0, 1.0, 0.0])
    if case == "condon":
        hamiltonian[0, 1] = hamiltonian[1, 0] = 22.0
        hamiltonian[1, 2] = hamiltonian[2, 1] = 45.0
    elif case == "noncondon":
        hamiltonian[0, 1] = hamiltonian[1, 0] = 2.0
        hamiltonian[1, 2] = hamiltonian[2, 1] = 2.0
        coupling[0, 1] = coupling[1, 0] = 0.17
        coupling[1, 2] = coupling[2, 1] = 0.055
    else:
        raise ValueError("case must be 'condon' or 'noncondon'")
    return CM_TO_RAD_PS * hamiltonian, coupling


def spectral_density(omega):
    """Ohmic density transformed from cm^-1 to angular ps units."""
    cutoff_cm = 100.0
    alpha = 10.02
    omega_cm = omega / CM_TO_RAD_PS
    density_cm = 0.5 * alpha * np.pi * omega_cm * np.exp(-omega_cm / cutoff_cm)
    return CM_TO_RAD_PS * density_cm


def make_model(case, *, phys_dim, n_modes):
    hamiltonian, coupling = _case(case)
    beta = 1.0 / (KB_CM_PER_K * TEMPERATURE_K * CM_TO_RAD_PS)
    bath = Bath(
        J=spectral_density,
        beta=beta,
        phys_dim=phys_dim,
        n_modes=n_modes,
        discretization="tedopa",
    )
    return SystemBath(h=hamiltonian, coupling=coupling, bath=bath)


def run_profile(profile="smoke", *, announce=False):
    config = PROFILES[profile] if isinstance(profile, str) else profile
    results = {}
    for case in ("condon", "noncondon"):
        results[case] = {}
        for label, phys_dim, trunc_eps in config.variants:
            if announce:
                print(f"[{config.name}] {case}/{label}: starting")
            results[case][label] = make_model(
                case, phys_dim=phys_dim, n_modes=config.n_modes,
            ).run(
                dt=config.dt_ps,
                t_max=config.t_max_ps,
                method="interaction-chain-trotter-mpo",
                trunc_eps=trunc_eps,
                bond_dim=None,
                initial=np.array([1.0, 0.0, 0.0]),
                observables=PROJECTORS,
            )
    return {"profile": config, "results": results}


def effective_lifetime(result):
    """Descriptive exponential lifetime of the resolved donor decay.

    This is not an elementary forward rate: bridge recrossing and back transfer
    are part of the fitted population trace.
    """
    population = np.asarray(result.expect["donor"], float)
    mask = (population > 0.15) & (population < 0.9)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    slope, _intercept = np.polyfit(result.t[mask], np.log(population[mask]), 1)
    return float(-1.0 / slope) if slope < 0 else float("nan")


def summarize(suite):
    summary = {}
    for case, variants in suite["results"].items():
        primary = variants["primary"]
        total = sum(np.asarray(primary.expect[name], float) for name in PROJECTORS)
        summary[case] = {
            "effective_lifetime_ps": effective_lifetime(primary),
            "peak_bridge_population": float(np.max(primary.expect["bridge"])),
            "final_acceptor_population": float(primary.expect["acceptor"][-1]),
            "normalization_error": float(np.max(np.abs(total - 1.0))),
            "max_bond": int(np.max(primary.max_bond)),
        }
        for label, result in variants.items():
            if label == "primary":
                continue
            difference = max(
                float(np.max(np.abs(
                    np.asarray(result.expect[name])
                    - np.asarray(primary.expect[name])
                ))) for name in PROJECTORS
            )
            summary[case][f"max_population_change_{label}"] = difference
    return summary


def save_suite(suite, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for case, variants in suite["results"].items():
        for label, result in variants.items():
            prefix = f"{case}_{label}"
            payload[f"{prefix}_t"] = result.t
            payload[f"{prefix}_populations"] = np.column_stack(
                [result.expect[name] for name in PROJECTORS]
            )
            payload[f"{prefix}_max_bond"] = result.max_bond
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
