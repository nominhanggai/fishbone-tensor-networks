"""Fig. 2 donor--bridge--acceptor model with non-Condon fluctuations.

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
BATH_ALPHA = 1.67
BATH_CUTOFF_CM = 600.0
REORGANIZATION_CM = 0.5 * BATH_ALPHA * BATH_CUTOFF_CM

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
    "docs": Profile("docs", 0.2, 0.002, None, (("primary", 6, 1e-4),)),
    "reference": Profile(
        "reference", 15.0, 0.001, None,
        (("primary", 20, 1e-4), ("fock", 40, 1e-4),
         ("svd", 20, 5e-5)),
    ),
}


def _case(case):
    """Return the paper's diabatic Hamiltonian and bath-coupling matrix."""
    energies = np.array([0.0, -150.0, -1000.0])
    hamiltonian = np.diag(energies)
    coupling = np.diag([2.0, 1.0, 0.0])
    if case == "diagonal_reference":
        hamiltonian[0, 1] = hamiltonian[1, 0] = 22.0
        hamiltonian[1, 2] = hamiltonian[2, 1] = 45.0
    elif case in {"weak_diagonal", "noncondon"}:
        hamiltonian[0, 1] = hamiltonian[1, 0] = 2.0
        hamiltonian[1, 2] = hamiltonian[2, 1] = 2.0
        if case == "noncondon":
            coupling[0, 1] = coupling[1, 0] = 0.17
            coupling[1, 2] = coupling[2, 1] = 0.055
    else:
        raise ValueError(
            "case must be 'diagonal_reference', 'weak_diagonal', or "
            "'noncondon'"
        )
    return CM_TO_RAD_PS * hamiltonian, coupling


def propagation_hamiltonian(case):
    """Hamiltonian used with Fishbone's unshifted harmonic bath.

    This reproduction interprets the quoted diabatic energies as the minima of
    displaced bath potentials.  In the bilinear ``SystemBath`` convention that
    requires the reorganization counterterm ``lambda_R M^2``.  ``SystemBath``
    deliberately does not add a model-dependent counterterm itself.
    """
    hamiltonian, coupling = _case(case)
    counterterm = CM_TO_RAD_PS * REORGANIZATION_CM * (coupling @ coupling)
    return hamiltonian + counterterm, coupling


def spectral_density(omega):
    """Ohmic density transformed from cm^-1 to angular ps units."""
    omega_cm = omega / CM_TO_RAD_PS
    density_cm = (
        0.5 * BATH_ALPHA * np.pi * omega_cm
        * np.exp(-omega_cm / BATH_CUTOFF_CM)
    )
    return CM_TO_RAD_PS * density_cm


def make_model(case, *, phys_dim, n_modes, domain):
    hamiltonian, coupling = propagation_hamiltonian(case)
    beta = 1.0 / (KB_CM_PER_K * TEMPERATURE_K * CM_TO_RAD_PS)
    bath = Bath(
        J=spectral_density,
        beta=beta,
        phys_dim=phys_dim,
        n_modes=n_modes,
        domain=domain,
        discretization="tedopa",
    )
    return SystemBath(h=hamiltonian, coupling=coupling, bath=bath)


def run_profile(profile="smoke", *, announce=False):
    config = PROFILES[profile] if isinstance(profile, str) else profile
    beta = 1.0 / (KB_CM_PER_K * TEMPERATURE_K * CM_TO_RAD_PS)
    resolved_bath = Bath(
        J=spectral_density,
        beta=beta,
        phys_dim=1,
        n_modes=config.n_modes,
        discretization="tedopa",
    ).resolved(config.t_max_ps)
    results = {}
    for case in ("diagonal_reference", "weak_diagonal", "noncondon"):
        results[case] = {}
        for label, phys_dim, trunc_eps in config.variants:
            if announce:
                print(f"[{config.name}] {case}/{label}: starting")
            results[case][label] = make_model(
                case,
                phys_dim=phys_dim,
                n_modes=resolved_bath.n_modes,
                domain=resolved_bath.domain,
            ).run(
                dt=config.dt_ps,
                t_max=config.t_max_ps,
                method="interaction-chain-trotter-mpo",
                trunc_eps=trunc_eps,
                bond_dim=None,
                initial=np.array([1.0, 0.0, 0.0]),
                observables=PROJECTORS,
            )
    return {
        "profile": config,
        "bath": {
            "alpha": BATH_ALPHA,
            "cutoff_cm": BATH_CUTOFF_CM,
            "reorganization_cm": REORGANIZATION_CM,
            "domain_cm": tuple(
                value / CM_TO_RAD_PS for value in resolved_bath.domain
            ),
            "n_modes": resolved_bath.n_modes,
        },
        "results": results,
    }


def effective_lifetime(result):
    """Descriptive exponential lifetime of the resolved donor decay.

    This is not an elementary forward rate: bridge recrossing and back transfer
    are part of the fitted population trace.
    """
    population = np.asarray(result.expect["donor"], float)
    # An early non-Condon slip can cross 0.9 without sampling the kinetic decay.
    # Require substantial depopulation before fitting an exponential lifetime.
    if np.min(population) > 0.5:
        return float("nan")
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
            "final_donor_population": float(primary.expect["donor"][-1]),
            "donor_population_loss": float(
                1.0 - primary.expect["donor"][-1]
            ),
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
    bath = suite["bath"]
    payload = {
        "bath_alpha": np.array(bath["alpha"]),
        "bath_cutoff_cm": np.array(bath["cutoff_cm"]),
        "bath_reorganization_cm": np.array(bath["reorganization_cm"]),
        "bath_domain_cm": np.asarray(bath["domain_cm"]),
        "bath_n_modes": np.array(bath["n_modes"]),
    }
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
    print("resolved bath:", suite["bath"])
    print(summarize(suite))
    return suite


if __name__ == "__main__":
    main()
