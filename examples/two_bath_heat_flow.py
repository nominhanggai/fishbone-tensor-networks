"""Heat flow through a two-level junction coupled to hot and cold baths.

The model follows Dunnett and Chin, Entropy 23, 77 (2021),
DOI:10.3390/e23010077. It demonstrates two independent baths on one system site
and measures the current from explicit system--bath correlations.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fishbonett import Bath, BathMode, Fishbone
from fishbonett.operators import annihilate, sigma_x, sigma_y, sigma_z


OMEGA_C = 1.0
OMEGA_0 = 0.2
# Coupling of each reservoir in the two-bath benchmark.
ALPHA = 0.025


@dataclass(frozen=True)
class Profile:
    name: str
    t_max: float
    dt: float
    n_modes: int | None
    variants: tuple


PROFILES = {
    "smoke": Profile("smoke", 0.20, 0.05, 4, (("primary", 3, 1e-3),)),
    "docs": Profile("docs", 25.0, 0.1, None, (("primary", 5, 1e-3),)),
    "reference": Profile(
        "reference", 40.0, 0.025, None,
        (("primary", 6, 1e-3), ("fock", 8, 1e-3),
         ("svd", 6, 5e-4)),
    ),
}


def spectral_density(omega):
    """Hard-cutoff Ohmic density J(omega)=2 pi alpha omega."""
    return 2.0 * np.pi * ALPHA * omega if 0.0 <= omega <= OMEGA_C else 0.0


def make_bath(beta, *, phys_dim, n_modes):
    return Bath(
        J=spectral_density,
        beta=beta,
        domain=(-OMEGA_C, OMEGA_C),
        n_modes=n_modes,
        phys_dim=phys_dim,
        discretization="tedopa",
        extra_breaks=(0.0,),
    )


def make_model(*, beta_hot, beta_cold, phys_dim, n_modes):
    hot = make_bath(beta_hot, phys_dim=phys_dim, n_modes=n_modes).bind(sigma_x)
    cold = make_bath(beta_cold, phys_dim=phys_dim, n_modes=n_modes).bind(sigma_x)
    return Fishbone(
        sites=[0.5 * OMEGA_0 * sigma_z],
        baths={0: [hot, cold]},
    )


def _run_case(config, beta_hot, beta_cold, phys_dim, trunc_eps):
    model = make_model(
        beta_hot=beta_hot,
        beta_cold=beta_cold,
        phys_dim=phys_dim,
        n_modes=config.n_modes,
    )
    destroy = annihilate(phys_dim)
    position = destroy + destroy.T
    hot_mode = BathMode(system_site=0, bath=0, mode=0)
    cold_mode = BathMode(system_site=0, bath=1, mode=0)
    result = model.run(
        dt=config.dt,
        t_max=config.t_max,
        method="schrodinger-chain-tree-tebd",
        trunc_eps=trunc_eps,
        bond_dim=None,
        initial=[np.array([0.0, 1.0])],
        observables={
            "sz": (sigma_z, 0),
            "hot_system_mode": (np.kron(sigma_y, position), (0, hot_mode)),
            "cold_system_mode": (np.kron(sigma_y, position), (0, cold_mode)),
        },
    )
    branches = result.meta["bath_branches"]
    hot_coupling = branches[0]["system_coupling"]
    cold_coupling = branches[1]["system_coupling"]
    hot_to_system = (
        hot_coupling * OMEGA_0
        * np.asarray(result.expect["hot_system_mode"], float)
    )
    cold_to_system = (
        cold_coupling * OMEGA_0
        * np.asarray(result.expect["cold_system_mode"], float)
    )
    return {
        "result": result,
        "hot_to_system": hot_to_system,
        "cold_to_system": cold_to_system,
    }


def run_profile(profile="smoke", *, announce=False):
    config = PROFILES[profile] if isinstance(profile, str) else profile
    results = {"nonequilibrium": {}, "equilibrium": {}}
    for label, phys_dim, trunc_eps in config.variants:
        if announce:
            print(f"[{config.name}] nonequilibrium/{label}: starting")
        results["nonequilibrium"][label] = _run_case(
            config, beta_hot=2.0, beta_cold=100.0,
            phys_dim=phys_dim, trunc_eps=trunc_eps,
        )
        if announce:
            print(f"[{config.name}] equilibrium/{label}: starting")
        results["equilibrium"][label] = _run_case(
            config, beta_hot=100.0, beta_cold=100.0,
            phys_dim=phys_dim, trunc_eps=trunc_eps,
        )
    return {"profile": config, "results": results}


def _case_summary(case):
    result = case["result"]
    hot = case["hot_to_system"]
    cold = case["cold_to_system"]
    energy = 0.5 * OMEGA_0 * np.asarray(result.expect["sz"], float)
    derivative = np.gradient(energy, result.t)
    continuity = derivative - hot - cold
    interior = continuity[1:-1] if len(continuity) > 2 else continuity
    steady = slice(max(0, int(0.8 * len(result.t))), None)
    return {
        "mean_hot_to_system": float(np.mean(hot[steady])),
        "mean_cold_to_system": float(np.mean(cold[steady])),
        "steady_balance_error": float(abs(np.mean((hot + cold)[steady]))),
        "continuity_rms": float(np.sqrt(np.mean(interior**2))),
        "final_sz": float(result.expect["sz"][-1]),
        "max_bond": int(np.max(result.max_bond)),
    }


def summarize(suite):
    summary = {}
    for condition, variants in suite["results"].items():
        summary[condition] = _case_summary(variants["primary"])
        primary = variants["primary"]
        for label, case in variants.items():
            if label == "primary":
                continue
            change = max(
                float(np.max(np.abs(
                    case[direction] - primary[direction]
                ))) for direction in ("hot_to_system", "cold_to_system")
            )
            summary[condition][f"max_current_change_{label}"] = change
    return summary


def save_suite(suite, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for condition, variants in suite["results"].items():
        for label, case in variants.items():
            prefix = f"{condition}_{label}"
            result = case["result"]
            payload[f"{prefix}_t"] = result.t
            payload[f"{prefix}_sz"] = result.expect["sz"]
            payload[f"{prefix}_hot_to_system"] = case["hot_to_system"]
            payload[f"{prefix}_cold_to_system"] = case["cold_to_system"]
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
