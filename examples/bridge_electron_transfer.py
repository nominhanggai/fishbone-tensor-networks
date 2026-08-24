"""Fig. 2 donor--bridge--acceptor model with non-Condon fluctuations.

The parameters follow Acharyya, Ovcharenko, and Fingerhut, J. Chem. Phys. 153,
185101 (2020), DOI:10.1063/5.0027976. Energies are specified in inverse
centimetres and converted consistently to angular ps units before propagation.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from fishbonett import Bath, SystemBath
from fishbonett.rates import predict_density_mat, transfer_mat


CM_TO_RAD_PS = 2.0 * np.pi * 2.99792458e10 * 1e-12
KB_CM_PER_K = 0.6950348009
TEMPERATURE_K = 300.0
BATH_ALPHA = 1.67
BATH_CUTOFF_CM = 600.0
REORGANIZATION_CM = 0.5 * BATH_ALPHA * BATH_CUTOFF_CM
PAPER_LIFETIMES_PS = {
    "diagonal_reference": 2.36,
    "noncondon": 2.50,
}
REFERENCE_DATA = Path(__file__).with_name("reference_data")
REFERENCE_MAPS = REFERENCE_DATA / "bridge_electron_transfer_ttm_maps.npz"
PAPER_FIG2_DATA = REFERENCE_DATA / "acharyya_2021_fig2_populations.csv"
REFERENCE_METHOD = "interaction-chain-trotter-mpo"
SIMULATION_LABEL = "tensor network + TTM"
PAPER_LABEL = "digitized paper Fig. 2"

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
        "reference", 0.15, 0.002, None, (("primary", 6, 1e-4),),
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


def quapi_equivalent_hamiltonian(case):
    """Explicit-bath Hamiltonian matching the paper's QUAPI convention.

    The paper diagonalizes ``M = U D U^dagger`` before applying the standard
    Makri influence coefficients.  Those coefficients include the local
    reorganization contribution ``lambda_R D^2``.  An explicit harmonic-bath
    propagation therefore uses ``lambda_R U D^2 U^dagger``, which is exactly
    ``lambda_R M^2``.  This is a conversion between propagation conventions;
    it is not an extra term printed in the paper's Eq. (1).
    """
    hamiltonian, coupling = _case(case)
    renormalization = (
        CM_TO_RAD_PS * REORGANIZATION_CM * (coupling @ coupling)
    )
    return hamiltonian + renormalization, coupling


def spectral_density(omega):
    """Ohmic density transformed from cm^-1 to angular ps units."""
    omega_cm = omega / CM_TO_RAD_PS
    density_cm = (
        0.5 * BATH_ALPHA * np.pi * omega_cm
        * np.exp(-omega_cm / BATH_CUTOFF_CM)
    )
    return CM_TO_RAD_PS * density_cm


def make_model(case, *, phys_dim, n_modes, domain):
    hamiltonian, coupling = quapi_equivalent_hamiltonian(case)
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
                method=REFERENCE_METHOD,
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


def tomography_states(dimension=3):
    """Pure states needed to span a ``dimension``-state Liouville space.

    Three populations and two superpositions for every pair reconstruct all
    nine columns of a three-state dynamical map without propagating nonphysical
    matrix units directly.
    """
    states = {}
    basis = np.eye(dimension, dtype=complex)
    for i in range(dimension):
        states[f"d{i}"] = basis[i]
    for i in range(dimension):
        for j in range(i + 1, dimension):
            states[f"r{i}{j}"] = (basis[i] + basis[j]) / np.sqrt(2.0)
            states[f"i{i}{j}"] = (basis[i] + 1j * basis[j]) / np.sqrt(2.0)
    return states


def assemble_dynamical_maps(rdms):
    """Assemble three-state dynamical maps from tomography trajectories."""
    diagonal = [np.asarray(rdms[f"d{i}"], complex) for i in range(3)]
    n_times = len(diagonal[0])
    if any(values.shape != (n_times, 3, 3) for values in diagonal):
        raise ValueError("each tomography trajectory must have shape (time, 3, 3)")

    propagated = {(i, i): diagonal[i] for i in range(3)}
    for i in range(3):
        for j in range(i + 1, 3):
            real_run = np.asarray(rdms[f"r{i}{j}"], complex)
            imag_run = np.asarray(rdms[f"i{i}{j}"], complex)
            if real_run.shape != (n_times, 3, 3) or imag_run.shape != (
                n_times, 3, 3
            ):
                raise ValueError(
                    "each tomography trajectory must have shape (time, 3, 3)"
                )
            population_sum = diagonal[i] + diagonal[j]
            propagated[(i, j)] = (
                real_run + 1j * imag_run
                - 0.5 * (1.0 + 1j) * population_sum
            )
            propagated[(j, i)] = (
                real_run - 1j * imag_run
                - 0.5 * (1.0 - 1j) * population_sum
            )

    maps = np.empty((n_times, 9, 9), dtype=complex)
    for column, index in enumerate(
        (i, j) for i in range(3) for j in range(3)
    ):
        maps[:, :, column] = propagated[index].reshape(n_times, 9)
    return maps


def run_reference_tomography(*, announce=False):
    """Generate the short dynamical maps used for the 15 ps validation.

    This is the expensive, tensor-network part of the reference workflow.  It
    performs nine 0.15 ps propagations for each coupling model.  The resulting
    maps can be saved and cheaply extrapolated with :func:`long_validation`.
    """
    config = PROFILES["reference"]
    _label, phys_dim, trunc_eps = config.variants[0]
    beta = 1.0 / (KB_CM_PER_K * TEMPERATURE_K * CM_TO_RAD_PS)
    resolved_bath = Bath(
        J=spectral_density,
        beta=beta,
        phys_dim=1,
        n_modes=config.n_modes,
        discretization="tedopa",
    ).resolved(config.t_max_ps)
    states = tomography_states(3)
    maps = {}
    for case in ("diagonal_reference", "noncondon"):
        rdms = {}
        for label, initial in states.items():
            if announce:
                print(f"[reference] {case}/{label}: starting")
            result = make_model(
                case,
                phys_dim=phys_dim,
                n_modes=resolved_bath.n_modes,
                domain=resolved_bath.domain,
            ).run(
                dt=config.dt_ps,
                t_max=config.t_max_ps,
                method=REFERENCE_METHOD,
                trunc_eps=trunc_eps,
                bond_dim=None,
                initial=initial,
                observables=PROJECTORS,
            )
            rdms[label] = result.rdm
        maps[case] = assemble_dynamical_maps(rdms)
    return {
        "dt_ps": config.dt_ps,
        "memory_ps": config.t_max_ps,
        "phys_dim": phys_dim,
        "trunc_eps": trunc_eps,
        "bath_n_modes": resolved_bath.n_modes,
        "bath_domain_cm": tuple(
            value / CM_TO_RAD_PS for value in resolved_bath.domain
        ),
        "maps": maps,
    }


def save_reference_maps(reference, path):
    """Save the dynamical maps and their numerical metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dt_ps": np.array(reference["dt_ps"]),
        "memory_ps": np.array(reference["memory_ps"]),
        "phys_dim": np.array(reference["phys_dim"]),
        "trunc_eps": np.array(reference["trunc_eps"]),
        "bath_n_modes": np.array(reference["bath_n_modes"]),
        "bath_domain_cm": np.asarray(reference["bath_domain_cm"], float),
        "bath_alpha": np.array(BATH_ALPHA),
        "bath_cutoff_cm": np.array(BATH_CUTOFF_CM),
        "temperature_k": np.array(TEMPERATURE_K),
        "method": np.array(REFERENCE_METHOD),
    }
    for case, maps in reference["maps"].items():
        payload[f"{case}_maps"] = maps
    np.savez_compressed(path, **payload)


def load_paper_fig2(path=PAPER_FIG2_DATA):
    """Load populations digitized from the vector paths in paper Fig. 2."""
    table = np.genfromtxt(
        path, delimiter=",", names=True, dtype=None, encoding="ascii"
    )
    curves = {}
    for case in PAPER_LIFETIMES_PS:
        selected = table[table["case"] == case]
        curves[case] = {
            "t": np.asarray(selected["time_ps"], float),
            "populations": np.column_stack([
                selected["donor"], selected["bridge"], selected["acceptor"],
            ]).astype(float),
        }
    return curves


def fit_donor_lifetime(times, donor):
    """Fit ``A exp(-t/tau) + C`` and return ``A``, ``tau``, ``C``, and RMSE."""
    times = np.asarray(times, float)
    donor = np.asarray(donor, float)

    def model(time, amplitude, lifetime, offset):
        return amplitude * np.exp(-time / lifetime) + offset

    parameters, _covariance = curve_fit(
        model,
        times,
        donor,
        p0=(0.95, 2.5, 0.01),
        bounds=([0.0, 0.01, -0.2], [2.0, 100.0, 0.2]),
    )
    residual = donor - model(times, *parameters)
    return {
        "amplitude": float(parameters[0]),
        "lifetime_ps": float(parameters[1]),
        "offset": float(parameters[2]),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def dynamical_map_diagnostics(maps):
    """Return trace-preservation and Choi-positivity diagnostics."""
    maps = np.asarray(maps, complex)
    trace_functional = np.eye(3).reshape(9)
    trace_error = 0.0
    minimum_choi_eigenvalue = np.inf
    for dynamical_map in maps:
        propagated_trace = dynamical_map[[0, 4, 8], :].sum(axis=0)
        trace_error = max(
            trace_error,
            float(np.max(np.abs(propagated_trace - trace_functional))),
        )
        choi = dynamical_map.reshape(3, 3, 3, 3).transpose(
            2, 0, 3, 1
        ).reshape(9, 9)
        choi = 0.5 * (choi + choi.conj().T)
        minimum_choi_eigenvalue = min(
            minimum_choi_eigenvalue,
            float(np.linalg.eigvalsh(choi).min()),
        )
    return {
        "trace_error": trace_error,
        "minimum_choi_eigenvalue": minimum_choi_eigenvalue,
    }


def memory_cutoff_convergence(
    maps, direct, reference_populations, *, dt_ps, n_steps,
    first_cutoff_ps=0.04, cutoff_step_ps=0.01,
):
    """Converge the long trajectory against retained TTM memory length.

    The longest available map is the reference.  Shorter prefixes are
    independently deconvolved and propagated to the same final time, so the
    returned population error measures the consequence of discarding the tail
    of the transfer-tensor kernel rather than only its instantaneous norm.
    """
    maps = np.asarray(maps, complex)
    direct = np.asarray(direct, complex)
    reference_populations = np.asarray(reference_populations, float)
    if len(maps) != len(direct):
        raise ValueError("maps and direct trajectory must have equal lengths")
    if n_steps < len(maps):
        raise ValueError("n_steps must include the complete direct trajectory")
    if reference_populations.shape != (n_steps, 3):
        raise ValueError("reference_populations must have shape (n_steps, 3)")

    first_depth = max(1, int(round(first_cutoff_ps / dt_ps)))
    depth_step = max(1, int(round(cutoff_step_ps / dt_ps)))
    if first_depth > len(maps):
        raise ValueError("first memory cutoff exceeds the available maps")
    depths = np.arange(first_depth, len(maps) + 1, depth_step, dtype=int)
    if depths[-1] != len(maps):
        depths = np.append(depths, len(maps))

    population_errors = []
    lifetimes = []
    times = np.arange(1, n_steps + 1, dtype=float) * dt_ps
    for depth in depths:
        transfer, _transfer_norm = transfer_mat(maps[:depth])
        predicted = predict_density_mat(n_steps, transfer, direct[:depth])
        populations = np.diagonal(predicted, axis1=1, axis2=2).real
        population_errors.append(float(np.max(np.abs(
            populations - reference_populations
        ))))
        lifetimes.append(
            fit_donor_lifetime(times, populations[:, 0])["lifetime_ps"]
        )
    return {
        "cutoff_ps": depths.astype(float) * dt_ps,
        "max_population_difference": np.asarray(population_errors),
        "donor_lifetime_ps": np.asarray(lifetimes),
    }


def long_validation(
    maps_path=REFERENCE_MAPS, paper_path=PAPER_FIG2_DATA, *, t_max_ps=15.0
):
    """Extrapolate the short maps and compare with digitized paper dynamics."""
    with np.load(maps_path, allow_pickle=False) as archive:
        dt_ps = float(archive["dt_ps"])
        metadata = {
            name: float(archive[name])
            for name in (
                "memory_ps", "trunc_eps", "bath_alpha", "bath_cutoff_cm",
                "temperature_k",
            )
        }
        metadata.update({
            "dt_ps": dt_ps,
            "phys_dim": int(archive["phys_dim"]),
            "bath_n_modes": int(archive["bath_n_modes"]),
            "bath_domain_cm": tuple(
                np.asarray(archive["bath_domain_cm"], float)
            ),
            "method": str(archive["method"]),
        })
        stored_maps = {
            case: np.asarray(archive[f"{case}_maps"], complex)
            for case in PAPER_LIFETIMES_PS
        }

    paper = load_paper_fig2(paper_path)
    n_steps = int(round(t_max_ps / dt_ps))
    initial = np.diag([1.0, 0.0, 0.0]).astype(complex)
    initial_vector = initial.reshape(9)
    results = {}
    summary = {}
    for case, maps in stored_maps.items():
        transfer, transfer_norm = transfer_mat(maps)
        direct = np.einsum("tij,j->ti", maps, initial_vector).reshape(-1, 3, 3)
        holdout_depth = int(round(0.12 / dt_ps))
        if not 0 < holdout_depth < len(maps):
            raise ValueError("stored maps do not extend beyond the holdout depth")
        holdout_transfer, _holdout_norm = transfer_mat(
            maps[:holdout_depth]
        )
        holdout_prediction = predict_density_mat(
            len(maps), holdout_transfer, direct[:holdout_depth]
        )
        heldout_population_error = float(np.max(np.abs(
            np.diagonal(
                holdout_prediction[holdout_depth:], axis1=1, axis2=2
            ).real
            - np.diagonal(direct[holdout_depth:], axis1=1, axis2=2).real
        )))
        map_diagnostics = dynamical_map_diagnostics(maps)
        predicted = predict_density_mat(n_steps, transfer, direct)
        times = np.arange(1, n_steps + 1, dtype=float) * dt_ps
        populations = np.real_if_close(
            np.diagonal(predicted, axis1=1, axis2=2)
        ).real
        paper_times = paper[case]["t"]
        paper_populations = paper[case]["populations"]
        simulation_at_paper_times = np.column_stack([
            np.interp(
                paper_times,
                np.r_[0.0, times],
                np.r_[initial[state, state].real, populations[:, state]],
            )
            for state in range(3)
        ])
        residual = simulation_at_paper_times - paper_populations
        simulation_fit = fit_donor_lifetime(times, populations[:, 0])
        paper_fit = fit_donor_lifetime(
            paper_times, paper_populations[:, 0]
        )
        memory_convergence = memory_cutoff_convergence(
            maps, direct, populations, dt_ps=dt_ps, n_steps=n_steps
        )
        results[case] = {
            "t": times,
            "rdm": predicted,
            "populations": populations,
            "transfer_norm": np.asarray(transfer_norm, float),
            "paper_t": paper_times,
            "paper_populations": paper_populations,
            "residual": residual,
            "fit": simulation_fit,
            "paper_fit": paper_fit,
            "memory_convergence": memory_convergence,
        }
        summary[case] = {
            "population_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "max_population_error": float(np.max(np.abs(residual))),
            "state_rmse": np.sqrt(np.mean(residual ** 2, axis=0)),
            "lifetime_ps": simulation_fit["lifetime_ps"],
            "paper_curve_lifetime_ps": paper_fit["lifetime_ps"],
            "reported_lifetime_ps": PAPER_LIFETIMES_PS[case],
            "final_populations": populations[-1],
            "last_transfer_norm": float(transfer_norm[-1]),
            "heldout_population_error": heldout_population_error,
            "direct_map_trace_error": map_diagnostics["trace_error"],
            "direct_map_minimum_choi_eigenvalue": map_diagnostics[
                "minimum_choi_eigenvalue"
            ],
            "trace_error": float(np.max(np.abs(np.trace(predicted, axis1=1, axis2=2) - 1.0))),
            "minimum_eigenvalue": float(min(
                np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min()
                for rho in predicted
            )),
        }
    return {"metadata": metadata, "results": results, "summary": summary}


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
    parser.add_argument(
        "--generate-reference-maps", type=Path,
        help="run 18 short tomography calculations and save their maps",
    )
    parser.add_argument(
        "--validate-maps", type=Path, nargs="?", const=REFERENCE_MAPS,
        help="propagate stored maps to 15 ps and compare with paper Fig. 2",
    )
    args = parser.parse_args(argv)
    if args.generate_reference_maps:
        reference = run_reference_tomography(announce=True)
        save_reference_maps(reference, args.generate_reference_maps)
        print("saved reference maps:", args.generate_reference_maps)
        return reference
    if args.validate_maps:
        validation = long_validation(args.validate_maps)
        print("reference metadata:", validation["metadata"])
        print("paper comparison:", validation["summary"])
        return validation
    suite = run_profile(args.profile, announce=True)
    if args.output:
        save_suite(suite, args.output)
    print("resolved bath:", suite["bath"])
    print(summarize(suite))
    return suite


if __name__ == "__main__":
    main()
