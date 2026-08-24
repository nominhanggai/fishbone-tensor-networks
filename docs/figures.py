"""Regenerate documentation figures from the public examples.

Figures and numerical summaries are build artifacts. ``docs/conf.py`` invokes
``build_all`` so a documentation build runs the same profiled calculations that
the tutorial text describes.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
IMG = DOCS / "img"
GENERATED = DOCS / "_generated"
T_MAX = 4.0
_TS = np.linspace(0.0, T_MAX, 400)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load_example(name):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"docs_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary(name, text):
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / f"{name}.md").write_text(text, encoding="utf-8")


def _c_disc(density, domain, n_modes, times):
    from fishbonett.bath.legendre import get_vn_squared
    frequency, weight = get_vn_squared(density, n_modes, list(domain))
    return (
        np.asarray(weight)[None, :] / np.pi
        * np.exp(-1j * np.outer(times, frequency))
    ).sum(axis=1)


def _panel(axis, times, exact, curves, title):
    axis.plot(times, exact.real, "-", color="#4C6EF5", lw=2.4,
              label=r"exact Re $C(t)$")
    axis.plot(times, exact.imag, "-", color="#E8590C", lw=2.4,
              label=r"exact Im $C(t)$")
    stride = slice(None, None, max(1, len(times) // 28))
    axis.plot(times[stride], curves["auto"].real[stride], "o", ms=5,
              mfc="none", color="#4C6EF5", label="automatic (Re)")
    axis.plot(times[stride], curves["auto"].imag[stride], "s", ms=5,
              mfc="none", color="#E8590C", label="automatic (Im)")
    axis.set(xlabel="time", ylabel="$C(t)$", title=title)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)


def _error_inset(axis, times, exact, curves, location=(0.44, 0.44, 0.52, 0.36)):
    inset = axis.inset_axes(location)
    scale = abs(exact[0])
    styles = {
        "auto": ("#2B8A3E", "-", "automatic"),
        "few": ("#868e96", "--", "too few modes"),
        "narrow": ("#C92A2A", ":", "domain too narrow"),
    }
    for key, (color, style, label) in styles.items():
        if key in curves:
            inset.semilogy(
                times, np.abs(curves[key] - exact) / scale, style,
                color=color, lw=1.4, label=label,
            )
    inset.set_ylim(1e-4, 2.0)
    inset.set_xlabel("$t$", fontsize=7)
    inset.set_ylabel("relative error", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.25, which="both")
    inset.legend(fontsize=6, frameon=False, loc="lower right")


def bath_correlation(path=None):
    """Zero-temperature Ohmic bath and two deliberately degraded grids."""
    from fishbonett import Bath
    plt = _mpl()
    eta, cutoff = 0.2, 5.0
    density = lambda omega: eta * omega * np.exp(-omega / cutoff)
    exact = (eta / np.pi) / (1.0 / cutoff + 1j * _TS) ** 2
    bath = Bath(J=density, phys_dim=10).resolved(T_MAX)
    curves = {
        "auto": _c_disc(density, bath.domain, bath.n_modes, _TS),
        "few": _c_disc(density, bath.domain, 20, _TS),
        "narrow": _c_disc(density, (0.0, 10.0), bath.n_modes, _TS),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    _panel(axis, _TS, exact, curves,
           f"Ohmic bath, automatic domain and {bath.n_modes} modes")
    _error_inset(axis, _TS, exact, curves)
    figure.tight_layout()
    output = Path(path or IMG / "bath_correlation.png")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def bath_correlation_finite_t(path=None):
    """Finite-temperature correlation on the thermofield signed domain."""
    from fishbonett import Bath, thermalize
    from scipy.integrate import quad
    plt = _mpl()
    eta, cutoff, temperature = 0.2, 5.0, 1.0
    beta = 1.0 / temperature
    density = lambda omega: eta * omega * np.exp(-omega / cutoff)

    def exact_c(time):
        real = quad(
            lambda omega: density(omega) / np.pi
            / np.tanh(beta * omega / 2.0) * np.cos(omega * time),
            0, 40 * cutoff, limit=400,
        )[0]
        imaginary = -quad(
            lambda omega: density(omega) / np.pi * np.sin(omega * time),
            0, 40 * cutoff, limit=400,
        )[0]
        return real + 1j * imaginary

    exact = np.array([exact_c(time) for time in _TS])
    bath = Bath(J=density, temperature=temperature, phys_dim=10).resolved(T_MAX)
    thermal_density = thermalize(density, beta)
    curves = {
        "auto": _c_disc(thermal_density, bath.domain, bath.n_modes, _TS),
        "few": _c_disc(thermal_density, bath.domain, 20, _TS),
        "narrow": _c_disc(thermal_density, (-1.0, 10.0), bath.n_modes, _TS),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    _panel(axis, _TS, exact, curves,
           f"Finite-temperature bath, signed domain, {bath.n_modes} modes")
    _error_inset(axis, _TS, exact, curves)
    figure.tight_layout()
    output = Path(path or IMG / "bath_correlation_finiteT.png")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def bath_structured(path=None):
    """Structured density: Ohmic background plus two Brownian peaks."""
    from fishbonett import Bath
    from fishbonett.bath.legendre import get_vn_squared
    from scipy.integrate import quad
    plt = _mpl()

    def density(omega):
        omega = np.asarray(omega, float)
        result = 0.05 * omega * np.exp(-omega / 2.5)
        for reorganization, damping, centre in ((0.6, 1.2, 6.0),
                                                 (0.5, 1.0, 13.0)):
            result += (
                2 * reorganization * damping * centre**2 * omega
                / ((centre**2 - omega**2) ** 2 + damping**2 * omega**2)
            )
        return result

    exact = np.array([
        quad(lambda omega: density(omega) / np.pi * np.cos(omega * time),
             0, 60, limit=600)[0]
        - 1j * quad(lambda omega: density(omega) / np.pi * np.sin(omega * time),
                    0, 60, limit=600)[0]
        for time in _TS
    ])
    bath = Bath(J=density, phys_dim=10).resolved(T_MAX)
    curves = {"auto": _c_disc(density, bath.domain, bath.n_modes, _TS)}
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    grid = np.linspace(1e-3, float(bath.domain[1]) * 1.05, 900)
    left.plot(grid, density(grid), color="#4C6EF5", lw=2.0,
              label=r"$J(\omega)$")
    frequency, _weight = get_vn_squared(
        density, bath.n_modes, list(bath.domain),
    )
    left.plot(frequency, density(np.asarray(frequency)), "o", ms=3,
              color="#E8590C", label=f"{bath.n_modes} star modes")
    left.set(xlabel=r"$\omega$", ylabel=r"$J(\omega)$",
             title="structured spectral density")
    left.legend(frameon=False, fontsize=8)
    left.grid(alpha=0.25)
    _panel(right, _TS, exact, curves, "correlation function")
    _error_inset(right, _TS, exact, curves, (0.46, 0.60, 0.50, 0.34))
    figure.tight_layout()
    output = Path(path or IMG / "bath_structured.png")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def vibronic_dimer(path=None):
    example = _load_example("vibronic_dimer")
    suite = example.run_profile("docs", announce=True)
    summary = example.summarize(suite)
    plt = _mpl()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for vibration, result in suite["results"].items():
        population = np.asarray(result.expect["population"])
        left.plot(result.t, population[:, 1], label=rf"$\omega_0={vibration:g}$")
    scan = summary["final_acceptor_population"]
    right.plot(list(scan), list(scan.values()), "o-", color="#6F42C1")
    left.set(xlabel=r"time ($J^{-1}$)", ylabel="acceptor population",
             title="vibronic dimer dynamics")
    horizon = suite["profile"].t_max
    right.set(xlabel=r"vibration $\omega_0/J$",
              ylabel=rf"$P_A({horizon:g}/J)$",
              title="frequency dependence")
    left.legend(frameon=False)
    for axis in (left, right):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output = Path(path or IMG / "vibronic_dimer.png")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    modes = ", ".join(
        f"{frequency:g}: {values[0]} modes/bath"
        for frequency, values in summary["resolved_modes"].items()
    )
    _write_summary("vibronic_dimer", f"""## Generated result

The documentation profile conserved the one-excitation population to
**{summary['normalization_error']:.2e}** and retained a peak bond dimension of
**{summary['max_bond']}**. The TEDOPA layout used {modes}; the frequency domain
was selected automatically and the documentation profile fixed this mode count
to keep the build bounded.

The calculation resolves how the transfer changes across the damping-scale and
near-resonant vibrations. The finite scan is evidence for this model and time
window, not a universal claim that one vibrational frequency is optimal.
""")
    return output


def nonadiabatic_spin_boson(path=None):
    example = _load_example("nonadiabatic_spin_boson")
    suite = example.run_profile("docs", announce=True)
    summary = example.summarize(suite)
    plt = _mpl()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for label, result in suite["results"].items():
        left.plot(result.t / np.pi, result.expect["population_up"], label=label)
    labels = list(summary["max_bond"])
    right.bar(labels, [summary["max_bond"][label] for label in labels],
              color=("#4C6EF5", "#E8590C"))
    left.set(xlabel=r"$t\Delta/\pi$", ylabel=r"$P_\uparrow(t)$",
             title="strong-coupling relaxation")
    right.set(ylabel="peak bond dimension",
              title="profile peak (different horizons)")
    left.legend(frameon=False)
    for axis in (left, right):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output = Path(path or IMG / "nonadiabatic_spin_boson.png")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    difference = summary["max_difference_from_interaction"].get(
        "schrodinger_chain", float("nan"),
    )
    _write_summary("nonadiabatic_spin_boson", f"""## Generated result

The interaction-chain and short Schrödinger-chain overlap calculation differ by
at most **{difference:.3g}** in the up-state population on their common interval.
Their peak retained bond dimensions are
**{summary['max_bond']['interaction']}** and
**{summary['max_bond']['schrodinger_chain']}**, respectively. These bond values
must not be read as an equal-time performance comparison because the TDVP check
ends earlier.

Agreement of two representations is a useful cross-check, but publication-scale
work should also halve the time step and repeat the local-Fock-space and SVD
threshold checks in the reference profile.
""")
    return output


def bridge_electron_transfer(path=None):
    example = _load_example("bridge_electron_transfer")
    suite = example.run_profile("docs", announce=True)
    early_summary = example.summarize(suite)
    validation = example.long_validation()
    plt = _mpl()
    figure, axes = plt.subplots(
        2, 2, figsize=(11.2, 7.0), sharex="col",
        gridspec_kw={"height_ratios": (2.4, 1.0)},
    )
    colors = {"donor": "#4C6EF5", "bridge": "#E8590C", "acceptor": "#2B8A3E"}
    cases = ("diagonal_reference", "noncondon")
    titles = {
        "diagonal_reference": r"(a) diagonal, $V_{DB}/V_{BA}=22/45$",
        "noncondon": r"(b) non-Condon, $V_{DB}/V_{BA}=2/2$",
    }
    display_names = {
        "diagonal_reference": "diagonal reference",
        "weak_diagonal": "weak diagonal control",
        "noncondon": "non-Condon",
    }
    for column, case in enumerate(cases):
        result = validation["results"][case]
        top, residual_axis = axes[:, column]
        for state_index, (state, color) in enumerate(colors.items()):
            top.plot(
                result["t"], result["populations"][:, state_index],
                color=color, lw=1.8, label=state,
            )
            top.plot(
                result["paper_t"][::10],
                result["paper_populations"][::10, state_index],
                "o", ms=3.8, mfc="white", mec=color, mew=1.0,
            )
            residual_axis.plot(
                result["paper_t"], result["residual"][:, state_index],
                color=color, lw=1.3,
            )
        top.set(title=titles[case], ylim=(-0.025, 1.025))
        top.text(
            0.97, 0.58,
            rf"$\tau_{{\rm TN}}={result['fit']['lifetime_ps']:.2f}$ ps"
            "\n"
            rf"$\tau_{{\rm paper}}={result['paper_fit']['lifetime_ps']:.2f}$ ps",
            transform=top.transAxes, ha="right", va="center", fontsize=9,
        )
        residual_axis.axhline(0.0, color="#495057", lw=0.8)
        residual_axis.set(
            xlabel="time (ps)", ylabel="simulation - paper",
            ylim=(-0.015, 0.015),
        )
        for axis in (top, residual_axis):
            axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("population")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper right")
    axes[0, 1].plot([], [], "-", color="#495057", label="tensor network + TTM")
    axes[0, 1].plot(
        [], [], "o", ms=4, mfc="white", mec="#495057",
        label="digitized paper Fig. 2",
    )
    axes[0, 1].legend(frameon=False, loc="upper right")
    figure.tight_layout()
    output = Path(path or IMG / "bridge_electron_transfer.png")
    figure.savefig(output, dpi=160)
    plt.close(figure)

    validation_rows = "\n".join(
        f"| {display_names[case]} | "
        f"{values['lifetime_ps']:.3f} | "
        f"{values['paper_curve_lifetime_ps']:.3f} | "
        f"{values['reported_lifetime_ps']:.2f} | "
        f"{values['population_rmse']:.4f} | "
        f"{values['max_population_error']:.4f} | "
        f"{values['last_transfer_norm']:.2e} |"
        for case, values in validation["summary"].items()
    )
    early_rows = "\n".join(
        f"| {display_names[case]} | "
        f"{values['donor_population_loss']:.3g} | "
        f"{values['peak_bridge_population']:.3g} | "
        f"{values['final_acceptor_population']:.3g} | "
        f"{values['normalization_error']:.2e} |"
        for case, values in early_summary.items()
    )
    diagonal_validation = validation["summary"]["diagonal_reference"]
    noncondon_validation = validation["summary"]["noncondon"]
    _write_summary("bridge_electron_transfer", fr"""## Generated result

### Fifteen-picosecond paper comparison

| coupling model | tensor-network lifetime (ps) | digitized-curve lifetime (ps) | paper label (ps) | all-population RMSE | maximum error | final transfer-tensor norm |
|---|---:|---:|---:|---:|---:|---:|
{validation_rows}

The solid curves in the generated figure are the 15 ps transfer-tensor
propagation; open circles are vector-path data extracted from the paper's Fig. 2.
Residuals compare all three populations at every 0.05 ps digitization point.
The same $A\exp(-t/\tau)+C$ model was fitted independently to the calculated and
digitized donor curves. Its fit to the digitized curves recovers the lifetimes
printed in the paper, so the comparison is not based only on copying those two
labels.

A 0.12 ps transfer kernel predicts the held-out end of the direct trajectory
with maximum population errors of
**{diagonal_validation['heldout_population_error']:.2e}** and
**{noncondon_validation['heldout_population_error']:.2e}**. The reconstructed
short maps preserve trace to **{max(diagonal_validation['direct_map_trace_error'], noncondon_validation['direct_map_trace_error']):.2e}**;
their most negative Choi eigenvalue is
**{min(diagonal_validation['direct_map_minimum_choi_eigenvalue'], noncondon_validation['direct_map_minimum_choi_eigenvalue']):.2e}**, which measures the small non-CP error introduced by independently truncating the tomography runs.

### Direct 0.2 ps documentation run

| coupling model | donor loss | max bridge | final acceptor | normalization error |
|---|---:|---:|---:|---:|
{early_rows}

The docs profile used the paper's $\alpha={suite['bath']['alpha']}$,
$\omega_c={suite['bath']['cutoff_cm']:.0f}\ \mathrm{{cm}}^{{-1}}$ bath and
{suite['bath']['n_modes']} automatically resolved modes. This independently
regenerated short run checks that the current propagation code retains the
published model's early population transfer; the checked-in short dynamical
maps make the longer validation affordable in CI.

Any fitted donor lifetime summarizes the full population trace; it is not an
isolated elementary $D\to A$ rate because bridge occupation and recrossing
remain in the dynamics.
""")
    return output


def two_bath_heat_flow(path=None):
    example = _load_example("two_bath_heat_flow")
    suite = example.run_profile("docs", announce=True)
    summary = example.summarize(suite)
    plt = _mpl()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for condition, style in (("nonequilibrium", "-"), ("equilibrium", "--")):
        case = suite["results"][condition]["primary"]
        result = case["result"]
        axes[0].plot(result.t, result.expect["sz"], style, label=condition)
    case = suite["results"]["nonequilibrium"]["primary"]
    result = case["result"]
    axes[1].plot(result.t, case["hot_to_system"], label=r"hot $\to$ system")
    axes[1].plot(result.t, -case["cold_to_system"], label=r"system $\to$ cold")
    axes[0].set(xlabel=r"time ($\omega_c^{-1}$)", ylabel=r"$\langle\sigma_z\rangle$",
                title="junction dynamics")
    axes[1].set(xlabel=r"time ($\omega_c^{-1}$)", ylabel="energy current",
                title="directional currents")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    output = Path(path or IMG / "two_bath_heat_flow.png")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    nonequilibrium = summary["nonequilibrium"]
    equilibrium = summary["equilibrium"]
    _write_summary("two_bath_heat_flow", f"""## Generated result

In the final 20% of the documentation run, the hot- and cold-bath currents into
the system are **{nonequilibrium['mean_hot_to_system']:.4g}** and
**{nonequilibrium['mean_cold_to_system']:.4g}**. Their residual balance is
**{nonequilibrium['steady_balance_error']:.3g}**, and the RMS continuity-equation
residual is **{nonequilibrium['continuity_rms']:.3g}**.

The equal-temperature control has a final-window hot current of
**{equilibrium['mean_hot_to_system']:.3g}**. A nonzero residual means the finite
run has not established a steady-state transport claim; extend the reference
profile and converge it before interpreting the plateau.
""")
    return output


FIGURES = (
    bath_correlation,
    bath_correlation_finite_t,
    bath_structured,
    vibronic_dimer,
    nonadiabatic_spin_boson,
    bridge_electron_transfer,
    two_bath_heat_flow,
)

OUTPUTS = {
    function.__name__: IMG / f"{function.__name__}.png"
    for function in FIGURES
}
OUTPUTS["bath_correlation_finite_t"] = IMG / "bath_correlation_finiteT.png"


def build_all(force=False):
    """Generate every figure, propagating failures to Sphinx and CI."""
    IMG.mkdir(parents=True, exist_ok=True)
    written = []
    for function in FIGURES:
        target = OUTPUTS[function.__name__]
        summary = GENERATED / f"{function.__name__}.md"
        tutorial = function.__name__ in {
            "vibronic_dimer", "nonadiabatic_spin_boson",
            "bridge_electron_transfer", "two_bath_heat_flow",
        }
        if target.exists() and (not tutorial or summary.exists()) and not force:
            continue
        written.append(function(target))
    return written


if __name__ == "__main__":
    for output in build_all(force=True):
        print("wrote", output)
