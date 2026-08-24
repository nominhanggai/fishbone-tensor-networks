"""Generate tutorial figures and numerical summaries from public examples.

``docs/conf.py`` invokes ``build_all``; ``build_selected`` can generate a named
subset before Sphinx starts.
"""

from argparse import ArgumentParser
import importlib.util
from pathlib import Path
import sys

import numpy as np


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / "src"))
IMG = DOCS / "img"
GENERATED = DOCS / "_generated"
T_MAX = 4.0
_TS = np.linspace(0.0, T_MAX, 400)
_REFERENCE_INPUTS = {
    "vibronic_dimer": ("dijkstra_2015_fig5_quantum_dynamics.csv",),
    "nonadiabatic_spin_boson": ("nuomin_2022_fig8_ic10.csv",),
    "bridge_electron_transfer": (
        "bridge_electron_transfer_ttm_maps.npz",
        "acharyya_2021_fig2_populations.csv",
    ),
}


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


def _input_mtime(name):
    """Newest source timestamp that can affect a generated figure."""
    inputs = [Path(__file__)]
    example = ROOT / "examples" / f"{name}.py"
    if example.exists():
        inputs.append(example)
    reference_data = ROOT / "examples" / "reference_data"
    inputs.extend(
        reference_data / filename
        for filename in _REFERENCE_INPUTS.get(name, ())
    )
    inputs.extend((ROOT / "src" / "fishbonett").rglob("*.py"))
    return max(path.stat().st_mtime_ns for path in inputs if path.is_file())


def _outputs_are_current(name, outputs):
    """Return whether every generated output is at least as new as its inputs."""
    return min(path.stat().st_mtime_ns for path in outputs) >= _input_mtime(name)


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
    """Zero-temperature Ohmic bath and two coarser comparison grids."""
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
    output = Path(path or IMG / "bath_correlation.svg")
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
    output = Path(path or IMG / "bath_correlation_finiteT.svg")
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
    output = Path(path or IMG / "bath_structured.svg")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def vibronic_dimer(path=None):
    example = _load_example("vibronic_dimer")
    suite = example.run_profile("docs", announce=True)
    summary = example.summarize(suite)
    paper = example.load_paper_figure5()
    plt = _mpl()
    figure, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.4), sharex=True, sharey=True,
    )
    for axis, (vibration, result) in zip(axes, suite["results"].items()):
        population = np.asarray(result.expect["population"])
        column = f"omega{int(vibration)}_acceptor"
        axis.plot(
            np.concatenate(([0.0], result.t)),
            np.concatenate(([0.0], population[:, 1])), lw=2.0,
            color="#4C6EF5", label=example.SIMULATION_LABEL,
        )
        axis.plot(
            paper["tJ"], paper[column], "o", ms=4.2, mfc="none",
            color="#E8590C", label=example.PAPER_LABEL,
        )
        axis.set(
            xlabel=r"time ($J^{-1}$)",
            title=rf"$\omega_0={vibration:g}J$",
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("acceptor population")
    for axis in axes:
        axis.set_ylim(-0.015, 0.74)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output = Path(path or IMG / "vibronic_dimer_centered_gap.svg")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    modes = ", ".join(
        rf"$\omega_0={frequency:g}J$: {values[0]} modes"
        for frequency, values in summary["resolved_modes"].items()
    )
    comparisons = ", ".join(
        (
            rf"$\omega_0={frequency:g}J$: RMSE "
            f"{summary['paper_curve_rmse'][frequency]:.4f}, maximum error "
            f"{summary['paper_curve_max_error'][frequency]:.4f}"
        )
        for frequency in summary["paper_curve_rmse"]
    )
    _write_summary("vibronic_dimer", f"""## Numerical result

The two-trajectory calculation uses timestep $0.025/J$, local Fock dimension
12, SVD threshold $10^{{-4}}$, and no maximum bond cap. It conserved the
one-excitation population to **{summary['normalization_error']:.2e}** and
retained a peak bond dimension of **{summary['max_bond']}**. Comparison with 41
samples derived from each published Figure 5 curve gives {comparisons}.

The comparison uses the centered energy-gap operator and reports the full-curve
error, not a selected endpoint. The tensor-network and HEOM calculations use
different numerical representations, so the remaining difference is
interpreted only after timestep, local-dimension, and SVD convergence checks.

Both the signed thermal frequency domain and the interaction-chain light cone
were resolved automatically. The resulting TEDOPA layouts were {modes}. This
prevents the finite-chain reflections that contaminate the endpoint when only a
few dozen modes are used.
""")
    return output


def nonadiabatic_spin_boson(path=None):
    example = _load_example("nonadiabatic_spin_boson")
    suite = example.run_profile("docs", announce=True)
    summary = example.summarize(suite)
    plt = _mpl()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    result = suite["results"]["interaction"]
    paper = example.load_paper_figure8()
    paper_mask = paper["t_delta_over_pi"] <= result.t[-1] / np.pi + 1e-6
    left.plot(
        result.t / np.pi, result.expect["population_up"],
        color="#4C6EF5", lw=2.0, label=example.SIMULATION_LABEL,
    )
    left.plot(
        paper["t_delta_over_pi"][paper_mask],
        paper["population_up"][paper_mask],
        "o", ms=4.5, mfc="none", color="#E8590C",
        label=example.PAPER_LABEL,
    )
    right.plot(
        result.t / np.pi, result.max_bond,
        color="#2B8A3E", lw=1.8,
    )
    left.set(xlabel=r"$t\Delta/\pi$", ylabel=r"$P_\uparrow(t)$",
             title="strong-coupling relaxation")
    right.set(xlabel=r"$t\Delta/\pi$", ylabel="retained bond dimension",
              title="adaptive bond growth")
    left.legend(frameon=False)
    for axis in (left, right):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output = Path(path or IMG / "nonadiabatic_spin_boson.svg")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    _write_summary("nonadiabatic_spin_boson", rf"""## Numerical result

The 200-mode calculation gives
$P_\uparrow={summary['final_population_up']:.4f}$ at
$t\Delta/\pi={result.t[-1] / np.pi:.4f}$. Against
**{summary['paper_points_compared']}** vector-path samples of the published IC10
curve on this interval, the population RMSE is
**{summary['paper_curve_rmse']:.4f}** and the maximum absolute difference is
**{summary['paper_curve_max_error']:.4f}**. The peak retained bond dimension is
**{summary['max_bond']['interaction']}**.

This trajectory advances 252 steps to $t\Delta/\pi=1$. The `reference` profile
uses the 600-mode chain displayed in the paper's bond-index plot and advances
1,257 steps to $t\Delta/\pi=5$. Both calculations assign ten oscillator states
to each bath mode, use timestep $0.0125/\Delta$, SVD threshold $10^{{-3}}$, and
no maximum bond cap. The finite frequency window is stated separately in the
tutorial because the paper does not report its numerical endpoints.
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
    state_handles = []
    for column, case in enumerate(cases):
        result = validation["results"][case]
        top, residual_axis = axes[:, column]
        for state_index, (state, color) in enumerate(colors.items()):
            state_line, = top.plot(
                result["t"], result["populations"][:, state_index],
                color=color, lw=1.8, label=state,
            )
            if column == 0:
                state_handles.append(state_line)
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
    method_handle, = axes[0, 1].plot(
        [], [], "-", color="#495057", label=example.SIMULATION_LABEL,
    )
    paper_handle, = axes[0, 1].plot(
        [], [], "o", ms=4, mfc="white", mec="#495057",
        label=example.PAPER_LABEL,
    )
    figure.legend(
        [*state_handles, method_handle, paper_handle],
        ["donor", "bridge", "acceptor", example.SIMULATION_LABEL,
         example.PAPER_LABEL],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output = Path(path or IMG / "bridge_electron_transfer.svg")
    figure.savefig(output, dpi=160)
    plt.close(figure)

    memory_figure, memory_axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    case_colors = {
        "diagonal_reference": "#4C6EF5",
        "noncondon": "#E8590C",
    }
    case_handles = []
    dt_ps = validation["metadata"]["dt_ps"]
    for case in cases:
        result = validation["results"][case]
        color = case_colors[case]
        lag = np.arange(1, len(result["transfer_norm"]) + 1) * dt_ps
        case_line, = memory_axes[0].semilogy(
            lag, result["transfer_norm"], color=color, lw=1.8,
            label=display_names[case],
        )
        case_handles.append(case_line)
        convergence = result["memory_convergence"]
        nonzero = convergence["max_population_difference"] > 0.0
        memory_axes[1].semilogy(
            convergence["cutoff_ps"][nonzero],
            convergence["max_population_difference"][nonzero],
            "o-", color=color, ms=4, lw=1.5,
        )
        memory_axes[2].plot(
            convergence["cutoff_ps"],
            convergence["donor_lifetime_ps"],
            "o-", color=color, ms=4, lw=1.5,
        )
    memory_band = memory_axes[0].axvspan(
        0.10, 0.12, color="#ADB5BD", alpha=0.28,
        label="paper's 0.10--0.12 ps memory range",
    )
    for axis in memory_axes[1:]:
        axis.axvspan(0.10, 0.12, color="#ADB5BD", alpha=0.28)
    memory_axes[0].set(
        xlabel="kernel lag (ps)", ylabel=r"transfer-tensor norm $\|T_n\|_F$",
        title="(a) kernel-tail decay",
    )
    memory_axes[1].set(
        xlabel="retained memory (ps)",
        ylabel="maximum population difference",
        title="(b) change from the 0.15 ps kernel",
    )
    memory_axes[2].set(
        xlabel="retained memory (ps)", ylabel="fitted donor lifetime (ps)",
        title="(c) long-time observable convergence",
    )
    for axis in memory_axes:
        axis.grid(alpha=0.25)
    memory_figure.legend(
        [*case_handles, memory_band],
        ["diagonal reference", "non-Condon",
         "paper's 0.10--0.12 ps memory range"],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
        frameon=False,
    )
    memory_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    memory_output = output.with_name("bridge_electron_transfer_memory.svg")
    memory_figure.savefig(memory_output, dpi=160)
    plt.close(memory_figure)

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
    convergence_at_012 = {}
    for case in cases:
        convergence = validation["results"][case]["memory_convergence"]
        index = int(np.argmin(np.abs(convergence["cutoff_ps"] - 0.12)))
        convergence_at_012[case] = {
            "population_difference": convergence[
                "max_population_difference"
            ][index],
            "lifetime_ps": convergence["donor_lifetime_ps"][index],
        }
    _write_summary("bridge_electron_transfer", fr"""## Numerical result

### Fifteen-picosecond paper comparison

| coupling model | tensor-network lifetime (ps) | digitized-curve lifetime (ps) | paper label (ps) | all-population RMSE | maximum error | final transfer-tensor norm |
|---|---:|---:|---:|---:|---:|---:|
{validation_rows}

The solid curves are the 15 ps transfer-tensor propagation; open circles are
vector-path data extracted from the paper's Fig. 2.
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

Retaining 0.12 ps of the complete 0.15 ps kernel changes the 15 ps populations
by at most
**{convergence_at_012['diagonal_reference']['population_difference']:.2e}**
and **{convergence_at_012['noncondon']['population_difference']:.2e}**. The
corresponding donor lifetimes are
**{convergence_at_012['diagonal_reference']['lifetime_ps']:.3f} ps** and
**{convergence_at_012['noncondon']['lifetime_ps']:.3f} ps**.

### Direct 0.2 ps tensor-network propagation

| coupling model | donor loss | max bridge | final acceptor | normalization error |
|---|---:|---:|---:|---:|
{early_rows}

The direct propagation uses the paper's $\alpha={suite['bath']['alpha']}$,
$\omega_c={suite['bath']['cutoff_cm']:.0f}\ \mathrm{{cm}}^{{-1}}$ bath and
{suite['bath']['n_modes']} automatically resolved modes. This 0.2 ps trajectory
independently confirms the early population transfer. The separate 0.15 ps
dynamical maps initialize the 15 ps TTM propagation tabulated above.

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
    conditions = (("temperature_biased", "-"), ("equal_temperature", "--"))
    for condition, style in conditions:
        case = suite["results"][condition]["primary"]
        result = case["result"]
        axes[0].plot(
            result.t, result.expect["sz"], style,
            label=example.CONDITION_LABELS[condition],
        )
    case = suite["results"]["temperature_biased"]["primary"]
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
    output = Path(path or IMG / "two_bath_heat_flow.svg")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    temperature_biased = summary["temperature_biased"]
    equal_temperature = summary["equal_temperature"]
    _write_summary("two_bath_heat_flow", f"""## Numerical result

The calculation uses the cited Ohmic coupling $\alpha={example.ALPHA:g}$ for
each reservoir. Over $20 \leq t\omega_c \leq 25$, the hot- and cold-bath
currents into the system are
**{temperature_biased['mean_hot_to_system']:.4g}** and
**{temperature_biased['mean_cold_to_system']:.4g}**. Their residual balance is
**{temperature_biased['steady_balance_error']:.3g}**, and the RMS continuity-equation
residual is **{temperature_biased['continuity_rms']:.3g}**.

The equal-temperature control has a mean hot current of
**{equal_temperature['mean_hot_to_system']:.3g}** over the same interval. A
nonzero residual means this trajectory has not established a steady-state
transport claim. The `reference` profile extends both temperature conditions to
$t\omega_c=40$ with timestep $0.025/\omega_c$ and independently refines the
Fock dimension and SVD threshold.
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
    function.__name__: IMG / f"{function.__name__}.svg"
    for function in FIGURES
}
OUTPUTS["bath_correlation_finite_t"] = IMG / "bath_correlation_finiteT.svg"
OUTPUTS["vibronic_dimer"] = IMG / "vibronic_dimer_centered_gap.svg"
EXTRA_OUTPUTS = {
    "bridge_electron_transfer": (IMG / "bridge_electron_transfer_memory.svg",),
}
_FIGURE_BY_NAME = {function.__name__: function for function in FIGURES}
_TUTORIAL_FIGURES = {
    "vibronic_dimer", "nonadiabatic_spin_boson",
    "bridge_electron_transfer", "two_bath_heat_flow",
}


def build_selected(names, force=False):
    """Generate selected named figures and their numerical summaries."""
    IMG.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        try:
            function = _FIGURE_BY_NAME[name]
        except KeyError as exc:
            available = ", ".join(sorted(_FIGURE_BY_NAME))
            raise ValueError(
                f"unknown documentation figure {name!r}; available: {available}"
            ) from exc
        target = OUTPUTS[name]
        targets = (target, *EXTRA_OUTPUTS.get(name, ()))
        summary = GENERATED / f"{name}.md"
        tutorial = name in _TUTORIAL_FIGURES
        outputs = (*targets, *((summary,) if tutorial else ()))
        if (all(item.exists() for item in outputs)
                and _outputs_are_current(name, outputs) and not force):
            continue
        written.append(function(target))
    return written


def build_all(force=False):
    """Generate every missing or stale figure and propagate failures."""
    return build_selected(_FIGURE_BY_NAME, force=force)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", action="append", choices=sorted(_FIGURE_BY_NAME),
        help="generate one named figure; repeat for multiple figures",
    )
    args = parser.parse_args(argv)
    names = args.figure or list(_FIGURE_BY_NAME)
    for output in build_selected(names, force=True):
        print("wrote", output)


if __name__ == "__main__":
    main()
