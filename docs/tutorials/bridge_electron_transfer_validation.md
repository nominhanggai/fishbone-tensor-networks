# Donor--bridge--acceptor validation [transfer tensor method]

This appendix contains the reproducibility material behind the long-time figure
in {doc}`bridge_electron_transfer`. It assumes the model, unit conversion, and
QUAPI-to-explicit-bath Hamiltonian conversion introduced there.

```{admonition} Orientation
:class: note

- **Level:** advanced validation and publication workflow.
- **You will learn:** how nine physical initial states reconstruct a qutrit
  dynamical map, how transfer tensors extend it, and how retained memory is
  tested.
- **Cost:** propagation from the included maps takes seconds; regenerating the
  maps performs 18 tensor-network simulations and is a production calculation.
- **Output:** 15 ps populations, paper residuals, fitted donor lifetimes, and
  memory-convergence diagnostics.
```

## From 0.15 ps direct dynamics to 15 ps

A direct 15 ps interaction-chain calculation resolved with TEDOPA's automatic
light-cone criterion requires nearly one thousand modes and develops large MPS
bonds. The bath
memory reported for Fig. 2 is only about 0.10--0.12 ps, so it is more efficient
to calculate the complete reduced dynamical map through 0.15 ps and then use
the transfer-tensor method (TTM).

For a three-state system the map has $3^2=9$ columns. The example propagates the
three basis states and the real and imaginary superpositions

$$
|r_{ij}\rangle=\frac{|i\rangle+|j\rangle}{\sqrt 2},\qquad
|q_{ij}\rangle=\frac{|i\rangle+i|j\rangle}{\sqrt 2},
$$

for every pair $i<j$. These physical pure states reconstruct
$\mathcal E_t(|i\rangle\langle j|)$. Propagating only the donor initial state
would not determine a dynamical map and would not support a valid TTM
extrapolation.

The inexpensive half of the reference calculation is completely reproducible
from the stored short maps:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from fishbonett.rates import predict_density_mat, transfer_mat


data = np.load(
    Path("examples/reference_data")
    / "bridge_electron_transfer_ttm_maps.npz"
)
dt = float(data["dt_ps"])
steps = round(15.0 / dt)
rho0 = np.diag([1.0, 0.0, 0.0]).astype(complex)
times = np.arange(1, steps + 1, dtype=float) * dt
trajectories = {}

for case in ("diagonal_reference", "noncondon"):
    maps = data[f"{case}_maps"]
    transfer_tensors, transfer_norm = transfer_mat(maps)

    # The directly simulated donor trajectory seeds one full memory window.
    direct = np.einsum(
        "tij,j->ti", maps, rho0.reshape(9)
    ).reshape(-1, 3, 3)
    rdm = predict_density_mat(steps, transfer_tensors, direct)
    population = np.diagonal(rdm, axis1=1, axis2=2).real
    trajectories[case] = population

    print(case)
    print("  final populations:", population[-1])
    print("  final transfer-tensor norm:", transfer_norm[-1])

# Load the vector-path samples used for the pointwise paper comparison.
paper = np.genfromtxt(
    Path("examples/reference_data")
    / "acharyya_2021_fig2_populations.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)


def lifetime(time, donor):
    """Fit P_D(t) = A exp(-t/tau) + C and return tau."""
    def model(t, amplitude, tau, offset):
        return amplitude * np.exp(-t / tau) + offset

    parameters, _ = curve_fit(
        model,
        time,
        donor,
        p0=(0.95, 2.5, 0.01),
        bounds=([0.0, 0.01, -0.2], [2.0, 100.0, 0.2]),
    )
    return parameters[1]


colors = {
    "donor": "#4C6EF5",
    "bridge": "#E8590C",
    "acceptor": "#2B8A3E",
}
figure, axes = plt.subplots(
    2,
    2,
    figsize=(11.2, 7.0),
    sharex="col",
    gridspec_kw={"height_ratios": (2.4, 1.0)},
)
state_handles = []

for column, case in enumerate(trajectories):
    selected = paper[paper["case"] == case]
    paper_t = np.asarray(selected["time_ps"], float)
    paper_population = np.column_stack(
        [selected[name] for name in colors]
    ).astype(float)
    population = trajectories[case]
    simulation_at_paper_t = np.column_stack([
        np.interp(
            paper_t,
            np.r_[0.0, times],
            np.r_[rho0[state, state].real, population[:, state]],
        )
        for state in range(3)
    ])
    residual = simulation_at_paper_t - paper_population
    top, bottom = axes[:, column]

    for state, (name, color) in enumerate(colors.items()):
        state_line, = top.plot(
            times, population[:, state], color=color, label=name
        )
        if column == 0:
            state_handles.append(state_line)
        top.plot(
            paper_t[::10],
            paper_population[::10, state],
            "o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=color,
        )
        bottom.plot(paper_t, residual[:, state], color=color)

    tau_simulation = lifetime(times, population[:, 0])
    tau_paper = lifetime(paper_t, paper_population[:, 0])
    top.text(
        0.97,
        0.58,
        fr"$\tau_{{\rm TN}}={tau_simulation:.2f}$ ps" "\n"
        fr"$\tau_{{\rm paper}}={tau_paper:.2f}$ ps",
        transform=top.transAxes,
        horizontalalignment="right",
    )
    top.set_title(
        "(a) diagonal, $V_{DB}/V_{BA}=22/45$"
        if case == "diagonal_reference"
        else "(b) non-Condon, $V_{DB}/V_{BA}=2/2$"
    )
    top.set_ylim(-0.025, 1.025)
    bottom.axhline(0.0, color="#495057", linewidth=0.8)
    bottom.set(
        xlabel="time (ps)",
        ylabel="simulation - paper",
        ylim=(-0.015, 0.015),
    )
    for axis in (top, bottom):
        axis.grid(alpha=0.25)

axes[0, 0].set_ylabel("population")
method_handle, = axes[0, 1].plot(
    [], [], "-", color="#495057", label="tensor network + TTM"
)
paper_handle, = axes[0, 1].plot(
    [], [], "o", markerfacecolor="white", markeredgecolor="#495057",
    label="digitized paper Fig. 2",
)
figure.legend(
    [*state_handles, method_handle, paper_handle],
    [
        "donor",
        "bridge",
        "acceptor",
        "tensor network + TTM",
        "digitized paper Fig. 2",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=5,
    frameon=False,
)
figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
plt.show()
```

To regenerate those maps instead of loading them, run the expensive tomography
profile explicitly:

```bash
python examples/bridge_electron_transfer.py \
  --generate-reference-maps examples/output/dba_ttm_maps.npz
```

That command performs 18 tensor-network simulations: nine initial states for
each of the two coupling models. It uses `dt=0.002 ps`, a 0.15 ps direct window,
95 automatically resolved TEDOPA modes, local Fock dimension 6, SVD threshold
$10^{-4}$, and no maximum bond cap. Documentation builds load the resulting
short maps but redo the TTM propagation, fitting, residual calculation, and
figure generation.

The donor population is fitted to

$$
P_D(t)=A\exp(-t/\tau)+C.
$$

The small $C$ accounts for the nonzero equilibrium donor population. Applying
this same fit to the digitized paper curves gives 2.362 ps and 2.481 ps, which
recovers the paper's printed 2.36 ps and 2.50 ps within the precision of the
plot. The tensor-network trajectories give 2.420 ps and 2.549 ps. These are
descriptive lifetimes of the complete donor--bridge--acceptor dynamics, not
elementary $k_{D\to A}$ values: bridge occupation, back transfer, and recrossing
are folded into them.

## Numerical evidence and remaining convergence checks

![Decay of the transfer-tensor kernel and convergence of the propagated populations and donor lifetime with retained memory](../img/bridge_electron_transfer_memory.svg)

The shaded interval is the paper's typical 0.10--0.12 ps memory range for this
spectral density. Panel (a) plots the Frobenius norm of every transfer tensor,
not merely its final value. Panel (b) repeats the complete 15 ps propagation
after truncating the kernel at each indicated memory and compares it with the
0.15 ps result. The zero-error 0.15 ps point is omitted from the logarithmic
axis because it is the reference by definition. Panel (c) applies the same
lifetime fit at every cutoff. Together, the panels distinguish a small-looking
kernel tail from its accumulated effect on the long-time observable.

The following checks have been performed for this validation:

- At $10^{-4}$ SVD threshold, changing the step from 2 fs to 3.33 fs changes
  any population by at most 0.0044 in the diagonal calculation and 0.0051 in
  the non-Condon calculation over the first 0.2 ps. A 4 fs step increases these
  changes to 0.0078 and 0.0089 and retains larger bonds, so it is not the
  preferred reference step.
- At the looser $10^{-3}$ threshold, increasing the Fock dimension from 6 to 10
  changed early populations by less than $5.6\times10^{-4}$. This is useful
  evidence but is not a substitute for repeating the check at $10^{-4}$.
- The final transfer-tensor norms after 0.15 ps are about
  $1.4\times10^{-4}$ and $1.5\times10^{-4}$. Holding out the end of the direct
  map showed that a 0.12 ps kernel predicts the remaining direct trajectory to
  better than $10^{-4}$ in the non-Condon case and about $1.5\times10^{-5}$ in
  the diagonal case.
- Over the complete 15 ps continuation, retaining 0.12 ps instead of 0.15 ps
  changes any population by about 0.0011 and 0.0020, and changes the fitted
  lifetimes by 0.008 ps and 0.016 ps. Kernel truncation therefore contributes
  to the residual but does not explain the full 0.010--0.011 maximum difference
  from the digitized paper curves.
- The reconstructed dynamical maps preserve trace to $3.4\times10^{-16}$. Their
  most negative Choi eigenvalue is $-2.4\times10^{-5}$, a small non-CP error
  from independently truncating the tomography trajectories that should also
  decrease in the tighter-threshold publication check.
- The 15 ps propagated density matrices preserve trace to $1.1\times10^{-11}$
  and remain positive to numerical precision.

For a final publication benchmark, also regenerate the complete nine-column
maps at Fock dimension 10, tighten the SVD threshold to $5\times10^{-5}$, and
repeat at a smaller timestep. Follow the coupled timestep/SVD procedure in
{doc}`convergence`. The full population residuals, not only the fitted
lifetimes, should remain stable under each refinement.

Return to {doc}`bridge_electron_transfer` for the physical interpretation and
the concise tutorial workflow.
