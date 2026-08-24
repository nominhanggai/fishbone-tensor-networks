# Vibrationally assisted transfer in a molecular dimer [interaction-chain Trotter MPO]

This tutorial reconstructs two quantized-vibration trajectories from Figure 5
of [Dijkstra *et al.*](https://arxiv.org/abs/1309.4910). It asks how an
underdamped molecular vibration changes excitation transfer between two
molecules whose electronic energies are far from resonance.

```{admonition} Orientation
:class: note

- **Level:** first tutorial; assumes a general open-quantum-systems background.
- **You will learn:** how a structured vibration is attached to one molecular
  site, propagated, and compared with a published population curve.
- **Cost:** the four-step `smoke` profile takes seconds. The plotted calculation
  propagates two automatically resolved bath chains for 800 steps each and
  commonly takes tens of minutes or longer on a CPU.
- **Output:** donor and acceptor populations through $tJ=20$.
```

The program constructs the electronic Hamiltonian and Brownian-oscillator
environment, identifies exactly which molecule is coupled to the bath,
propagates to the paper's endpoint, and checks the full result against samples
derived from the published curves. The paper uses HEOM; this tutorial uses an
interaction-chain tensor network. Agreement is therefore a model-level check,
not a claim that the two numerical algorithms are identical.

## 1. Electronic dimer and environmental coordinate

Restrict the molecule to the one-excitation states $|D\rangle$ and $|A\rangle$.
In units of the electronic coupling $J$,

$$
H_S/J = \begin{pmatrix}8&-1\\-1&0\end{pmatrix}.
$$

The donor is $8J$ above the acceptor. Direct coherent transfer is consequently
off-resonant. The paper's quantum calculation couples the molecular energy
difference to a Brownian coordinate. In the one-excitation basis this is the
centered operator

$$
H_{SB}=\frac{|D\rangle\langle D|-|A\rangle\langle A|}{2}\otimes X.
$$

Its two eigenvalues differ by one, so
$E_D-E_A\mapsto E_D-E_A+X$. In the local two-level representation used below,
the same operator is attached to the donor site:

```python
baths = {0: gap_bath.bind(GAP_OPERATOR)}
```

where key `0` means electronic site 0, the donor. There is no bath entry for
site 1. Replacing the centered operator by the donor projector is not an
innocent rewrite for the factorized thermal initial state: its identity part
displaces the bath and changes the transient dynamics. An independent HEOM
cross-check with the paper's hierarchy depth, Matsubara count, and timestep
selects the centered convention; it reproduces both digitized curves to about
$10^{-3}$ RMS error. Two independent local baths would instead define a
different gap-correlation function and require a corresponding change in
spectral-density strength.

The bath has the Brownian-oscillator spectral density

$$
J_b(\omega)=\frac{2\lambda\gamma\omega_0^2\omega}
{(\omega_0^2-\omega^2)^2+\gamma^2\omega^2},
\qquad
\lambda=0.2J,\quad \gamma=2J/3,
$$

at $T=10J$, or $\beta J=0.1$. We compare $\omega_0=4J$, near the
critical-damping feature, with $\omega_0=8J$, resonant with the bare electronic
gap.

## 2. Complete runnable calculation

Save the following program as `vibronic_dimer.py`. The automatic mode count is
intentional: a chain containing only a few dozen modes reflects bath excitations
back to the system before $t=20/J$ and gives the wrong long-time population.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fishbonett import Bath, Fishbone
from fishbonett.spectral_densities import brownian


# Local electronic basis: |0> = unexcited, |1> = excited.
EMPTY = np.array([1.0, 0.0])
EXCITED = np.array([0.0, 1.0])
OCCUPIED = np.diag([0.0, 1.0])
GAP_OPERATOR = OCCUPIED - 0.5 * np.eye(2)

# H_S in the one-excitation basis {|D>, |A>}, in units of J.
electronic = np.array([
    [8.0, -1.0],
    [-1.0, 0.0],
])


def make_gap_bath(vibration):
    """Brownian environment that fluctuates the donor-acceptor gap."""
    def spectral_density(omega):
        return brownian(
            omega,
            lam=0.2,          # reorganization energy / J
            gam=2.0 / 3.0,   # damping / J
            w0=vibration,     # vibrational frequency / J
        )

    return Bath(
        J=spectral_density,
        beta=0.1,
        n_modes=None,        # resolve the interaction-chain light cone
        phys_dim=12,
        discretization="tedopa",
        # domain=None also resolves the signed thermal frequency window.
    )


def run(vibration):
    gap_bath = make_gap_bath(vibration)

    # Site 0 is the donor. Site 1 has no bath in this benchmark.
    model = Fishbone.from_single_excitation(
        electronic,
        baths={0: gap_bath.bind(GAP_OPERATOR)},
    )

    return model.run(
        dt=0.025,
        t_max=20.0,
        method="interaction-chain-fishbone-trotter-mpo",
        trunc_eps=1e-4,
        bond_dim=None,       # let the SVD threshold control the bond
        initial=[EXCITED, EMPTY],
        observables={"population": OCCUPIED},
    )


results = {frequency: run(frequency) for frequency in (4.0, 8.0)}
paper = np.genfromtxt(
    Path("examples/reference_data/dijkstra_2015_fig5_quantum_dynamics.csv"),
    delimiter=",",
    names=True,
)

figure, axes = plt.subplots(
    1, 2, figsize=(11.2, 4.4), sharex=True, sharey=True
)

for axis, (frequency, result) in zip(axes, results.items()):
    population = np.asarray(result.expect["population"], float)

    # A bare two-level operator is measured on each electronic site, so the
    # two columns are donor and acceptor populations.
    assert population.shape == (len(result.t), 2)
    conservation_error = np.max(np.abs(population.sum(axis=1) - 1.0))
    paper_population = paper[f"omega{int(frequency)}_acceptor"]
    times = np.concatenate(([0.0], result.t))
    acceptor = np.concatenate(([0.0], population[:, 1]))
    calculated = np.interp(paper["tJ"], times, acceptor)
    difference = calculated - paper_population

    print(
        f"omega_0={frequency:g}: "
        f"P_A(20/J)={population[-1, 1]:.6f}, "
        f"curve RMSE={np.sqrt(np.mean(difference**2)):.4f}, "
        f"maximum error={np.max(np.abs(difference)):.4f}, "
        f"probability error={conservation_error:.2e}, "
        f"peak bond={np.max(result.max_bond)}"
    )
    print("resolved bath layout:", result.meta["bath_branches"])
    axis.plot(
        times,
        acceptor,
        linewidth=2.0,
        color="#4C6EF5",
        label="interaction-chain tensor network",
    )
    axis.plot(
        paper["tJ"],
        paper_population,
        "o",
        markersize=4.2,
        markerfacecolor="none",
        color="#E8590C",
        label="Fig. 5 (vector-path samples)",
    )
    axis.set(
        xlabel=r"time ($J^{-1}$)",
        title=fr"$\omega_0={frequency:g}J$",
        ylim=(-0.015, 0.74),
    )
    axis.grid(alpha=0.25)

axes[0].set_ylabel("acceptor population")
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.01),
)
figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
plt.show()
```

## 3. What the representation does

`Fishbone.from_single_excitation` represents each molecule by a local two-level
site. Its number-conserving hopping term has the supplied $2\times2$ electronic
Hamiltonian as its one-excitation block. `GAP_OPERATOR` couples the donor site
to the Brownian coordinate, while `OCCUPIED` remains the population observable.

The bath is first discretized in star form by the TEDOPA quadrature. The
calculation then takes the interaction picture with respect to that diagonal
star-bath Hamiltonian and applies the star-to-chain transformation to the
time-dependent couplings. Equivalently, this is the inverse of diagonalizing a
finite chain into star modes. It produces the interaction-chain coefficients
used by the tensor network. The final words in the method name independently
select the Trotter-MPO integrator.

At finite temperature, `beta=0.1` produces a signed thermofield spectral
density. Leaving `domain` unset resolves a frequency window for the requested
spectral tolerance. Leaving `n_modes` unset then sizes the chain from the
propagation horizon instead of imposing an arbitrary small mode count.

## 4. Reading and validating the result

`result.expect["population"][n, 0]` and `[n, 1]` are $P_D$ and $P_A$ at recorded
time `n`. The first invariant is

$$
P_D(t)+P_A(t)=1.
$$

This detects a broken initial state or propagation error, but it does not prove
bath, Fock-space, timestep, or SVD convergence. The `bath_branches` metadata
shows that there is exactly one branch, attached to `system_site: 0`, together
with its automatically selected mode count and local Fock dimension.

The paper reports one Matsubara frequency, hierarchy depth 6, and HEOM timestep
$0.001/J$. Those are HEOM controls and do not translate directly into bond or
Fock dimensions. The tensor-network calculation plotted below uses:

| control | plotted calculation | refinement calculation |
|---|---:|---:|
| timestep | `0.025` | `0.0125` |
| local Fock dimension | `12` | `16` |
| SVD threshold | `1e-4` | `5e-5` |
| chain modes | automatic | automatic |
| maximum bond | unlimited | unlimited |

Follow the coupled timestep/SVD procedure in {doc}`convergence`. For this
800-step benchmark, `1e-3` is useful for a quick exploratory run but changes the
full population curve visibly; it is not the control used in the figure. In
addition, inspect `bath_branches` to confirm that automatic resolution postpones
the chain boundary beyond $tJ=20$.

## 5. Dynamics and conclusion

![Centered-gap tensor-network dynamics compared point by point with two published Figure 5 curves](../img/vibronic_dimer_centered_gap.svg)

```{include} ../_generated/vibronic_dimer.md
```

The resonant $8J$ vibration moves population rapidly and approaches a large
acceptor population, whereas the $4J$ case transfers more slowly. The numerical
summary reports errors at 41 times along each published curve. Those pointwise
errors and the peak bond dimension diagnose the plotted calculation more
strongly than agreement at one endpoint.

The centered gap operator is fixed by an independent HEOM reconstruction of
the published calculation; it is not fitted to the plotted data. The
tensor-network controls must then be converged against that same Hamiltonian.
A donor projector can appear closer for one curve while solving a different
factorized-initial-state problem. A short fixed chain also develops finite-size
artifacts before the benchmark endpoint.

The `reference` profile in `examples/vibronic_dimer.py` scans the ten integer
frequencies from $J$ through $10J$. It propagates each automatically resolved
chain to $tJ=20$ with timestep $0.0125/J$, local Fock dimension 16, SVD
threshold $5\times10^{-5}$, and no maximum bond cap. The plotted pointwise
comparison uses only the $4J$ and $8J$ trajectories for which Figure 5 provides
population curves.

## 6. Common mistakes

- Adding the same Brownian bath independently to both molecules changes the
  energy-difference correlation strength. The mapping used here is the single
  centered gap coordinate `{0: gap_bath.bind(GAP_OPERATOR)}`.
- Replacing `GAP_OPERATOR` with the donor projector `OCCUPIED` adds an
  identity-coupled bath force. For the factorized thermal initial state, that
  changes the physical transient; it is not merely an energy-zero shift.
- Fixing `n_modes` to 24 or 48 is not safe at $t=20/J$. The resulting return from
  the chain boundary can look like physical population recurrence.
- Starting both local sites in `EXCITED` leaves the one-excitation sector and
  changes the model.
- A finite `bond_dim` can hide discarded weight. Use `trunc_eps` as the primary
  cutoff and leave the maximum bond unlimited unless memory protection is needed.
- Chain-mode and star-mode occupations are observables of different represented
  coordinates and cannot be compared mode by mode.

Next, {doc}`nonadiabatic_spin_boson` applies the same representation to a hot,
strongly coupled bath where discretization and local Fock convergence are more
demanding.
