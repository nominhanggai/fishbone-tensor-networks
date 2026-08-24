# Vibrationally assisted transfer in a molecular dimer [interaction-chain Trotter MPO]

This tutorial reconstructs two quantized-vibration trajectories from Figure 5
of [Dijkstra *et al.*](https://arxiv.org/abs/1309.4910). It asks how an
underdamped molecular vibration changes excitation transfer between two
molecules whose electronic energies are far from resonance.

The calculation below is self-contained. It constructs the electronic
Hamiltonian and Brownian-oscillator environment, identifies exactly which
molecule is coupled to the bath, propagates to the paper's endpoint, and checks
the full result against samples derived from the published curves. The paper
uses HEOM; this tutorial uses an interaction-chain tensor network. Agreement is
therefore a model-level check, not a claim that the two numerical algorithms
are identical.

## 1. Electronic dimer and environmental coordinate

Restrict the molecule to the one-excitation states $|D\rangle$ and $|A\rangle$.
In units of the electronic coupling $J$,

$$
H_S/J = \begin{pmatrix}8&-1\\-1&0\end{pmatrix}.
$$

The donor is $8J$ above the acceptor. Direct coherent transfer is consequently
off-resonant. The paper couples their energy difference to a Brownian
coordinate but does not print a coupling-operator matrix. We implement the
stated gap fluctuation as

$$
H_{SB}=|D\rangle\langle D|\otimes X.
$$

so that $E_D-E_A\mapsto E_D-E_A+X$. In the code this is the mapping

```python
baths = {0: gap_bath.bind(OCCUPIED)}
```

where key `0` means electronic site 0, the donor. There is no bath entry for
site 1. The difference between the two diagonal coupling eigenvalues is one, so
this operator produces the required gap fluctuation. Re-centering it as
$\sigma_z/2$ also entails a bath displacement and a choice of reorganization
counterterm; the supplement does not state that convention. Two independent
local baths would instead define a different gap-correlation function and
require a corresponding change in spectral-density strength.

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
        baths={0: gap_bath.bind(OCCUPIED)},
    )

    return model.run(
        dt=0.025,
        t_max=20.0,
        method="interaction-chain-fishbone-trotter-mpo",
        trunc_eps=1e-3,
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

for frequency, result in results.items():
    population = np.asarray(result.expect["population"], float)

    # A bare two-level operator is measured on each electronic site, so the
    # two columns are donor and acceptor populations.
    assert population.shape == (len(result.t), 2)
    conservation_error = np.max(np.abs(population.sum(axis=1) - 1.0))
    paper_population = paper[f"omega{int(frequency)}_acceptor"]
    calculated = np.interp(paper["tJ"], result.t, population[:, 1])
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
    color = {4.0: "tab:blue", 8.0: "tab:green"}[frequency]
    plt.plot(
        result.t, population[:, 1], color=color,
        label=fr"fishbonett, $\omega_0={frequency:g}J$",
    )
    plt.plot(
        paper["tJ"], paper_population, "o", ms=4, mfc="none", color=color,
        label=fr"Fig. 5, $\omega_0={frequency:g}J$",
    )

plt.xlabel(r"time ($J^{-1}$)")
plt.ylabel("acceptor population")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

## 3. What the representation does

`Fishbone.from_single_excitation` represents each molecule by a local two-level
site. Its number-conserving hopping term has the supplied $2\times2$ electronic
Hamiltonian as its one-excitation block. `OCCUPIED` therefore serves both as the
donor--bath coupling operator and as the local population observable.

The bath is first discretized in star form by the TEDOPA quadrature. The
calculation then takes the interaction picture with respect to that diagonal
star-bath Hamiltonian and applies the inverse star-to-chain transformation to
the time-dependent couplings. This produces the interaction-chain coefficients
used by the tensor network. That representation is independent of the evolution
algorithm; the final words in the method name select the Trotter-MPO integrator.

At finite temperature, `beta=0.1` produces a signed thermofield spectral
density. Leaving `domain` unset resolves a frequency window for the requested
spectral tolerance. Leaving `n_modes` unset then sizes the chain from the
propagation horizon, rather than from an arbitrary small mode count.

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
Fock dimensions. The documentation calculation instead uses:

| control | documentation run | manual refinement |
|---|---:|---:|
| timestep | `0.025` | `0.0125` |
| local Fock dimension | `12` | `16` |
| SVD threshold | `1e-3` | `5e-4` |
| chain modes | automatic | automatic |
| maximum bond | unlimited | unlimited |

When refining the timestep, also tighten the SVD threshold: a smaller step
performs more truncations over the same physical interval. Judge convergence by
the population trajectory and endpoint, not by probability conservation alone.

## 5. Dynamics and conclusion

![Acceptor dynamics compared point by point with two published Figure 5 curves](../img/vibronic_dimer_dynamics.svg)

```{include} ../_generated/vibronic_dimer.md
```

The resonant $8J$ vibration moves population rapidly and approaches a large
acceptor population, whereas the $4J$ case transfers more slowly. The generated
summary reports errors over 41 points along each published curve, rather than
checking only the final value. The comparison is not identical: the $4J$ curve
is close, but the tensor-network $8J$ trajectory transfers too quickly around
$tJ=6$--10.

The supplement specifies the Brownian density and HEOM controls but not an
explicit coupling matrix or reorganization-counterterm convention. Numerical
checks of the other natural conventions do not improve both curves: a centered
counterterm improves the $8J$ transient but worsens $4J$, while the coupling
equivalent to two independent local baths is worse and substantially more
expensive. The tutorial therefore retains the direct single-gap convention and
does not tune an unspecified convention to the plotted data. A short fixed
chain also develops finite-size artifacts before the benchmark endpoint.

The manual `reference` profile in `examples/vibronic_dimer.py` scans integer
frequencies from $J$ through $10J$ with refined settings. It can be used to
investigate the two maxima in the frequency scan, but the validated comparison
on this page is deliberately limited to the two trajectories shown above.

## 6. Common mistakes

- Adding the same Brownian bath independently to both molecules changes the
  energy-difference correlation strength. The mapping used here is the single
  gap coordinate `{0: gap_bath.bind(OCCUPIED)}`.
- Fixing `n_modes` to 24 or 48 is not safe at $t=20/J$. The resulting return from
  the chain boundary can look like physical population recurrence.
- Starting both local sites in `EXCITED` leaves the one-excitation sector and
  changes the model.
- A finite `bond_dim` can hide discarded weight. Use `trunc_eps` as the primary
  cutoff and leave the maximum bond unlimited unless memory protection is needed.
- Chain-mode and star-mode occupations are observables of different represented
  coordinates and cannot be compared mode by mode.
