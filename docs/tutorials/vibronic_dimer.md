# Vibrationally assisted transfer in a molecular dimer [interaction-chain Trotter MPO]

This tutorial reproduces the quantized-vibration benchmark in Figure 5 of
[Dijkstra *et al.*](https://arxiv.org/abs/1309.4910). It asks how an underdamped
molecular vibration changes excitation transfer between two molecules whose
electronic energies are far from resonance.

The calculation below is self-contained. It constructs the electronic
Hamiltonian and Brownian-oscillator environment, identifies exactly which
molecule is coupled to the bath, propagates to the paper's endpoint, and checks
the result against the published population values.

## 1. Electronic dimer and environmental coordinate

Restrict the molecule to the one-excitation states $|D\rangle$ and $|A\rangle$.
In units of the electronic coupling $J$,

$$
H_S/J = \begin{pmatrix}8&-1\\-1&0\end{pmatrix}.
$$

The donor is $8J$ above the acceptor. Direct coherent transfer is consequently
off-resonant. The quantum calculation in the paper lets one environmental
coordinate modulate their energy difference:

$$
H_{SB}=|D\rangle\langle D|\otimes X.
$$

Only the donor needs an explicit bath because changing its energy relative to
the uncoupled acceptor changes the donor--acceptor gap. In the code this is the
mapping

```python
baths = {0: gap_bath.bind(OCCUPIED)}
```

where key `0` means electronic site 0, the donor. There is no bath entry for
site 1. Attaching an independent bath with the same strength to both sites would
double the spectral power of the fluctuating energy difference and would not be
the quantum model used for Figure 5.

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
paper_endpoints = {4.0: 0.27, 8.0: 0.67}  # approximate values read from Fig. 5

for frequency, result in results.items():
    population = np.asarray(result.expect["population"], float)

    # A bare two-level operator is measured on each electronic site, so the
    # two columns are donor and acceptor populations.
    assert population.shape == (len(result.t), 2)
    conservation_error = np.max(np.abs(population.sum(axis=1) - 1.0))
    endpoint_error = abs(population[-1, 1] - paper_endpoints[frequency])

    print(
        f"omega_0={frequency:g}: "
        f"P_A(20/J)={population[-1, 1]:.6f}, "
        f"paper≈{paper_endpoints[frequency]:.2f}, "
        f"absolute difference={endpoint_error:.3g}, "
        f"probability error={conservation_error:.2e}, "
        f"peak bond={np.max(result.max_bond)}"
    )
    print("resolved bath layout:", result.meta["bath_branches"])
    plt.plot(result.t, population[:, 1], label=fr"$\omega_0={frequency:g}J$")

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
density. Leaving `domain` unset chooses a frequency window containing 99.9% of
the physical reorganization energy. Leaving `n_modes` unset then sizes the chain
from the propagation horizon, rather than from an arbitrary small mode count.

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

The documentation profile uses these practical convergence controls:

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

![Acceptor dynamics and comparison with the published Figure 5 endpoints](../img/vibronic_dimer.svg)

```{include} ../_generated/vibronic_dimer.md
```

The resonant $8J$ vibration moves population rapidly and approaches a large
acceptor population, whereas the $4J$ case transfers more slowly. Both endpoint
populations agree closely with the quantum calculation in Figure 5. A
two-bath model has a different gap-correlation strength, while a short fixed
chain develops finite-size artifacts before the benchmark endpoint.

The manual `reference` profile in `examples/vibronic_dimer.py` scans integer
frequencies from $J$ through $10J$ with the refined settings. That scan is needed
to reproduce the complete two-maximum frequency dependence; the documentation
build evaluates the two characteristic trajectories so that CI remains bounded.

## 6. Common mistakes

- Adding the same Brownian bath to both molecules changes the energy-difference
  correlation strength. Follow the paper's quantum model with the single mapping
  `{0: gap_bath.bind(OCCUPIED)}`.
- Fixing `n_modes` to 24 or 48 is not safe at $t=20/J$. The resulting return from
  the chain boundary can look like physical population recurrence.
- Starting both local sites in `EXCITED` leaves the one-excitation sector and
  changes the model.
- A finite `bond_dim` can hide discarded weight. Use `trunc_eps` as the primary
  cutoff and leave the maximum bond unlimited unless memory protection is needed.
- Chain-mode and star-mode occupations are observables of different represented
  coordinates and cannot be compared mode by mode.
