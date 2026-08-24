# Vibrationally assisted transfer in a molecular dimer [interaction-chain MPO]

This tutorial calculates excitation transfer between two electronically
mismatched molecules and asks whether a damped intramolecular vibration can
assist the transfer. The model follows
[Dijkstra *et al.*](https://arxiv.org/abs/1309.4910).

The page is self-contained: the program below constructs the Hamiltonian,
attaches the baths, propagates the tensor network, checks probability
conservation, and plots the dynamics.

## 1. The physical model

Restrict the electronic system to the one-excitation states
$|D\rangle$ and $|A\rangle$. In units of the electronic coupling $J$,

$$
H_S/J = \begin{pmatrix}8&-1\\-1&0\end{pmatrix}.
$$

The donor is therefore $8J$ above the acceptor and the off-diagonal coupling has
magnitude $J$. Each molecule couples through its local excitation projector to
an independent Brownian-oscillator bath,

$$
J_b(\omega)=\frac{2\lambda\gamma\omega_0^2\omega}
{(\omega_0^2-\omega^2)^2+\gamma^2\omega^2},
\qquad \lambda=0.2J,\quad \gamma=2J/3,
$$

at $T=10J$, or $\beta J=0.1$. The calculation compares
$\omega_0=4J$, near the bath-relaxation scale, with $\omega_0=8J$, near the
electronic energy gap.

The tensor-network model contains two local two-level system sites. The state
$|1\rangle$ means that a molecule is excited and $|0\rangle$ that it is not:

| code object | physical meaning |
|---|---|
| `electronic[i, i]` | excitation energy of molecule `i` |
| `electronic[i, j]` | excitation hopping between molecules |
| `OCCUPIED` | $|1\rangle\langle1|$, both the bath coupling and population operator |
| `initial=[EXCITED, EMPTY]` | donor excited, acceptor empty |
| two separate `Bath` objects | statistically independent local environments |

## 2. Complete runnable calculation

Save this as `vibronic_dimer.py` and run it with Python. It uses the same
documentation profile that generates the figure below.

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


def make_bath(vibration):
    """One finite-temperature Brownian bath."""
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
        n_modes=24,
        phys_dim=8,
        discretization="tedopa",
        # domain=None: choose the signed thermal domain automatically.
    )


def run(vibration):
    # Each Bath instance is independent. bind(OCCUPIED) states which
    # system operator couples to that bath.
    baths = [
        make_bath(vibration).bind(OCCUPIED),
        make_bath(vibration).bind(OCCUPIED),
    ]
    model = Fishbone.from_single_excitation(electronic, baths=baths)

    return model.run(
        dt=0.05,
        t_max=10.0,
        method="interaction-chain-fishbone-trotter-mpo",
        trunc_eps=1e-3,
        bond_dim=None,                 # let the SVD threshold set the bond
        initial=[EXCITED, EMPTY],
        observables={"population": OCCUPIED},
    )


results = {frequency: run(frequency) for frequency in (4.0, 8.0)}

for frequency, result in results.items():
    population = np.asarray(result.expect["population"], float)
    # Bare OCCUPIED was measured on every system site, so columns are D and A.
    assert population.shape == (len(result.t), 2)
    conservation_error = np.max(np.abs(population.sum(axis=1) - 1.0))
    print(
        f"omega_0={frequency:g}: "
        f"P_A(t_final)={population[-1, 1]:.6f}, "
        f"probability error={conservation_error:.2e}, "
        f"peak bond={np.max(result.max_bond)}"
    )
    print("resolved bath layout:", result.meta["bath_branches"])
    plt.plot(result.t, population[:, 1], label=fr"$\omega_0={frequency:g}J$")

plt.xlabel(r"time ($J^{-1}$)")
plt.ylabel("acceptor population")
plt.legend()
plt.tight_layout()
plt.show()
```

## 3. What the API is doing

`Fishbone.from_single_excitation` does not replace the electronic system by one
two-level system. It maps each molecule to a local two-level site and constructs
number-conserving hopping operators whose one-excitation block is exactly the
matrix `electronic`. Consequently, the returned population array has one column
per molecule.

`bind(OCCUPIED)` is equally important. A bath contains a spectral density and
discretization settings; binding says that its collective coordinate modulates
the energy of the occupied molecule. Constructing two baths rather than reusing
one shared multichannel bath makes their fluctuations independent.

The method name separates three choices:

- `interaction-chain`: first form the bath in star coordinates, take the
  interaction picture with respect to that free star bath, and then transform
  the time-dependent coupling from star to chain coordinates;
- `fishbone`: retain the two system sites and their two bath branches as a comb
  tensor-network topology;
- `trotter-mpo`: apply each branch propagator as a conditional-displacement MPO.

This representation is attractive for a hot structured bath because the free
bath evolution does not itself have to be carried as state entanglement.

## 4. Reading and checking the result

`result.expect["population"][n, 0]` and `[n, 1]` are the donor and acceptor
populations after integration step `n`. `result.max_bond[n]` records the largest
bond retained at that sampled time. `result.meta["bath_branches"]` reports the
resolved representation, node layout, number of modes, and local Fock dimension
for each independent bath.

The first scientific check is

$$
P_D(t)+P_A(t)=1.
$$

A failure here indicates a numerical problem or an incorrectly specified
initial state, not interesting transfer physics. Population conservation alone
does not establish convergence: repeat with smaller `dt`, larger `phys_dim`, more
modes, and a tighter `trunc_eps`.

## 5. Dynamics and conclusion

![Acceptor population at two characteristic vibrational frequencies](../img/vibronic_dimer.svg)

```{include} ../_generated/vibronic_dimer.md
```

The generated trajectories show substantially more acceptor population for the
near-resonant $8J$ vibration over this finite interval. That supports
vibrationally assisted transfer in this model, but two frequencies do not define
a resonance curve. The `reference` profile in
`examples/vibronic_dimer.py` scans $2J$ through $10J$, uses a 0.001/$J$ step and
Fock dimension 20, and lets the bath light-cone resolver determine the mode
count. Those convergence calculations are required before locating an optimum.

## 6. Common mistakes

- Starting both local sites in `EXCITED` leaves the one-excitation sector and
  changes the model.
- Passing one shared bath instead of two independent bath instances introduces
  correlated fluctuations.
- Treating `bond_dim` as the accuracy control can conceal discarded weight. Use
  `trunc_eps` as the primary cutoff and leave the cap unlimited unless memory
  protection is required.
- Comparing chain-mode and star-mode occupations as though they were the same
  observable is invalid; they are different represented coordinates.
