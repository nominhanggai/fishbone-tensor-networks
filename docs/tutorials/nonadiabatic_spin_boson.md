# Strong-coupling nonadiabatic spin--boson dynamics

This tutorial calculates the population relaxation of a two-state system in a
hot, strongly coupled bath. It is based on the benchmark of
[Nuomin, Beratan, and Zhang](https://arxiv.org/abs/2111.14308) and demonstrates
how two Hamiltonian representations can be used as an internal numerical
cross-check.

Everything needed to run the calculation and interpret the returned arrays is
included below.

## 1. Benchmark Hamiltonian

$$
H = \Delta\sigma_x
  + \sigma_z\sum_k g_k(a_k+a_k^\dagger)
  + \sum_k\omega_k a_k^\dagger a_k,
$$

with $\Delta=1$ and

$$
J(\omega)=\frac{\eta\omega_c\omega}{\omega_c^2+\omega^2},
\qquad \eta=4,\quad \omega_c=4,\quad T=4.
$$

The system begins in the $+1$ eigenstate of $\sigma_z$. We record

$$
P_\uparrow(t)=\left\langle\frac{I+\sigma_z}{2}\right\rangle.
$$

The large $\eta$ and temperature make this a useful nonperturbative test: a
smooth-looking curve from a weak-coupling approximation is not an adequate
reference.

## 2. Complete runnable comparison

The interaction-chain trajectory covers $t\Delta/\pi\simeq1$. The
Schrödinger-chain TDVP trajectory covers the first quarter of that interval as a
build-time cross-check. Both use exactly the same system Hamiltonian, spectral
density, discretized domain, and initial state.

```python
import numpy as np
import matplotlib.pyplot as plt

from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z


DELTA = 1.0
ETA = 4.0
OMEGA_C = 4.0
TEMPERATURE = 4.0
DT = 0.025


def spectral_density(omega):
    return ETA * OMEGA_C * omega / (OMEGA_C**2 + omega**2)


def make_model():
    bath = Bath(
        J=spectral_density,
        beta=1.0 / TEMPERATURE,
        domain=(-16.0, 80.0),
        n_modes=24,
        phys_dim=6,
        discretization="tedopa",
    )
    return SystemBath(
        h=DELTA * sigma_x,
        coupling=sigma_z,
        bath=bath,
    )


population_up = 0.5 * (np.eye(2) + sigma_z)

# round() is explicit because pi is not an integer multiple of DT.
interaction_steps = int(round(np.pi / DT))
interaction = make_model().run(
    dt=DT,
    n_steps=interaction_steps,
    method="interaction-chain-trotter-mpo",
    trunc_eps=1e-3,
    bond_dim=None,
    initial="up",
    observables={"population_up": population_up},
)

# A shorter independent representation check suitable for a docs build.
schrodinger_steps = int(round(0.25 * np.pi / DT))
schrodinger = make_model().run(
    dt=DT,
    n_steps=schrodinger_steps,
    method="schrodinger-chain-tdvp2",
    trunc_eps=1e-3,
    bond_dim=None,
    initial="up",
    observables={"population_up": population_up},
)

p_interaction = np.asarray(interaction.expect["population_up"], float)
p_schrodinger = np.asarray(schrodinger.expect["population_up"], float)

# Both runs use the same DT, so their first samples refer to the same times.
n_overlap = min(len(p_interaction), len(p_schrodinger))
max_difference = np.max(np.abs(
    p_interaction[:n_overlap] - p_schrodinger[:n_overlap]
))

print("maximum difference on common interval:", max_difference)
print("interaction peak bond:", np.max(interaction.max_bond))
print("Schrodinger peak bond:", np.max(schrodinger.max_bond))

plt.plot(
    interaction.t / np.pi,
    p_interaction,
    label="interaction chain",
)
plt.plot(
    schrodinger.t / np.pi,
    p_schrodinger,
    "--",
    label="Schrodinger chain (overlap check)",
)
plt.xlabel(r"$t\Delta/\pi$")
plt.ylabel(r"$P_\uparrow(t)$")
plt.legend()
plt.tight_layout()
plt.show()
```

## 3. Why construct a fresh model for each run?

`run` does not intentionally mutate the declarative `Bath`, but constructing two
models makes the comparison unambiguous: both simulations start from a new
product state, resolve the same bath specification, and use no tensor state left
over from the other method.

The finite-temperature bath is represented on a signed thermofield frequency
axis. `beta=1/T=0.25` is in the same inverse-energy units as the Hamiltonian. The
TEDOPA discretizer adapts its quadrature to the bath measure instead of applying
uniform-weight Gauss--Legendre quadrature.

The Drude tail decays slowly. An automatic domain covering 99.9% of its
reorganization energy is consequently very wide and produces hundreds of modes
for this horizon. The documentation calculation states its finite window and
mode count rather than hiding that cost. The reference calculation removes this
shortcut and is the one to use for cutoff convergence.

## 4. What differs between the two methods?

Both methods approximate the same physical Hamiltonian, but represent it
differently.

### Interaction-chain Trotter MPO

The free star bath is used to define the interaction picture. The resulting
time-dependent system--bath coupling is transformed from star to chain
coordinates. `trotter-mpo` applies the commuting single-channel bath coupling as
a conditional-displacement MPO. The state therefore does not carry free-bath
evolution as entanglement.

### Schrödinger-chain TDVP2

The chain frequencies and nearest-neighbour hoppings remain explicitly in a
static Hamiltonian MPO. Two-site TDVP evolves a pair of tensors and splits it by
SVD, allowing the bond to grow. This is a genuinely independent representation
and integrator, which makes agreement more meaningful than comparing two output
paths through the same propagator.

The peak bonds printed by the example must not be interpreted as a fair timing
comparison: the TDVP run intentionally ends earlier.

## 5. Result layout and numerical tests

For `SystemBath`, a named observable produces a one-dimensional array:

```python
interaction.t.shape                       # (interaction_steps,)
interaction.expect["population_up"].shape # (interaction_steps,)
interaction.rdm.shape                     # (interaction_steps, 2, 2)
interaction.max_bond.shape                # (interaction_steps,)
```

Check the following independently:

1. **Representation agreement.** Compare only equal times, as the code does.
2. **Time step.** Halve `DT` and compare values at common physical times.
3. **Fock dimension.** Repeat with `phys_dim=10`, 20, and 40.
4. **SVD threshold.** Tighten `trunc_eps` from $10^{-3}$ to $5\times10^{-4}$.
5. **Bath resolution.** Expand the finite domain and mode count, or use the
   automatic reference profile.

Changing all five at once cannot identify the source of a difference.

## 6. Dynamics and conclusion

![Strong-coupling population dynamics and retained bond dimensions](../img/nonadiabatic_spin_boson.png)

```{include} ../_generated/nonadiabatic_spin_boson.md
```

The two representations agree closely during their common interval while the
interaction-chain trajectory resolves the subsequent nonmonotonic relaxation.
That agreement is a strong implementation check, not a complete convergence
study. The `reference` profile in
`examples/nonadiabatic_spin_boson.py` propagates to $5\pi/\Delta$, halves the
step, increases the local Fock space, uses automatic bath resolution, and adds a
Schrödinger-star comparison.

## 7. Common mistakes

- Passing `t_max=5*np.pi` with `dt=0.025` is rejected because the values are not
  commensurate. Pass an explicit rounded `n_steps`, as above.
- Comparing arrays by index is valid here only because both runs use the same
  step and starting time. In general, interpolate onto common physical times.
- A small peak bond does not prove accuracy; local Fock truncation and bath
  discretization can dominate while the bond remains small.
- One-site TDVP uses a fixed-bond manifold and requires an explicit bond cap.
  This tutorial uses two-site TDVP so `trunc_eps` can control adaptive growth.
