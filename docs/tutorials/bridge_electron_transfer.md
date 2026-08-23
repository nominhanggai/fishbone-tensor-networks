# Donor--bridge--acceptor electron transfer

This tutorial propagates electron transfer through a three-state molecular
bridge. It compares ordinary diagonal energy-gap fluctuations with a
non-Condon bath operator that also modulates the electronic couplings. The model
comes from [Acharyya, Ovcharenko, and Fingerhut](https://doi.org/10.1063/5.0027976)
([preprint](https://arxiv.org/abs/2108.11175)).

The complete program below includes the unit conversion, both Hamiltonians, the
correlated bath, the initial state, population observables, diagnostics, and
plotting.

## 1. Diabatic states and two coupling models

Use the ordered basis $\{|D\rangle,|B\rangle,|A\rangle\}$: donor, bridge, and
acceptor. The common diabatic energies are

$$
(E_D,E_B,E_A)=(0,-150,-1000)\ {\rm cm}^{-1}.
$$

The two cases differ as follows:

| parameter (cm$^{-1}$ where applicable) | Condon | non-Condon |
|---|---:|---:|
| $V_{DB}$ | 22 | 2 |
| $V_{BA}$ | 45 | 2 |
| $V_{DA}$ | 0 | 0 |
| diagonal bath operator | $\operatorname{diag}(2,1,0)$ | same |
| off-diagonal $M_{DB}$ | 0 | 0.17 |
| off-diagonal $M_{BA}$ | 0 | 0.055 |

There is one bath, not three independent baths. Its collective coordinate
couples through the full matrix $M$, so diagonal and off-diagonal fluctuations
are correlated.

The zero-temperature density before thermofield thermalization is

$$
J(\omega)=\frac{\alpha\pi}{2}\omega e^{-\omega/\omega_c},
\qquad \alpha=10.02,\quad\omega_c=100\ {\rm cm}^{-1}.
$$

Its reorganization energy is

$$
\lambda=\frac{1}{\pi}\int_0^\infty\frac{J(\omega)}{\omega}\,d\omega
=\frac{\alpha\omega_c}{2}=501\ {\rm cm}^{-1}.
$$

Only positive physical frequencies enter this reorganization-energy integral;
negative thermofield frequencies encode finite-temperature occupation and must
not be counted as additional physical modes.

## 2. Unit conversion

The package uses $\hbar=1$, so a time measured in ps requires angular
frequencies in rad ps$^{-1}$. The conversion is

$$
1\ {\rm cm}^{-1}=2\pi c\times10^{-12}=0.1883651567\ {\rm rad\ ps}^{-1}.
$$

If $\omega'=q\omega$ with $q=0.1883651567$, the discrete definition
$J(\omega)=\pi\sum_k g_k^2\delta(\omega-\omega_k)$ implies

$$
J'(\omega')=qJ(\omega'/q).
$$

The Hamiltonian, spectral density, and inverse temperature must all use the same
conversion. Converting only the system Hamiltonian changes the physical model.

## 3. Complete runnable transient calculation

This documentation calculation propagates the first 0.2 ps. It is long enough
to exercise the real strongly coupled Hamiltonian and resolve early bridge
population, but intentionally too short to estimate the published 2--3 ps
donor lifetime.

```python
import numpy as np
import matplotlib.pyplot as plt

from fishbonett import Bath, SystemBath


CM_TO_RAD_PS = 2.0 * np.pi * 2.99792458e10 * 1e-12
KB_CM_PER_K = 0.6950348009
TEMPERATURE_K = 300.0

P_D = np.diag([1.0, 0.0, 0.0])
P_B = np.diag([0.0, 1.0, 0.0])
P_A = np.diag([0.0, 0.0, 1.0])
OBSERVABLES = {"donor": P_D, "bridge": P_B, "acceptor": P_A}


def system_matrices(case):
    """Return H_S in rad/ps and dimensionless bath operator M."""
    h_cm = np.diag([0.0, -150.0, -1000.0])
    coupling = np.diag([2.0, 1.0, 0.0])

    if case == "condon":
        h_cm[0, 1] = h_cm[1, 0] = 22.0
        h_cm[1, 2] = h_cm[2, 1] = 45.0
    elif case == "noncondon":
        h_cm[0, 1] = h_cm[1, 0] = 2.0
        h_cm[1, 2] = h_cm[2, 1] = 2.0
        coupling[0, 1] = coupling[1, 0] = 0.17
        coupling[1, 2] = coupling[2, 1] = 0.055
    else:
        raise ValueError("case must be 'condon' or 'noncondon'")

    return CM_TO_RAD_PS * h_cm, coupling


def spectral_density(omega_rad_ps):
    """J in rad/ps, transformed from the published cm^-1 expression."""
    omega_cm = omega_rad_ps / CM_TO_RAD_PS
    alpha = 10.02
    cutoff_cm = 100.0
    j_cm = (
        0.5 * alpha * np.pi * omega_cm
        * np.exp(-omega_cm / cutoff_cm)
    )
    return CM_TO_RAD_PS * j_cm


def make_model(case):
    hamiltonian, coupling = system_matrices(case)
    kbt_rad_ps = KB_CM_PER_K * TEMPERATURE_K * CM_TO_RAD_PS
    bath = Bath(
        J=spectral_density,
        beta=1.0 / kbt_rad_ps,
        n_modes=12,
        phys_dim=6,
        discretization="tedopa",
        # The signed thermal frequency domain is selected automatically.
    )
    return SystemBath(h=hamiltonian, coupling=coupling, bath=bath)


def run(case):
    return make_model(case).run(
        dt=0.005,             # ps
        t_max=0.2,            # ps: early-time docs profile
        method="interaction-chain-trotter-mpo",
        trunc_eps=1e-3,
        bond_dim=None,
        initial=np.array([1.0, 0.0, 0.0]),
        observables=OBSERVABLES,
    )


results = {case: run(case) for case in ("condon", "noncondon")}

figure, axes = plt.subplots(1, 2, sharey=True)
for axis, (case, result) in zip(axes, results.items()):
    populations = {
        name: np.asarray(result.expect[name], float)
        for name in OBSERVABLES
    }
    total = populations["donor"] + populations["bridge"] + populations["acceptor"]
    normalization_error = np.max(np.abs(total - 1.0))

    print(case)
    print("  normalization error:", normalization_error)
    print("  maximum bridge population:", np.max(populations["bridge"]))
    print("  final acceptor population:", populations["acceptor"][-1])
    print("  peak bond:", np.max(result.max_bond))

    for name, values in populations.items():
        axis.plot(result.t, values, label=name)
    axis.set_title(case)
    axis.set_xlabel("time (ps)")

axes[0].set_ylabel("population")
axes[0].legend()
figure.tight_layout()
plt.show()
```

## 4. How the code represents correlated fluctuations

`SystemBath` treats the donor, bridge, and acceptor as one three-level system.
Its single coupling matrix `coupling` multiplies one collective bath coordinate.
In the non-Condon case that matrix has both diagonal and off-diagonal entries, so
the same bath fluctuation that shifts diabatic energies also changes electronic
couplings.

This is different from constructing three independent baths bound to three
operators. Independent baths have zero cross-correlation and would implement a
different stochastic model.

The interaction-chain method first discretizes the thermal star bath, uses the
free star Hamiltonian to define the interaction picture, and then transforms the
time-dependent coupling into chain coordinates. The system remains a single
three-level tensor site; the chain contains only the represented bath modes.

## 5. Time-step reasoning

The largest stated electronic energy difference is 1000 cm$^{-1}$, or about
188.4 rad ps$^{-1}$. Its oscillation period is approximately

$$
2\pi/188.4\simeq0.033\ {\rm ps}.
$$

The documentation step of 0.005 ps gives about seven steps per period. This is a
reasonable transient check, not the final time-step convergence test. The
reference profile uses 0.001 ps.

## 6. Population, flux, and rate are different quantities

The projectors return $P_D(t)$, $P_B(t)$, and $P_A(t)$. Their derivatives are net
fluxes. For example,

$$
-\dot P_D=k_{D\to B}P_D-k_{B\to D}P_B+\cdots.
$$

Therefore neither $-\dot P_D$ nor an electronic coherence is automatically the
elementary forward rate. Back transfer and bridge recrossing must be separated by
a kinetic model or an appropriate forward-flux correlation calculation.

For comparison with a reported effective donor lifetime, fit only a long,
converged interval that is approximately single exponential:

```python
def effective_lifetime(result):
    donor = np.asarray(result.expect["donor"], float)
    # Exclude the initial inertial transient and the low-population noisy tail.
    mask = (result.t > 0.1) & (donor > 0.15) & (donor < 0.9)
    if np.count_nonzero(mask) < 3:
        raise ValueError("trajectory is too short for a lifetime fit")
    slope, intercept = np.polyfit(result.t[mask], np.log(donor[mask]), 1)
    if slope >= 0:
        raise ValueError("selected donor population is not decaying")
    return -1.0 / slope
```

This lifetime still summarizes the whole donor--bridge--acceptor dynamics; it is
not an isolated microscopic $k_{D\to A}$.

## 7. Dynamics and conclusion

![Early donor, bridge, and acceptor populations](../img/bridge_electron_transfer.png)

```{include} ../_generated/bridge_electron_transfer.md
```

The early-time result shows that the non-Condon bath operator creates appreciable
bridge response even though its bare electronic couplings are much smaller. The
0.2 ps figure cannot establish the published donor lifetimes and deliberately
reports them as unresolved.

The `reference` profile in `examples/bridge_electron_transfer.py` propagates to
10 ps with a 0.001 ps step, automatic bath modes, Fock dimensions 20 and 40, and
SVD thresholds $10^{-3}$ and $5\times10^{-4}$. Only after those comparisons are
stable should the approximately 2.36 ps and 2.50 ps literature lifetimes be used
as quantitative validation targets.

## 8. Common mistakes

- Using cm$^{-1}$ Hamiltonian entries directly with a ps time step changes every
  dynamical time scale.
- Transforming $H_S$ but not $J(\omega)$ and $\beta$ is an inconsistent unit
  conversion.
- Adding negative thermofield frequencies to the physical reorganization-energy
  integral double-counts temperature rather than molecular reorganization.
- Calling a short-time population derivative “the forward rate” ignores backward
  and bridge-mediated fluxes.
- Treating the 0.2 ps docs profile as converged kinetics confuses an executable
  tutorial with the much more expensive reference calculation.
