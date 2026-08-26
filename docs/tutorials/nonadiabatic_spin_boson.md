# Strong-coupling nonadiabatic spin--boson dynamics [interaction-chain Trotter MPO]

This tutorial reconstructs the nonadiabatic spin--boson benchmark in Figure 8
of [Nuomin, Beratan, and Zhang](https://arxiv.org/abs/2111.14308). It is a useful
stress test because the bath is both hot and strongly coupled: convergence is
controlled by the bath discretization, oscillator basis, tensor-network
truncation, and time step, not by the smoothness of the population curve.

The plotted calculation uses a 200-mode TEDOPA chain and covers the first fifth
of the published time axis, through $t\Delta/\pi=1$. The `reference` profile in
`examples/nonadiabatic_spin_boson.py` uses 600 bath modes and extends the
trajectory to the paper's endpoint, $t\Delta/\pi=5$.

```{admonition} Orientation
:class: note

- **Level:** intermediate; read {doc}`vibronic_dimer` first if interaction-chain
  notation is new.
- **You will learn:** how to reproduce strong-coupling spin relaxation and
  separate bath-domain, mode-count, Fock-space, and tensor truncation checks.
- **Cost:** the four-step `smoke` profile takes seconds. The plotted 200-mode
  calculation advances 252 steps; the 600-mode calculation advances 1,257
  steps and can require substantial CPU time and memory.
- **Output:** spin-up population and retained bond dimension.
```

## Model and observable

The zero-bias spin--boson Hamiltonian is

$$
H = \Delta\sigma_x
  + \sigma_z\int h(\omega)(a_\omega+a_\omega^\dagger)\,d\omega
  + \int \omega a_\omega^\dagger a_\omega\,d\omega,
$$

with the Drude spectral density

$$
J(\omega)=h(\omega)^2
 = \frac{\eta\omega_c\omega}{\omega_c^2+\omega^2}.
$$

Taking $\Delta=1$, Figure 8 uses

$$
\eta=4,\qquad \omega_c=4,\qquad T=4.
$$

The initial state is $|\uparrow\rangle$ and the plotted quantity is

$$
P_\uparrow(t)=\left\langle\frac{I+\sigma_z}{2}\right\rangle.
$$

At finite temperature the package uses a thermofield spectral density on a
signed frequency interval. The oscillator state remains a vacuum product state;
thermal absorption is carried by the negative-frequency part of the transformed
bath.

## Numerical settings

For the full Figure 8 calculation, the bond-index plots in Figure 9 show a
600-mode bath chain. Here, “600 modes” means 600 bath sites in the TEDOPA chain;
the two-level system is one additional MPS site. The finite-temperature mapping
discretizes the complete signed interval into those 600 modes, not into 600
positive- plus 600 negative-frequency modes.

The paper also reports oscillator cutoff 10, time step
$\delta t=1.25\times10^{-2}/\Delta$, SVD threshold $10^{-3}$, and a maximum
bond dimension of 1000. The plotted calculation uses 200 bath modes through
$t\Delta/\pi=1$; the `reference` profile uses 600 through $t\Delta/\pi=5$.
Both assign ten oscillator states to every bath mode, use the reported time
step and SVD threshold, and leave the maximum bond unrestricted so the SVD
threshold determines the retained rank.

The paper writes the continuum on a finite interval $[\Omega_0,\Omega_1]$ but
does not report numerical endpoints. The package calculation therefore states
its additional discretization choice explicitly:

```python
domain=(-16.0, 80.0)
```

Increasing the mode count at fixed domain checks the quadrature resolution;
expanding the domain checks the frequency cutoff. These are distinct tests.

## Complete runnable calculation

Run this code from the repository root so it can also load samples digitized
from the vector graphics in the paper's Figure 8.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z


DELTA = 1.0
ETA = 4.0
OMEGA_C = 4.0
TEMPERATURE = 4.0
DT = 0.0125


def spectral_density(omega):
    return ETA * OMEGA_C * omega / (OMEGA_C**2 + omega**2)


bath = Bath(
    J=spectral_density,
    beta=1.0 / TEMPERATURE,
    domain=(-16.0, 80.0),
    n_modes=200,
    phys_dim=10,
    discretization="tedopa",
)

model = SystemBath(
    h=DELTA * sigma_x,
    coupling=sigma_z,
    bath=bath,
)

population_up = 0.5 * (np.eye(2) + sigma_z)
n_steps = int(np.ceil(np.pi / DT))

result = model.run(
    dt=DT,
    n_steps=n_steps,
    method="interaction-chain-trotter-mpo",
    trunc_eps=1e-3,
    bond_dim=None,
    initial="up",
    observables={"population_up": population_up},
)

population = np.asarray(result.expect["population_up"], float)
scaled_time = result.t / np.pi

paper = np.genfromtxt(
    Path("examples/reference_data/nuomin_2022_fig8_ic10.csv"),
    delimiter=",",
    names=True,
)
mask = paper["t_delta_over_pi"] <= scaled_time[-1] + 1e-6
paper_time = paper["t_delta_over_pi"][mask]
paper_population = paper["population_up"][mask]

simulation_at_paper_times = np.interp(
    paper_time,
    np.r_[0.0, scaled_time],
    np.r_[1.0, population],
)
residual = simulation_at_paper_times - paper_population

print("final scaled time:", scaled_time[-1])
print("final population:", population[-1])
print("paper-curve RMSE:", np.sqrt(np.mean(residual**2)))
print("maximum paper-curve error:", np.max(np.abs(residual)))
print("peak bond dimension:", np.max(result.max_bond))

figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.plot(scaled_time, population, label="fishbonett")
left.plot(
    paper_time,
    paper_population,
    "o",
    markerfacecolor="none",
    label="digitized paper Fig. 8 IC10",
)
left.set(xlabel=r"$t\Delta/\pi$", ylabel=r"$P_\uparrow(t)$")

right.plot(scaled_time, result.max_bond)
right.set(xlabel=r"$t\Delta/\pi$", ylabel="retained bond dimension")
handles, labels = left.get_legend_handles_labels()
figure.legend(
    handles, labels, frameon=False, loc="upper center", ncol=2,
    bbox_to_anchor=(0.5, 1.0),
)
figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
plt.show()
```

The CSV contains samples digitized from the converged IC10 curve in the arXiv figure;
it is not raw numerical data supplied by the authors. Comparing the complete
curve is more informative than matching a single endpoint.

## What the method represents

The bath is discretized in star modes first. The interaction picture is taken
with respect to the diagonal free star bath, after which the time-dependent
couplings are transformed from star to chain coordinates. At $t=0$ the coupling
is localized at the first chain mode and then travels outward.

`interaction-chain` names this Hamiltonian representation.
`trotter-mpo` names the integrator: for one Hermitian coupling operator, the
mode-coupling terms commute and their conditional-displacement propagator has a
compact MPO form.

The paper applies the same interaction-chain Hamiltonian with a swap-gate TEBD
scheme. This tutorial instead uses the package's Trotter MPO, so the population
comparison tests the represented dynamics; it is not a reproduction of the
paper's timing or bond-dimension comparison between algorithms.

## Result and interpretation

![Spin population compared with Figure 8 and the retained bond dimension](../img/nonadiabatic_spin_boson.svg)

```{include} ../_generated/nonadiabatic_spin_boson.md
```

The population falls rapidly from one and then relaxes more slowly toward the
unbiased value $1/2$. The agreement with the published curve is quantitative on
the displayed interval. A 24-mode calculation can still show close agreement
between two representations of that same finite bath while deviating
substantially from Figure 8, so the paper comparison is an independent check.

## Common mistakes and benchmark-specific convergence

Do not infer continuum convergence from agreement between two representations
of the same short finite bath. Do not compare their peak bonds unless the time
horizon, timestep, local basis, and truncation rule are also identical. Use the
coupled timestep/SVD workflow in {doc}`convergence`, then add the checks specific
to this benchmark:

1. raise `phys_dim` to test the oscillator basis;
2. raise `n_modes` at fixed `domain` to test quadrature resolution;
3. expand `domain` while increasing `n_modes` to maintain resolution;
4. compare the full population trajectory with the digitized Figure 8 curve,
   not only normalization or peak bond.

The longer calculation is selected explicitly:

```bash
python examples/nonadiabatic_spin_boson.py --profile reference
```

The `reference` profile discretizes `domain=(-16, 80)` into 600 TEDOPA bath
modes and advances 1,257 steps of size $0.0125/\Delta$ to
$t\Delta/\pi=5$. Each bath mode has local dimension 10, the SVD threshold is
$10^{-3}$, and no maximum bond dimension is imposed. Compare its complete
population curve with the 200-mode result before drawing conclusions about the
published time interval.

For a chemically structured environment with the same interaction-chain
representation, return to {doc}`vibronic_dimer`. To see how short reduced
dynamics can be extended beyond the direct tensor-network window, continue to
{doc}`bridge_electron_transfer`.
