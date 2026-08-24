# Strong-coupling nonadiabatic spin--boson dynamics [interaction-chain Trotter MPO]

This tutorial reconstructs the nonadiabatic spin--boson benchmark in Figure 8
of [Nuomin, Beratan, and Zhang](https://arxiv.org/abs/2111.14308). It is a useful
stress test because the bath is both hot and strongly coupled: convergence is
controlled by the bath discretization, oscillator basis, tensor-network
truncation, and time step, not by the smoothness of the population curve.

The documentation calculation covers the first fifth of the published time
axis, through $t\Delta/\pi=1$. The `reference` profile in
`examples/nonadiabatic_spin_boson.py` extends the same calculation to the
paper's full endpoint, $t\Delta/\pi=5$.

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
600-mode bath chain. The paper also reports oscillator cutoff 10, time step
$\delta t=1.25\times10^{-2}/\Delta$, SVD threshold $10^{-3}$, and a maximum
bond dimension of 1000. The shortened calculation on this page uses 200 modes
through $t\Delta/\pi=1$; the manual full-horizon profile uses 600. Both use ten
oscillator states per mode, the reported time step, and the reported SVD
threshold. The maximum bond is left unrestricted so the threshold, rather than
an artificial ceiling, determines the retained rank.

The paper writes the continuum on a finite interval $[\Omega_0,\Omega_1]$ but
does not report numerical endpoints. The package calculation therefore states
its additional discretization choice explicitly:

```python
domain=(-16.0, 80.0)
```

Increasing the mode count at fixed domain checks the quadrature resolution;
expanding the domain checks the frequency cutoff. These are distinct tests.

## Complete runnable calculation

Run this code from the repository root so it can also load the vector-path
samples extracted from the paper's Figure 8.

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
    label="Figure 8 IC10",
)
left.set(xlabel=r"$t\Delta/\pi$", ylabel=r"$P_\uparrow(t)$")
left.legend()

right.plot(scaled_time, result.max_bond)
right.set(xlabel=r"$t\Delta/\pi$", ylabel="retained bond dimension")
figure.tight_layout()
plt.show()
```

The CSV contains samples of the converged IC10 vector path in the arXiv figure;
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

## Common mistakes and convergence checks

Do not infer continuum convergence from agreement between two representations
of the same short finite bath. Do not compare their peak bonds unless the time
horizon, timestep, local basis, and truncation rule are also identical. When
refining this calculation, change one control at a time:

1. raise `phys_dim` to test the oscillator basis;
2. raise `n_modes` at fixed `domain` to test quadrature resolution;
3. expand `domain` while increasing `n_modes` to maintain resolution;
4. halve `DT` and tighten `trunc_eps` together, because more steps also mean
   more SVD truncations; and
5. compare the full population trajectory, not only normalization or peak bond.

The manual command

```bash
python examples/nonadiabatic_spin_boson.py --profile reference
```

uses the 600-mode paper-length controls through $t\Delta/\pi=5$. It is kept out of
the documentation build because the longer tensor-network propagation should
not determine routine documentation build time.
