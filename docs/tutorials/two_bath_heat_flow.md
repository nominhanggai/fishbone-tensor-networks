# Heat flow through a two-level molecular junction [tree TEBD]

A molecule connected to environments at different temperatures is a minimal
model of nanoscale heat transport. Unlike a one-bath relaxation calculation,
this problem has a genuine nonequilibrium question: after the molecule has
relaxed, does energy continue to enter from the hot bath and leave through the
cold bath?

This tutorial assumes familiarity with open quantum systems, but not with
thermal rectifiers or the internal layout of Fishbone. It constructs the entire
calculation, derives the measured current, plots the dynamics, and explains the
checks needed before interpreting a current as steady state. The model follows
the two-bath spin--boson junction discussed by
[Dunnett and Chin](https://doi.org/10.3390/e23010077).

```{admonition} Orientation
:class: note

- **Level:** advanced; read {doc}`convergence` before making a steady-current
  claim.
- **You will learn:** how two independent baths attach to one system site, how
  to derive a directional energy-current observable, and how to test continuity.
- **Cost:** the four-step `smoke` profile takes seconds. The plotted calculation
  advances the temperature-biased junction and equal-temperature control for
  250 steps each, through $t\omega_c=25$, and commonly takes tens of minutes or
  longer on a CPU.
- **Output:** molecular relaxation, hot and cold currents, and a zero-bias
  control.
```

## 1. Physical model

The junction is a two-level system,

$$
H_S=\frac{\omega_0}{2}\sigma_z,
$$

coupled through $\sigma_x$ to independent hot and cold harmonic baths:

$$
H=H_S+\sum_{b\in\{h,c\}}\left[
\sum_k \omega_{bk}a_{bk}^\dagger a_{bk}
+\sigma_x\sum_k g_{bk}(a_{bk}+a_{bk}^\dagger)
\right].
$$

Both baths have the hard-cutoff Ohmic spectral density

$$
J_b(\omega)=2\pi\alpha\omega\,\Theta(\omega_c-\omega).
$$

We use $\omega_c=1$ as the unit of energy (and therefore $\omega_c^{-1}$
as the unit of time), $\omega_0=0.2$, and $\alpha=0.1$ for each reservoir,
matching the cited benchmark setting. The transport run has
$\beta_h=2$ and $\beta_c=100$. A second run with
$\beta_h=\beta_c=100$ is a zero-temperature-bias control.

The correspondence between the equations and the main code objects is:

| Physical object | Code object |
|---|---|
| $H_S$ | the only entry of `Fishbone(sites=...)` |
| hot bath | bath index 0 attached to system site 0 |
| cold bath | bath index 1 attached to system site 0 |
| $\sigma_x$ | the operator passed to each bath's `bind` method |
| first hot/cold chain coordinate | `BathMode(0, 0, 0)` / `BathMode(0, 1, 0)` |

## 2. Why the model has two baths on one site

The construction

```python
baths={0: [hot, cold]}
```

means "attach both entries in this list to system site 0." The list order fixes
the bath indices used by `BathMode`: hot is bath 0 and cold is bath 1. Each
`Bath` is discretized and chain-mapped independently. They share the system
operator $\sigma_x$, but they do not share oscillator modes or noise.

This is distinct from one bath coupled to two system sites, which would describe
correlated environmental fluctuations. It is also distinct from adding two
spectral densities and making one bath: doing that would discard the identity
of the hot and cold reservoirs and prevent the two currents from being measured
separately.

## 3. Deriving the current observable

After chain mapping, bath $b$ couples to the system through

$$
H_{Sb}=\kappa_b\sigma_x X_b,
\qquad X_b=b_{b0}+b_{b0}^{\dagger},
$$

where $b_{b0}$ is the first mode of that bath's chain. The contribution of this
coupling to the change in molecular energy is

$$
I_{b\to S}
=i\langle[H_{Sb},H_S]\rangle
=\kappa_b\omega_0\langle\sigma_yX_b\rangle.
$$

This fixes both the prefactor and sign convention: a positive value means that
bath $b$ is adding energy to the molecule. The raw
$\langle\sigma_yX_b\rangle$ correlation is therefore not itself an energy
current. The script reads $\kappa_b$ from `result.meta["bath_branches"]` because
it is determined by the numerical chain transformation.

`BathMode(system_site, bath, mode)` names a represented bath coordinate. It is
preferable to an internal tensor index: the latter depends on the chosen state
geometry, whereas the physical address remains stable.

Summing the two bath contributions gives a useful identity,

$$
\frac{d}{dt}\langle H_S\rangle=I_{h\to S}+I_{c\to S}.
$$

The finite-difference residual printed by the script tests the observable
definition and time resolution. It is not expected to be exactly zero because
the derivative and tensor-network evolution are approximate.

## 4. Complete runnable calculation

The following program runs the biased junction and its equal-temperature
control to $t\omega_c=25$ with timestep $0.1/\omega_c$, local Fock dimension 5,
SVD threshold $10^{-3}$, and no maximum bond cap. It measures both bath
currents, checks energy continuity, and writes a dynamics figure. The
convergence section specifies the longer and more tightly resolved variants.

```python
import numpy as np
import matplotlib.pyplot as plt

from fishbonett import Bath, BathMode, Fishbone
from fishbonett.operators import annihilate, sigma_x, sigma_y, sigma_z


# omega_c sets the energy unit; times are consequently in omega_c^{-1}.
OMEGA_C = 1.0
OMEGA_0 = 0.2
ALPHA = 0.1
PHYS_DIM = 5


def spectral_density(omega):
    """Hard-cutoff Ohmic J(omega) for positive physical frequencies."""
    if 0.0 <= omega <= OMEGA_C:
        return 2.0 * np.pi * ALPHA * omega
    return 0.0


def make_bath(beta):
    return Bath(
        J=spectral_density,
        beta=beta,
        # A finite-temperature TEDOPA bath uses a signed effective domain.
        domain=(-OMEGA_C, OMEGA_C),
        n_modes=None,                 # resolve the mode count automatically
        phys_dim=PHYS_DIM,
        discretization="tedopa",
        extra_breaks=(0.0,),          # preserve the change at omega = 0
    )


def run_junction(beta_hot, beta_cold):
    hot = make_bath(beta_hot).bind(sigma_x)
    cold = make_bath(beta_cold).bind(sigma_x)

    # The dictionary key is the system-site index. Its list contains every
    # independent bath attached to that site, in the order hot then cold.
    model = Fishbone(
        sites=[0.5 * OMEGA_0 * sigma_z],
        baths={0: [hot, cold]},
    )

    destroy = annihilate(PHYS_DIM)
    position = destroy + destroy.T
    hot_mode = BathMode(system_site=0, bath=0, mode=0)
    cold_mode = BathMode(system_site=0, bath=1, mode=0)

    result = model.run(
        dt=0.1,
        t_max=25.0,
        method="schrodinger-chain-tree-tebd",
        trunc_eps=1e-3,
        bond_dim=None,
        # sigma_z |1> = -|1>, so this is the molecular ground state.
        initial=[np.array([0.0, 1.0])],
        observables={
            "sz": (sigma_z, 0),
            "hot_sy_x": (
                np.kron(sigma_y, position), (0, hot_mode)
            ),
            "cold_sy_x": (
                np.kron(sigma_y, position), (0, cold_mode)
            ),
        },
    )

    # Chain mapping changes the system--first-mode coupling. Use the resolved
    # coefficients recorded by the run instead of reconstructing them.
    branches = result.meta["bath_branches"]
    kappa_hot = branches[0]["system_coupling"]
    kappa_cold = branches[1]["system_coupling"]
    hot_to_system = (
        kappa_hot * OMEGA_0
        * np.asarray(result.expect["hot_sy_x"], dtype=float)
    )
    cold_to_system = (
        kappa_cold * OMEGA_0
        * np.asarray(result.expect["cold_sy_x"], dtype=float)
    )
    return result, hot_to_system, cold_to_system


# A temperature-biased calculation and an equal-temperature control.
biased, hot_current, cold_current = run_junction(2.0, 100.0)
equal_temperature, control_hot_current, control_cold_current = (
    run_junction(100.0, 100.0)
)

# The molecular energy and its numerical derivative provide an independent
# continuity check: d<E_S>/dt = I_hot->S + I_cold->S.
system_energy = (
    0.5 * OMEGA_0 * np.asarray(biased.expect["sz"], dtype=float)
)
energy_derivative = np.gradient(system_energy, biased.t)
continuity_residual = energy_derivative - hot_current - cold_current
interior = continuity_residual[1:-1]  # omit one-sided derivative endpoints

# "Steady" below means only the final fifth of this finite simulation.
tail = slice(int(0.8 * len(biased.t)), None)
print("observable shape:", biased.expect["sz"].shape)
print("maximum bond dimension:", int(np.max(biased.max_bond)))
print("mean late hot current:", float(np.mean(hot_current[tail])))
print("mean late cold current:", float(np.mean(cold_current[tail])))
print(
    "late current-balance error:",
    float(abs(np.mean((hot_current + cold_current)[tail]))),
)
print("continuity RMS:", float(np.sqrt(np.mean(interior**2))))
print(
    "equal-temperature late net current:",
    float(np.mean((control_hot_current + control_cold_current)[tail])),
)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
axes[0].plot(
    biased.t, biased.expect["sz"], label="temperature-biased run"
)
axes[0].plot(
    equal_temperature.t,
    equal_temperature.expect["sz"],
    "--",
    label="equal-temperature control",
)
axes[1].plot(biased.t, hot_current, label=r"hot $\to$ system")
axes[1].plot(biased.t, -cold_current, label=r"system $\to$ cold")
axes[0].set(
    xlabel=r"time ($\omega_c^{-1}$)",
    ylabel=r"$\langle\sigma_z\rangle$",
    title="junction dynamics",
)
axes[1].set(
    xlabel=r"time ($\omega_c^{-1}$)",
    ylabel="energy current",
    title="directional currents",
)
for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
fig.tight_layout()
fig.savefig("two_bath_heat_flow.svg")
```

The result contains the time grid in `result.t`, one array per named observable
in `result.expect`, the maximum MPS bond dimension after each step in
`result.max_bond`, and representation-specific information in `result.meta`.
Every observable array has the same length as `result.t`.

## 5. Representation and numerical choices

`schrodinger-chain-tree-tebd` has three independent parts:

- `schrodinger-chain` is the Hamiltonian representation. Each star bath is
  transformed to a nearest-neighbour chain and remains in the Schrodinger
  picture.
- `tree` is the tensor-network state geometry. The two bath chains branch from
  their common system site.
- `tebd` is the time integrator.

This representation is convenient here because the current operator acts only
on the system and the first site of one static chain. TEDOPA performs the bath
discretization and chain transformation on the signed finite-temperature domain
$(-\omega_c,\omega_c)$. `extra_breaks=(0.0,)` tells the quadrature about the
change between the positive- and negative-frequency parts. Negative effective
frequencies encode thermal occupation; they are not extra physical modes to be
included in a reorganization-energy integral.

The initial state is factorized: the molecule starts in its ground state and the
represented baths start in their reference vacua. Consequently even the
equal-temperature control can have a short correlation-building transient.
Zero bias predicts vanishing long-time transport, not identically zero current
at every early time.

## 6. Reading the dynamics

![Junction population and hot/cold currents](../img/two_bath_heat_flow.svg)

```{include} ../_generated/two_bath_heat_flow.md
```

The left panel shows molecular relaxation; the right panel shows where its
energy comes from and goes. A credible transport regime requires all of the
following:

1. the molecular energy has reached a plateau;
2. $I_{h\to S}$ and $I_{c\to S}$ are approximately equal and opposite;
3. the continuity residual is small on the scale of either current;
4. the equal-temperature control tends toward zero current; and
5. the apparent plateau occurs before reflections from the finite bath chains.

The final-fifth averages in the script are diagnostics, not an automatic proof
of steady state. If either current is still drifting, extend the time horizon
and the automatically resolved chain length together.

## 7. Convergence study

Use the shared timestep, SVD, Fock-space, and finite-bath workflow in
{doc}`convergence`. For this transport problem, accept a refinement only when
the complete hot and cold current curves, their late-time balance, and the
continuity residual are stable together. Extend `t_max` only with enough
automatically resolved modes to keep the chain recurrence outside the measured
window.

`python examples/two_bath_heat_flow.py --profile reference` advances both
temperature conditions to $t\omega_c=40$ with timestep $0.025/\omega_c$ and
automatically resolved bath chains. It performs six simulations: the primary
pair uses local Fock dimension 6 and SVD threshold $10^{-3}$; a second pair
raises the Fock dimension to 8; and a third pair returns to dimension 6 while
tightening the threshold to $5\times10^{-4}$. All six runs leave the maximum
bond unrestricted. Agreement of both currents and the zero-bias control is more
informative than convergence of $\langle\sigma_z\rangle$ alone.

## 8. Physical conclusion

The example illustrates the defining difference between thermal relaxation and
transport. During relaxation, the net current changes the molecular energy. In
a steady transport regime, the molecular energy can remain nearly constant
while a nonzero current enters from the hot bath and an equal current leaves for
the cold bath. That conclusion comes from resolved system--bath correlations;
it cannot be inferred from a population plateau by itself.

## 9. Common mistakes

- A raw system--mode correlation is not a current until its coupling and energy
  prefactors are included.
- A flat population does not prove transport; both bath currents are needed.
- Equal bath temperatures do not remove the initial factorization transient.
- Reversing the order of `[hot, cold]` also reverses their `BathMode` indices.
- Increasing `t_max` without enough bath modes can turn a finite-chain
  recurrence into a false steady-state signal.
- A small current-balance error alone does not establish convergence if both
  currents change when the time step, Fock dimension, or SVD threshold changes.

This is the most advanced tutorial in the sequence. Return to
{doc}`bridge_electron_transfer` for a single-bath chemical example, or use the
{doc}`tutorial index <index>` to choose another path.
