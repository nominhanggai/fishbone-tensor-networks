# Donor--bridge--acceptor electron transfer

This tutorial tests how fluctuations of electronic couplings can accelerate
electron transfer through a molecular bridge. It reproduces the model behind
Fig. 2 of [Acharyya, Ovcharenko, and
Fingerhut](https://doi.org/10.1063/5.0027976)
([preprint](https://arxiv.org/abs/2108.11175)) with a tensor-network bath.

The complete program below first performs an automatically resolved 0.2 ps run,
which is short enough to execute while learning the API. The page then builds a
15 ps transfer-tensor trajectory from a numerically checked 0.15 ps dynamical
map and compares every donor, bridge, and acceptor population with curves
digitized from the paper's Fig. 2. This separates an approachable calculation
from the expensive reference-map generation without replacing the numerical
validation by a quoted lifetime.

## 1. Physical model

Use the diabatic basis $\{|D\rangle,|B\rangle,|A\rangle\}$ for donor, bridge,
and acceptor. The paper writes the Hamiltonian as

$$
H=H_S+H_B+M\sum_j c_jx_j,
$$

with

$$
H_S=
\begin{pmatrix}
0 & V_{DB} & 0\\
V_{DB} & -150 & V_{BA}\\
0 & V_{BA} & -1000
\end{pmatrix}\mathrm{cm}^{-1},
\qquad
M=\begin{pmatrix}
2 & C_{12} & 0\\
C_{12} & 1 & C_{23}\\
0 & C_{23} & 0
\end{pmatrix}.
$$

One collective bath coordinate multiplies the entire matrix $M$. Consequently,
the site-energy and coupling fluctuations are correlated; this is not a model of
three independent local baths.

There is an important propagation-convention detail. The paper diagonalizes the
coupling operator,

$$
U^\dagger M U=D,
$$

and performs the MACGIC-QUAPI calculation in that basis. The standard Makri
influence coefficients used by that calculation include the local Hamiltonian
renormalization $\lambda_R D^2$. `SystemBath`, by contrast, propagates an
explicit unshifted harmonic bath and does not insert such a model-dependent
renormalization. The Hamiltonian passed to `SystemBath` must therefore be

$$
H_S^{\mathrm{explicit}}=H_S+\lambda_R U D^2U^\dagger
                       =H_S+\lambda_R M^2.
$$

This is a conversion between the paper's QUAPI convention and an explicit-bath
Hamiltonian. The $\lambda_RM^2$ term is not printed in Eq. (1) or in the
supplement. Omitting the conversion instead reproduces the unrenormalized
bilinear Hamiltonian, not the published QUAPI propagation.

For non-Condon coupling, $M^2$ has off-diagonal entries. Keeping them follows
directly from rotating $\lambda_RD^2$ back to the diabatic basis. Keeping only
the diagonal of $M^2$ would not be invariant under the basis transformation used
in the paper and would define a different model.

Fig. 2 requires three calculations:

| calculation | $V_{DB}$ | $V_{BA}$ | $C_{12}$ | $C_{23}$ | purpose |
|---|---:|---:|---:|---:|---|
| diagonal reference | 22 | 45 | 0 | 0 | ordinary sequential transfer, $\tau\simeq2.36$ ps |
| weak diagonal control | 2 | 2 | 0 | 0 | fixed-Hamiltonian control, about 100 ps |
| non-Condon | 2 | 2 | 0.17 | 0.055 | bath-modulated transfer, $\tau\simeq2.50$ ps |

All dimensional entries in the table are in cm$^{-1}$. Comparing only the first
and third rows shows that small bare electronic couplings can be compensated by
non-Condon fluctuations. The second row is essential: it isolates the effect of
$C_{12}$ and $C_{23}$ without changing $H_S$.

## 2. The paper's bath

The positive-frequency spectral density is

$$
J(\omega)=\frac{\alpha\pi}{2}\omega e^{-\omega/\omega_c},
\qquad \alpha=1.67,\qquad \omega_c=600\ \mathrm{cm}^{-1},
$$

at $T=300$ K. Its reorganization energy is

$$
\lambda_R=\frac{1}{\pi}\int_0^\infty
\frac{J(\omega)}{\omega}\,d\omega
=\frac{\alpha\omega_c}{2}=501\ \mathrm{cm}^{-1}.
$$

The individual values of $\alpha$ and $\omega_c$ matter. For example,
$\alpha=10.02$ and $\omega_c=100$ cm$^{-1}$ give the same reorganization
energy but a cutoff timescale six times longer. That is a different bath and
does not reproduce Fig. 2.

Only positive physical frequencies enter $\lambda_R$. `Bath` extends the
density to a signed thermofield domain internally to represent finite
temperature; those negative effective frequencies are not additional physical
reorganization energy.

## 3. Unit conversion

Fishbone uses $\hbar=1$. With time in ps, Hamiltonian entries must therefore be
angular frequencies in rad ps$^{-1}$:

$$
q=2\pi c\times10^{-12}
=0.1883651567\ \frac{\mathrm{rad\ ps}^{-1}}{\mathrm{cm}^{-1}}.
$$

Under $\omega'=q\omega$, the Hamiltonian and spectral density transform as

$$
H'_S=qH_S,
\qquad J'(\omega')=qJ(\omega'/q),
\qquad \beta'=\frac{1}{qk_BT}.
$$

The factor multiplying $J$ is necessary: a spectral density has one power of
energy in the package convention. Converting $H_S$ but not $J$ and $\beta$
would change the physical model.

## 4. Complete runnable early-time calculation

The following program uses the paper parameters, constructs all three controls,
asks the TEDOPA light-cone resolver for the bath mode count, checks probability
conservation, and plots the first 0.2 ps.

```python
import numpy as np
import matplotlib.pyplot as plt

from fishbonett import Bath, SystemBath


CM_TO_RAD_PS = 2.0 * np.pi * 2.99792458e10 * 1e-12
KB_CM_PER_K = 0.6950348009
TEMPERATURE_K = 300.0
BATH_ALPHA = 1.67
BATH_CUTOFF_CM = 600.0
REORGANIZATION_CM = 0.5 * BATH_ALPHA * BATH_CUTOFF_CM

DT_PS = 0.002
T_MAX_PS = 0.2
PHYS_DIM = 6

P_D = np.diag([1.0, 0.0, 0.0])
P_B = np.diag([0.0, 1.0, 0.0])
P_A = np.diag([0.0, 0.0, 1.0])
OBSERVABLES = {"donor": P_D, "bridge": P_B, "acceptor": P_A}
COLORS = {"donor": "#4C6EF5", "bridge": "#E8590C", "acceptor": "#2B8A3E"}


def quapi_equivalent_matrices(case):
    """Return the explicit-bath form of the paper's QUAPI model."""
    h_cm = np.diag([0.0, -150.0, -1000.0])
    coupling = np.diag([2.0, 1.0, 0.0])

    if case == "diagonal_reference":
        h_cm[0, 1] = h_cm[1, 0] = 22.0
        h_cm[1, 2] = h_cm[2, 1] = 45.0
    elif case in {"weak_diagonal", "noncondon"}:
        h_cm[0, 1] = h_cm[1, 0] = 2.0
        h_cm[1, 2] = h_cm[2, 1] = 2.0
        if case == "noncondon":
            coupling[0, 1] = coupling[1, 0] = 0.17
            coupling[1, 2] = coupling[2, 1] = 0.055
    else:
        raise ValueError("unknown calculation")

    # Standard Makri influence coefficients include lambda_R D^2 after the
    # paper diagonalizes M.  Rotating back gives lambda_R M^2.
    h_explicit_cm = (
        h_cm + REORGANIZATION_CM * (coupling @ coupling)
    )
    return CM_TO_RAD_PS * h_explicit_cm, coupling


def spectral_density(omega_rad_ps):
    """Paper's J, transformed from cm^-1 to rad/ps."""
    omega_cm = omega_rad_ps / CM_TO_RAD_PS
    density_cm = (
        0.5 * BATH_ALPHA * np.pi * omega_cm
        * np.exp(-omega_cm / BATH_CUTOFF_CM)
    )
    return CM_TO_RAD_PS * density_cm


beta = 1.0 / (
    KB_CM_PER_K * TEMPERATURE_K * CM_TO_RAD_PS
)

# Resolve the domain and light-cone mode count once. They depend on the bath and
# propagation horizon, but not on which of the three system matrices is used.
resolved = Bath(
    J=spectral_density,
    beta=beta,
    n_modes=None,
    phys_dim=1,                 # placeholder; resolution does not depend on it
    discretization="tedopa",
).resolved(T_MAX_PS)

print("resolved modes:", resolved.n_modes)
print(
    "signed domain (cm^-1):",
    tuple(value / CM_TO_RAD_PS for value in resolved.domain),
)


def run(case):
    hamiltonian, coupling = quapi_equivalent_matrices(case)
    bath = Bath(
        J=spectral_density,
        beta=beta,
        domain=resolved.domain,
        n_modes=resolved.n_modes,
        phys_dim=PHYS_DIM,
        discretization="tedopa",
    )
    model = SystemBath(
        h=hamiltonian,
        coupling=coupling,
        bath=bath,
    )
    return model.run(
        dt=DT_PS,
        t_max=T_MAX_PS,
        method="interaction-chain-trotter-mpo",
        trunc_eps=1e-4,
        bond_dim=None,
        initial=np.array([1.0, 0.0, 0.0]),
        observables=OBSERVABLES,
    )


cases = ("diagonal_reference", "weak_diagonal", "noncondon")
results = {case: run(case) for case in cases}

for case, result in results.items():
    populations = {
        name: np.asarray(result.expect[name], dtype=float)
        for name in OBSERVABLES
    }
    total = sum(populations.values())
    print(case)
    print("  probability error:", np.max(np.abs(total - 1.0)))
    print("  maximum bridge population:", np.max(populations["bridge"]))
    print("  final acceptor population:", populations["acceptor"][-1])
    print("  peak MPS bond:", np.max(result.max_bond))

figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
plot_cases = ("diagonal_reference", "noncondon")
titles = ("(a) diagonal, V=22/45", "(b) V=2/2")

for axis, case, title in zip(axes, plot_cases, titles):
    result = results[case]
    for name in OBSERVABLES:
        axis.plot(
            result.t, result.expect[name], color=COLORS[name], label=name,
        )
    if case == "noncondon":
        control = results["weak_diagonal"]
        for name in OBSERVABLES:
            axis.plot(
                control.t, control.expect[name], "--", color=COLORS[name],
                alpha=0.7,
            )
        axis.plot([], [], "k--", label="weak diagonal control")
    axis.set_title(title)
    axis.set_xlabel("time (ps)")

axes[0].set_ylabel("population")
axes[0].legend()
axes[1].legend()
figure.tight_layout()
plt.show()
```

`resolved.n_modes` is much larger than 12 for this bath and horizon. That is a
consequence of the paper's $600$ cm$^{-1}$ cutoff and the signed thermal domain,
not an arbitrary accuracy preference.

## 5. What the API represents

`SystemBath` keeps the three electronic states on one tensor site. Its coupling
matrix multiplies one collective bath coordinate, exactly as $M$ does in the
Hamiltonian. In the non-Condon calculation, the same fluctuation therefore
changes site energies and the $D$--$B$ and $B$--$A$ couplings. The
`h_explicit_cm` construction translates the QUAPI local-renormalization
convention to this explicit-bath API.

`interaction-chain-trotter-mpo` means:

- construct and thermalize the bath in star form;
- take the interaction picture with respect to the free star bath;
- transform the time-dependent star coupling to chain coordinates; and
- apply the resulting coupling with a Trotter MPO on an MPS.

The free-bath phase is integrated analytically over each step. Nevertheless,
the noncommuting system and coupling terms still produce Trotter error, so the
time step must be converged.

The initial state is $|D\rangle\langle D|$ with a factorized thermal bath, as in
the paper. No bath displacement conditioned on the donor is included.

## 6. Comparing the complete dynamics with Fig. 2

![Tensor-network and transfer-tensor donor, bridge, and acceptor populations overlaid with digitized paper curves, with pointwise residuals below](../img/bridge_electron_transfer.png)

```{include} ../_generated/bridge_electron_transfer.md
```

Solid lines are this package's calculation. Open circles are values extracted
from the actual vector paths in the paper PDF at 0.05 ps intervals; they are not
points estimated by clicking a raster image. The lower panels show

$$
\Delta P_i(t)=P_i^{\mathrm{TN+TTM}}(t)-P_i^{\mathrm{paper}}(t)
$$

for all three states. The digitized paper curves were not renormalized: their
population sums deviate from one by as much as about 0.006, which provides a
practical scale for interpreting sub-percent residuals.

The comparison is more informative than checking two lifetimes. It tests the
initial donor loss, the height and time of the transient bridge population, and
the acceptor rise over the entire 15 ps window. The weak-diagonal and non-Condon
calculations still have identical $H_S$ in the paper's convention; their
different dynamics comes from the off-diagonal entries of $M$. The diagonal
reference uses larger electronic couplings and is a separate benchmark.

## 7. From 0.15 ps direct dynamics to 15 ps

A direct 15 ps interaction-chain calculation asks the automatic light-cone
resolver for nearly one thousand modes and develops large MPS bonds. The bath
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

import numpy as np

from fishbonett.rates import predict_density_mat, transfer_mat


data = np.load(
    Path("examples/reference_data")
    / "bridge_electron_transfer_ttm_maps.npz"
)
dt = float(data["dt_ps"])
steps = round(15.0 / dt)
rho0 = np.diag([1.0, 0.0, 0.0]).astype(complex)

for case in ("diagonal_reference", "noncondon"):
    maps = data[f"{case}_maps"]
    transfer_tensors, transfer_norm = transfer_mat(maps)

    # The directly simulated donor trajectory seeds one full memory window.
    direct = np.einsum(
        "tij,j->ti", maps, rho0.reshape(9)
    ).reshape(-1, 3, 3)
    rdm = predict_density_mat(steps, transfer_tensors, direct)
    population = np.diagonal(rdm, axis1=1, axis2=2).real

    print(case)
    print("  final populations:", population[-1])
    print("  final transfer-tensor norm:", transfer_norm[-1])
```

To regenerate those maps rather than load them, run the expensive tomography
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

## 8. Numerical evidence and remaining convergence checks

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
- The reconstructed dynamical maps preserve trace to $3.4\times10^{-16}$. Their
  most negative Choi eigenvalue is $-2.4\times10^{-5}$, a small non-CP error
  from independently truncating the tomography trajectories that should also
  decrease in the tighter-threshold publication check.
- The 15 ps propagated density matrices preserve trace to $1.1\times10^{-11}$
  and remain positive to numerical precision.

For a final publication benchmark, also regenerate the complete nine-column
maps at Fock dimension 10, tighten the SVD threshold to $5\times10^{-5}$, and
repeat at a smaller timestep while tightening the threshold with it. Timestep
and truncation cannot be converged independently: smaller steps create smaller
Schmidt values, which a fixed loose threshold may discard before they
accumulate. The full population residuals, not merely the fitted lifetimes,
should remain stable under each refinement.

## 9. Physical conclusion

The calculation reproduces the complete Fig. 2 dynamics to an all-population
RMSE below 0.006 for both coupling models. It captures the roughly 25% transient
bridge population, the donor decay, and the acceptor rise, as well as the
reported low-picosecond lifetimes.

The physical result is striking: off-diagonal bath fluctuations restore
low-picosecond transfer even when both bare electronic couplings are reduced to
2 cm$^{-1}$. The weak-diagonal fixed-Hamiltonian control remains slow, showing
that the recovery is caused by non-Condon fluctuations rather than by the weak
electronic Hamiltonian alone.

## 10. Common mistakes

- Preserving $\lambda_R$ while changing $\alpha$ and $\omega_c$ does not preserve
  the bath correlation function.
- Omitting the weak-diagonal control prevents a fixed-Hamiltonian attribution of
  the non-Condon enhancement.
- Using cm$^{-1}$ Hamiltonian entries directly with a ps step changes every
  dynamical timescale.
- Converting $H_S$ but not $J(\omega)$ and $\beta$ is inconsistent.
- Omitting the QUAPI-to-explicit-bath renormalization propagates a different
  Hamiltonian and can localize the donor for these parameters.
- Keeping only the diagonal of $M^2$ is not equivalent to applying
  $\lambda_RD^2$ in the coupling eigenbasis.
- Counting negative thermofield frequencies in $\lambda_R$ double-counts
  temperature rather than molecular reorganization.
- Fitting the 0.2 ps transient cannot recover a 2.5 ps lifetime.
- Calling $-\dot P_D$ a forward rate ignores bridge-mediated back flux and
  recrossing.
