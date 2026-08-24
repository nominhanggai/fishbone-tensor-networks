# Donor--bridge--acceptor electron transfer [transfer tensor method]

This tutorial tests how fluctuations of electronic couplings can accelerate
electron transfer through a molecular bridge. It reproduces the model behind
Fig. 2 of [Acharyya, Ovcharenko, and
Fingerhut](https://doi.org/10.1063/5.0027976)
([preprint](https://arxiv.org/abs/2108.11175)) with a tensor-network bath.

```{admonition} Orientation
:class: note

- **Level:** advanced; familiarity with reduced density matrices is helpful.
- **You will learn:** how correlated diagonal and off-diagonal bath fluctuations
  change donor--bridge--acceptor transfer, and why a transfer tensor requires a
  complete reduced dynamical map.
- **Cost:** the early-time tutorial is a minutes-scale tensor-network job; the
  stored-map TTM continuation takes seconds; regenerating the maps requires 18
  production tensor-network runs.
- **Output:** early populations and a validated 15 ps donor-decay comparison.
```

The program below performs an automatically resolved 0.2 ps calculation. It
then explains how a 0.15 ps reduced dynamical map supports the 15 ps
transfer-tensor result shown on this page. Map tomography, executable
long-time reconstruction, and memory-kernel convergence are kept in the
companion {doc}`bridge_electron_transfer_validation` appendix.

## Physical model

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

## The paper's bath

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

## Unit conversion

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

## Complete runnable early-time calculation

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

## What the API represents

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

## Comparing the complete dynamics with Fig. 2

![Tensor-network and transfer-tensor donor, bridge, and acceptor populations overlaid with digitized paper curves, with pointwise residuals below](../img/bridge_electron_transfer.svg)

```{include} ../_generated/bridge_electron_transfer.md
```

The legend identifies the solid curves as `tensor network + TTM` and the open
circles as `digitized paper Fig. 2`. The circles were extracted from the actual
vector paths in the paper PDF at 0.05 ps intervals; they are not points
estimated by clicking a raster image. The lower panels show

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

The curves should not be expected to coincide point for point yet. The paper
used a continuous-bath MACGIC-QUAPI calculation converged by reducing its
Trotter step while increasing the influence-functional memory. Here, the short
dynamical maps use a finite 95-mode TEDOPA bath, local Fock dimension 6, a 2 fs
step, and an SVD threshold of $10^{-4}$. Those settings reproduce the paper to
within about one percent, but they are documentation-scale and not yet the
common numerical limit of the two methods. The plotted paper curves are
vector-path samples, not the authors' raw data; their population sums
differ from one by as much as 0.006.

## How the 15 ps result is obtained

Direct propagation to 15 ps would require nearly one thousand automatically
resolved modes. Instead, nine short initial-state calculations reconstruct the
reduced dynamical map $\mathcal E_t$ through 0.15 ps. The transfer tensors are
defined recursively from those maps and retain the bath memory needed to
continue the reduced state.

A donor trajectory alone is insufficient: a three-state system needs all
$3^2=9$ map columns. The calculation uses the three basis states plus real and
imaginary pair superpositions to reconstruct those columns from physical pure
states. A 0.12 ps retained kernel predicts the held-out end of the direct
trajectory and leaves the 15 ps populations stable at the few-$10^{-3}$ level.

The complete reconstruction code, map-generation command, kernel-tail plot,
Choi and trace checks, and retained-memory study are in
{doc}`bridge_electron_transfer_validation`. General timestep, SVD, Fock-space,
and finite-bath refinement is described in {doc}`convergence`.

## Physical conclusion

The calculation reproduces the complete Fig. 2 dynamics to an all-population
RMSE below 0.006 for both coupling models. It captures the roughly 25% transient
bridge population, the donor decay, and the acceptor rise, as well as the
reported low-picosecond lifetimes.

The physical result is striking: off-diagonal bath fluctuations restore
low-picosecond transfer even when both bare electronic couplings are reduced to
2 cm$^{-1}$. The weak-diagonal fixed-Hamiltonian control remains slow, showing
that the recovery is caused by non-Condon fluctuations instead of the weak
electronic Hamiltonian alone.

## Common mistakes

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
  temperature, not molecular reorganization.
- Fitting the 0.2 ps transient cannot recover a 2.5 ps lifetime.
- Calling $-\dot P_D$ a forward rate ignores bridge-mediated back flux and
  recrossing.

For another strong-coupling interaction-chain calculation, see
{doc}`nonadiabatic_spin_boson`. For a multiple-bath problem with a directly
measured energy flux, continue to {doc}`two_bath_heat_flow`.
