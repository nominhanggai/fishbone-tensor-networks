# Donor--bridge--acceptor electron transfer

This tutorial tests how fluctuations of electronic couplings can accelerate
electron transfer through a molecular bridge. It reproduces the model behind
Fig. 2 of [Acharyya, Ovcharenko, and
Fingerhut](https://doi.org/10.1063/5.0027976)
([preprint](https://arxiv.org/abs/2108.11175)) with a tensor-network bath.

The complete program below is an automatically resolved-bath *early-time
tutorial* that runs to 0.2 ps. It shows the onset of population transfer and the
non-Condon transient. The paper's 2.36 and 2.50 ps donor lifetimes remain
separate 15 ps validation targets and must not be fitted from this short trace.

## 1. Physical model

Use the diabatic basis $\{|D\rangle,|B\rangle,|A\rangle\}$ for donor, bridge,
and acceptor. The Hamiltonian is

$$
H=H_S+H_B+M\sum_j c_jx_j+\lambda_R M^2,
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

The last term is the reorganization counterterm. This reproduction interprets
the quoted diabatic energies as the minima of the three displaced bath
potentials. Fishbone's `SystemBath` represents a raw bilinear Hamiltonian and
does not guess whether a particular model uses vertical energies or
potential-minimum energies, so the tutorial adds the counterterm explicitly.
Omitting it changes the energy landscape: for diagonal
$M=\operatorname{diag}(2,1,0)$, the bath would lower the donor minimum by
$4\lambda_R$ and spuriously trap the initial population.

For non-Condon coupling, $M^2$ is itself non-diagonal. The counterterm must
therefore be evaluated as the full matrix product; shifting only the three
diagonal energies is not equivalent.

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


def system_matrices(case):
    """Return the propagation Hamiltonian and dimensionless bath operator M."""
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

    # The paper quotes potential-minimum energies.  SystemBath supplies the
    # bilinear M X interaction but deliberately adds no model-specific
    # counterterm, so include lambda_R M^2 here.
    h_propagation_cm = (
        h_cm + REORGANIZATION_CM * (coupling @ coupling)
    )
    return CM_TO_RAD_PS * h_propagation_cm, coupling


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
    hamiltonian, coupling = system_matrices(case)
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
`h_propagation_cm` construction keeps the paper's potential-minimum energy
convention while using this bilinear API.

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

## 6. Reading the early dynamics

![Early donor, bridge, and acceptor populations with the weak-coupling control shown dashed](../img/bridge_electron_transfer.png)

```{include} ../_generated/bridge_electron_transfer.md
```

The weak-diagonal and non-Condon calculations have identical *bare* $H_S$.
Their different dynamics comes from the off-diagonal entries of $M$, including
the corresponding terms required in $\lambda_RM^2$. The diagonal reference uses
much larger bare electronic couplings and is a separate benchmark trajectory.

The donor loss and compensating bridge/acceptor growth are the population
transfer. Probability conservation checks that the apparent loss is not tensor
truncation. At 0.2 ps, however, none of the curves has sampled a 2.4 ps decay;
the generated table therefore reports the lifetimes as unresolved. Fitting this
inertial transient would manufacture a rate.

## 7. Reproducing the reported lifetimes

The paper plots approximately 15 ps and reports single-exponential donor
lifetimes of 2.36 ps for the diagonal reference and 2.50 ps for the non-Condon
case. The example provides a manual reference profile with that horizon:

```bash
python examples/bridge_electron_transfer.py --profile reference \
  --output examples/output/bridge_electron_transfer_reference.npz
```

That profile is intentionally unsuitable for a documentation build: the bath
light cone approaches one thousand modes at 15 ps and it repeats the calculation
with larger Fock spaces and a tighter SVD threshold.

For a converged long trajectory, a descriptive lifetime can be extracted from
the approximately exponential portion of $P_D(t)$:

```python
def effective_lifetime(result):
    donor = np.asarray(result.expect["donor"], dtype=float)
    mask = (donor < 0.9) & (donor > 0.15)
    if np.count_nonzero(mask) < 3:
        raise ValueError("the donor decay is not resolved")
    slope, intercept = np.polyfit(result.t[mask], np.log(donor[mask]), 1)
    if slope >= 0.0:
        raise ValueError("the selected donor population is not decaying")
    return -1.0 / slope
```

This lifetime characterizes the full donor--bridge--acceptor population trace.
It is not an elementary $k_{D\to A}$: bridge occupation, back transfer, and
recrossing are all folded into the fit.

A transfer-tensor extrapolation can reduce the long-time cost only after all
nine electronic dynamical-map columns have been simulated and the transfer
tensor norm has decayed within the directly simulated memory window. A transfer
tensor built from the donor trajectory alone is not a valid reduced dynamical
map.

## 8. Required convergence checks

Before comparing a fitted lifetime with the paper:

1. Compare `DT_PS=0.002`, 0.001, and 0.0005 at common physical times.
2. Increase `PHYS_DIM` from 6 to 10, 20, and, if needed, 40.
3. Tighten `trunc_eps` from $10^{-4}$ to $5\times10^{-5}$ while leaving
   `bond_dim=None`, so discarded SVD weight remains the primary bond control.
4. Compare the automatically resolved mode count with a larger explicit count
   and verify that the entire population curves are unchanged.
5. Verify the signed frequency-domain tails by increasing their coverage.
6. Confirm probability conservation and inspect `result.max_bond` for continuing
   growth.
7. Fit only after the 15 ps populations themselves are stable under all of the
   above changes.

The time step and SVD threshold must be refined together. An MPO step creates
new Schmidt values proportional to the step size; if `trunc_eps` is too loose,
halving `DT_PS` can discard those values before they accumulate and falsely
suppress transfer. For the 0.2 ps profile at `trunc_eps=1e-4`, the largest
population change was $1.63\times10^{-3}$ from 2 fs to 1 fs and
$8.09\times10^{-4}$ from 1 fs to 0.5 fs. At `trunc_eps=1e-3`, the same step
refinement did not converge, which is why the plotted profile uses the tighter
threshold with no bond cap.

Agreement of the two fitted lifetimes alone is insufficient: an erroneous bath
can preserve a rate accidentally while changing bridge and acceptor dynamics.

## 9. Physical conclusion

The paper's comparison is striking because adding the off-diagonal bath
operator restores low-picosecond transfer even when both bare electronic
couplings are reduced from tens of cm$^{-1}$ to 2 cm$^{-1}$. The weak-diagonal
control shows that the enhancement is caused by non-Condon fluctuations rather
than by the weak electronic Hamiltonian itself.

The early-time tutorial demonstrates that mechanism and checks the package
mapping, including the energy counterterm needed for this electron-transfer
convention. Recovery of the numerical values $2.36$ and $2.50$ ps remains a
long-time convergence result, not a conclusion encoded into the page.

## 10. Common mistakes

- Preserving $\lambda_R$ while changing $\alpha$ and $\omega_c$ does not preserve
  the bath correlation function.
- Omitting the weak-diagonal control prevents a fixed-Hamiltonian attribution of
  the non-Condon enhancement.
- Using cm$^{-1}$ Hamiltonian entries directly with a ps step changes every
  dynamical timescale.
- Converting $H_S$ but not $J(\omega)$ and $\beta$ is inconsistent.
- Omitting $\lambda_RM^2$ interprets the quoted potential-minimum energies as
  vertical energies and can spuriously localize the donor.
- Keeping only the diagonal of $M^2$ changes the non-Condon Hamiltonian.
- Counting negative thermofield frequencies in $\lambda_R$ double-counts
  temperature rather than molecular reorganization.
- Fitting the 0.2 ps transient cannot recover a 2.5 ps lifetime.
- Calling $-\dot P_D$ a forward rate ignores bridge-mediated back flux and
  recrossing.
