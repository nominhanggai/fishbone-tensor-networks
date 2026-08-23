# Donor--bridge--acceptor electron transfer

Electron transfer through a molecular bridge need not be controlled only by the
static electronic couplings. This tutorial compares diagonal (Condon) bath
fluctuations with bath-modulated off-diagonal (non-Condon) coupling using the
model of [Acharyya, Ovcharenko, and Fingerhut](https://doi.org/10.1063/5.0027976)
([preprint](https://arxiv.org/abs/2108.11175)).

## Three-state model and units

The diabatic energies are

$$
(E_D,E_B,E_A)=(0,-150,-1000)\ {\rm cm}^{-1}.
$$

The Condon case uses $V_{DB}=22$, $V_{BA}=45$, and $V_{DA}=0$ cm$^{-1}$,
with bath operator $M=\operatorname{diag}(2,1,0)$. The non-Condon case reduces
both bare nearest-neighbour couplings to 2 cm$^{-1}$ and adds
$M_{DB}=0.17$ and $M_{BA}=0.055$ (with Hermitian conjugates).

Both use one correlated bath,

$$
J(\omega)=\frac{\alpha\pi}{2}\omega e^{-\omega/\omega_c},
\quad \alpha=10.02,\quad\omega_c=100\ {\rm cm}^{-1},
$$

whose reorganization energy is $\lambda=\alpha\omega_c/2=501$ cm$^{-1}$.
The large apparent donor--acceptor gap should therefore not be interpreted
without the bath reorganization: this is close to a barrierless transfer regime.

The code converts every energy to angular ps$^{-1}$ using

```python
CM_TO_RAD_PS = 2 * np.pi * 2.99792458e10 * 1e-12
```

and transforms the spectral density and inverse temperature consistently. Mixing
cm$^{-1}$ Hamiltonian entries with a ps time step would change the model by a
factor of $2\pi c$.

## Propagation and observables

The three diabatic states form one three-level `SystemBath`; this correctly
represents their shared correlated fluctuations. The interaction-chain
representation avoids carrying the free bath evolution in the state.

```python
result = model.run(
    dt=0.002, t_max=4.0,
    method="interaction-chain-trotter-mpo",
    trunc_eps=1e-3, bond_dim=None,
    initial=np.array([1.0, 0.0, 0.0]),
    observables={"donor": P_D, "bridge": P_B, "acceptor": P_A},
)
```

The documentation build uses 12 TEDOPA modes on the automatically selected
frequency domain. Its 0.005 ps step gives about seven points across the fastest
stated electronic period ($1000$ cm$^{-1}$ corresponds to about 0.033 ps), and
propagates the first 0.2 ps with a local Fock dimension of 6. This is a build-time
dynamics check, not a converged lifetime calculation. The reference
profile uses a 0.001 ps step, extends the run to 10 ps, lets the light-cone resolver
choose the mode count, and checks Fock dimensions 20 and 40 plus SVD thresholds
$10^{-3}$ and $5\times10^{-4}$.

## Dynamics and conclusion

![Donor, bridge, and acceptor populations](../img/bridge_electron_transfer.png)

```{include} ../_generated/bridge_electron_transfer.md
```

The reported lifetime is a descriptive exponential fit to the donor population.
It is not the elementary forward rate $k_{D\to A}$: population dynamics obey a
network of forward, backward, and bridge-mediated fluxes, and an instantaneous
flux contains all of them. The bridge population shown in the figure is therefore
essential evidence when comparing mechanisms. Reference values near 2.36 ps
(Condon) and 2.50 ps (non-Condon) are useful validation targets only after the
time-step, mode, Fock-space, and SVD studies have converged.
