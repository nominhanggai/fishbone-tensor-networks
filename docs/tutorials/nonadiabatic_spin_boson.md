# Strong-coupling nonadiabatic spin--boson dynamics

This tutorial reproduces a demanding two-state benchmark from
[Nuomin, Beratan, and Zhang](https://arxiv.org/abs/2111.14308). The purpose is not
just to draw a relaxation curve: it shows how two Hamiltonian representations can
cross-check the same strong-coupling dynamics.

## Benchmark Hamiltonian

$$
H = \Delta\sigma_x + \sigma_z\sum_k g_k(a_k+a_k^\dagger)
    + \sum_k\omega_k a_k^\dagger a_k,
$$

with $\Delta=1$ and

$$
J(\omega)=\frac{\eta\omega_c\omega}{\omega_c^2+\omega^2},
\qquad \eta=4,\quad \omega_c=4,\quad T=4.
$$

The system starts in the positive $\sigma_z$ state. These parameters are both
hot and strongly coupled, so weak-coupling kinetic equations are not a reliable
reference.

```python
bath = Bath(J=spectral_density, beta=0.25,
            discretization="tedopa", phys_dim=profile.phys_dim)
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)
```

## Representation comparison

The primary calculation uses `interaction-chain-trotter-mpo`. The comparison uses
`schrodinger-chain-tdvp2`, in which the bath Hamiltonian remains explicit and a
two-site TDVP sweep grows the bond as needed. Agreement is informative because
the two calculations organize bath dynamics and truncation differently.

This is not a timing contest between integrators. A difference could come from
time discretization, chain resolution, Fock truncation, SVD truncation, or the
TDVP projection. Each must be converged independently.

| profile | $dt$ | final $t\Delta/\pi$ | Fock dimension | representations |
|---|---:|---:|---:|---|
| `smoke` | 0.01 | 0.013 | 3 | interaction chain |
| `docs` | 0.025 | 1 | 6 | interaction chain; Schrödinger-chain overlap to $t\Delta/\pi=0.25$; 24 modes on $[-16,80]$ |
| `reference` | 0.0125 | 5 | 20 | both chains plus Schrödinger star, automatic domain/modes |

The Drude tail makes a 99.9%-reorganization automatic domain extremely wide.
The documentation profile therefore states its finite window explicitly; it is a
build-time comparison, not the final cutoff-convergence claim. The manual
reference profile removes that shortcut and should be used to check sensitivity
to the high-frequency tail. The TDVP comparison is deliberately limited to the
initial quarter interval in the generated docs: an uncapped full-interval run is
too expensive for every Sphinx build, while the reference profile compares the
complete horizon.

```bash
python examples/nonadiabatic_spin_boson.py --profile docs \
  --output examples/output/nonadiabatic_spin_boson_docs.npz
```

## Dynamics and conclusion

![Strong-coupling population dynamics and retained bond dimensions](../img/nonadiabatic_spin_boson.png)

```{include} ../_generated/nonadiabatic_spin_boson.md
```

The scientific conclusion is the converged population trace, not the fact that
one representation used fewer tensors or a smaller peak bond in one run. For a
publication calculation, repeat at Fock dimensions 10, 20, and 40, halve the time
step, tighten the SVD threshold, and verify that the star comparison remains
consistent over the same physical interval.
