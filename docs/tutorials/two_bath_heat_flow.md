# Heat flow through a two-level molecular junction

A molecule between environments at different temperatures is the smallest model
that distinguishes relaxation from transport. This tutorial follows the
two-bath spin--boson junction of
[Dunnett and Chin](https://doi.org/10.3390/e23010077) and shows how to obtain an
energy current from a system--bath correlation rather than from a population fit.

## Model

$$
H_S=\frac{\omega_0}{2}\sigma_z,\qquad
J_b(\omega)=2\pi\alpha\omega\,\Theta(\omega_c-\omega),
$$

with $\omega_c=1$, $\omega_0=0.2$, and $\alpha=0.1$. Two independent baths
couple through $\sigma_x$. The nonequilibrium case uses
$\beta_h\omega_c=2$ and $\beta_c\omega_c=100$; an equal-temperature
$\beta_h=\beta_c=100$ run is the zero-bias control.

```python
model = Fishbone(
    sites=[0.5 * omega_0 * sigma_z],
    baths={0: [hot.bind(sigma_x), cold.bind(sigma_x)]},
)
```

The mapping makes the attachment explicit: both list entries belong to system
site 0, bath 0 is hot, and bath 1 is cold.

## Measuring energy current

For the represented static chain Hamiltonian, the first chain coordinate of bath
$b$ couples as $\kappa_b\sigma_xX_b$. Therefore

$$
I_{b\to S}=\frac{d\langle H_S\rangle_b}{dt}
=\kappa_b\omega_0\langle\sigma_yX_b\rangle.
$$

`BathMode` identifies those coordinates without relying on internal tensor-node
numbers:

```python
hot_mode = BathMode(system_site=0, bath=0, mode=0)
cold_mode = BathMode(system_site=0, bath=1, mode=0)
observables = {
    "sz": (sigma_z, 0),
    "hot_system_mode": (np.kron(sigma_y, x), (0, hot_mode)),
    "cold_system_mode": (np.kron(sigma_y, x), (0, cold_mode)),
}
```

The resolved $\kappa_b$ values come from `result.meta["bath_branches"]`. The
current is not the raw correlation alone. As a consistency check,

$$
\frac{\omega_0}{2}\frac{d\langle\sigma_z\rangle}{dt}
=I_{h\to S}+I_{c\to S}.
$$

## Numerical profiles

The method is `schrodinger-chain-tree-tebd`, because the current has a simple
local form in the static chain representation. TEDOPA uses the exact signed hard
cutoff domain $(-\omega_c,\omega_c)$ and automatically resolves the mode count.

| profile | step | horizon | Fock dimension | SVD threshold |
|---|---:|---:|---:|---:|
| `smoke` | 0.05 | 0.2 | 3 | $10^{-3}$ |
| `docs` | 0.1 | 25 | 5 | $10^{-3}$ |
| `reference` | 0.025 | 40 | 6 and 8 | $10^{-3}$ and $5\times10^{-4}$ |

```bash
python examples/two_bath_heat_flow.py --profile docs \
  --output examples/output/two_bath_heat_flow_docs.npz
```

## Dynamics and conclusion

![Junction population and hot/cold currents](../img/two_bath_heat_flow.png)

```{include} ../_generated/two_bath_heat_flow.md
```

A transport conclusion requires three observations together: the system energy
must approach a plateau, the hot and cold currents must become equal and opposite,
and the equal-temperature control must approach zero current. A short transient
can satisfy none of these even when its current direction is physically sensible.
Extend and converge the reference profile if the generated balance diagnostics
have not plateaued before finite-chain recurrences.
