# Baths: discretization, chain mapping and temperature

Every method starts from a continuous bath spectral density $J(\omega)$ and turns
it into a finite set of harmonic modes.  The {py:class}`~fishbonett.simulate.Bath`
object bundles that spectral density with the choices that control the mapping.

```python
from fishbonett.simulate import Bath

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5),   # spectral density J(w)
            domain=(-25, 36),                       # signed frequency window
            n_modes=40, phys_dim=20,                # modes, boson Hilbert dim
            temperature=1.0,                        # for thermalization
            discretization="legendre")              # or "orthpol"
```

- **`domain`** — the (signed) frequency window the spectral density is sampled on
  (optional — see *Automatic defaults*).
- **`n_modes`** — the number of discretized modes (optional — see below).
- **`phys_dim`** — the local boson Hilbert-space truncation per mode.
- **`temperature` / `beta`** — finite-temperature thermalization (below).
- **`discretization`** — `"legendre"` (default) or `"orthpol"`.

## Automatic defaults

`domain` and `n_modes` can both be left unspecified; they are then derived from
the spectral density and the propagation time:

```python
bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), temperature=1.0, phys_dim=20)
# domain and n_modes chosen automatically at run() time
```

- **`domain`** defaults to the window that captures **99.9% of the reorganization
  energy** $\lambda = \tfrac{1}{\pi}\int_0^\infty J(\omega)/\omega\,d\omega$
  ($(0, \omega_{hi})$, or $(-\omega_{hi}, \omega_{hi})$ when a temperature is set,
  since a thermofield density lives on both frequency halves).
- **`n_modes`** defaults to the **light-cone extent** of the interaction-picture
  chain couplings $d_j(t)$: the coupling to chain site $j$ stays negligible until
  the excitation front reaches it, so a run of length `t_max` only needs the first
  $j_{max}$ sites (plus a buffer).  Because `n_modes` depends on `t_max`, it is
  resolved when you call `run`.

Both live in {py:mod}`fishbonett.bath.auto`
({py:func}`~fishbonett.bath.auto.reorganization_energy`,
{py:func}`~fishbonett.bath.auto.auto_domain`,
{py:func}`~fishbonett.bath.auto.auto_n_modes`) and can be called directly.  A
heavy-tailed density (e.g. Drude, $J\sim 1/\omega$) has a slowly-converging
reorganization integral, so its 99.9% window is wide — set `domain` explicitly if
you want a tighter one.

### Are the automatic choices good?

The single quantity that a discretized bath has to reproduce is the **bath
correlation function**

$$
C(t) = \frac{1}{\pi}\int_0^\infty d\omega\, J(\omega)
       \big[\coth(\tfrac{\beta\omega}{2})\cos\omega t - i\sin\omega t\big]
       \;\xrightarrow{\;T=0\;}\;
       \frac{1}{\pi}\int_0^\infty d\omega\, J(\omega)\,e^{-i\omega t},
$$

which is what enters the influence functional.  The discretized bath gives
$C_{\mathrm{disc}}(t) = \sum_k g_k^2\, e^{-i\omega_k t}$ (with $g_k^2 = J(\omega_k)
w_k/\pi$ the Gauss couplings), and it tests **both** automatic choices at once: the
`domain` fixes the spectral mass — and hence $C(0)$ and the short-time behaviour —
while `n_modes` sets how long $C_{\mathrm{disc}}$ tracks the exact $C$ before the
finite mode count produces a spurious recurrence.

For an Ohmic bath with an exponential cutoff, $J(\omega) = \eta\,\omega\,
e^{-\omega/\omega_c}$, the correlation function is analytic,
$C(t) = (\eta/\pi)\,(1/\omega_c + i t)^{-2}$, so the discretization error is exact
to read off:

```python
import numpy as np
from fishbonett.simulate import Bath
from fishbonett.bath.legendre import get_vn_squared

eta, wc, t_max = 0.2, 5.0, 4.0
J = lambda w: eta * w * np.exp(-w / wc)
C_exact = lambda t: (eta / np.pi) / (1 / wc + 1j * t) ** 2      # analytic C(t)

def C_disc(domain, n_modes, ts):
    freq, v_sq = get_vn_squared(J, n_modes, list(domain))       # star nodes, g^2*pi
    return (np.asarray(v_sq)[None, :] / np.pi
            * np.exp(-1j * np.outer(ts, freq))).sum(axis=1)

bath = Bath(J=J, phys_dim=10).resolved(t_max)     # -> domain=(0, 34.9), n_modes=91
ts = np.linspace(0, t_max, 400)
rel = lambda d, n: np.max(np.abs(C_disc(d, n, ts) - C_exact(ts))) / abs(C_exact(0))
print(rel(bath.domain, bath.n_modes))             # 7.5e-3
```

The automatic bath reproduces $C(t)$ to better than 1% over the whole run, while
degrading either choice breaks it — the peak relative error
$\max_t |C_{\mathrm{disc}}(t) - C(t)| / |C(0)|$ on $[0, t_{max}]$ is:

| discretization | peak relative error |
|----------------|:-------------------:|
| **auto domain + auto `n_modes`** | **7.5 × 10⁻³** |
| auto domain, too few modes (20) | 5.5 × 10⁻¹ |
| too-narrow domain $(0, 10)$, auto modes | 4.1 × 10⁻¹ |

```{figure} img/bath_correlation.png
:alt: Real and imaginary parts of the bath correlation function C(t); the auto-discretized bath lands exactly on the exact curve, with an inset showing its relative error stays ~1e-2 while too-few-modes and too-narrow-domain discretizations rise toward 1.
:width: 100%

The automatic bath (markers) reproduces both the real and imaginary parts of the
exact correlation function (lines).  **Inset:** the relative error stays around
$10^{-2}$ for the automatic choice, but a too-small mode count develops a
finite-size recurrence and a too-narrow domain misses spectral weight — each wrong
by tens of percent.
```

So the reorganization-energy window and the light-cone mode count are each doing
real work: drop either and the bath correlation function is wrong by tens of
percent; keep both and it is faithful for the entire propagation.

## TEDOPA: discretization then chain mapping

`fishbonett` uses the TEDOPA construction (Chin *et al.* 2010; Prior *et al.*
2010).  Two steps:

1. **Discretize** the continuum into a *star* of modes.  On the default
   `"legendre"` setting the frequency window is covered by an $n$-point
   Gauss–Legendre grid (de Vega & Bañuls 2015), giving mode frequencies
   $\omega_k$ (the
   nodes) and couplings $g_k = \sqrt{J(\omega_k)\, w_k / \pi}$ from the quadrature
   weights $w_k$.  This is {py:func}`fishbonett.bath.legendre.get_vn_squared`.
2. **Chain-map** the star to a nearest-neighbour chain.  A Lanczos iteration
   tridiagonalizes the star, producing on-site energies $\epsilon_i$ and hoppings
   $t_i$ together with the single system–bath coupling $c_0$ to the first chain
   site.  This is {py:func}`fishbonett.common.get_bath_nn_paras`.

The chain form is what the MPS/MPO/tree engines evolve; the star form is what the
`*-ip-*` interaction-picture engines use directly.  The whole mapping is pure
NumPy/SciPy — there is no external Fortran (ORTHPOL) dependency.

```python
from fishbonett.common import get_bath_nn_paras

eps_i, t_i = get_bath_nn_paras(bath.spectral_density(), n=40, domain=(-25, 36))
```

## `legendre` vs `orthpol`

The default uniform-measure Gauss–Legendre grid is robust and shared across
channels (which is why a {doc}`multichannel bath <systems/composite_multichannel>`
requires it).  For spectral densities that are sharply peaked or infrared-divergent,
a uniform grid resolves them poorly.  The `"orthpol"` setting instead builds a
**measure-adapted** star, using $J$ itself as the weight of an orthogonal-polynomial
quadrature (RKPW Lanczos + Golub–Welsch); it places nodes where the density
actually lives and can reproduce bath autocorrelation functions to near machine
precision where the uniform grid gives only a few digits.

```python
peaked = Bath(J=my_peaked_density, domain=(0, 40), n_modes=40, phys_dim=20,
              discretization="orthpol")            # resolves the peak
```

See {py:mod}`fishbonett.bath.orthpol`; the `discretization` choice is threaded all
the way down to the chain mapping and the star transforms.

## Finite temperature (thermofield / T-TEDOPA)

Finite temperature is handled by **thermofield / T-TEDOPA**: a
zero-temperature density $J(\omega)$ on $\omega > 0$ is folded into an effective
density on the **signed** axis,

$$
J_\beta(\omega) = \tfrac{1}{2}\big[1 + \coth(\tfrac{\beta\omega}{2})\big]\, J(\omega),
\qquad J_\beta(-\omega)\ \text{carrying the }\coth\text{ tail},
$$

so a *pure-state* simulation on the doubled (signed) domain reproduces the thermal
dynamics.  In practice you just pass a `temperature` (or `beta`) and a signed
`domain`; `Bath.spectral_density()` returns the thermalized density.  The
transform is {py:func}`fishbonett.simulate.thermalize`, usable on its own:

```python
from fishbonett.simulate import thermalize

J0 = lambda w: 0.2 * w * np.exp(-w / 5)            # zero-T density on w > 0
J_beta = thermalize(J0, beta=1.0)                  # signed, finite-T density
```

Set `thermalized=True` if you are supplying an already-thermalized density and
want to skip the internal transform.

## Convergence checklist

- **`n_modes`** and **`domain`** together control how much of the bath (and how
  much of the correlation time) is captured — widen the domain / add modes until
  observables stop moving.
- **`phys_dim`** must be large enough to hold the boson occupation the dynamics
  populate; watch the highest-mode population.
- **`bond_dim` / `trunc_eps`** are properties of the {doc}`propagation method
  <methods/index>`, not the bath, but interact with it — a stiffer bath needs more
  bond dimension.
