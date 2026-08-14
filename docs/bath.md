# Baths: discretization, chain mapping and temperature

```{admonition} Summary
:class: tip
{py:class}`~fishbonett.bath.spec.Bath` bundles a spectral density with its
discretization settings (`n_modes`, `phys_dim`, `domain`, `discretization`).
Both `domain` and `n_modes` can be left unset — they are derived automatically.
For finite temperature, set `temperature` or `beta` (T-TEDOPA thermalization).
Use `discretization="tedopa"` when $J$ is sharply peaked or infrared-divergent.
```

Every method starts from a continuous bath spectral density $J(\omega)$ and turns
it into a finite set of harmonic modes.  The {py:class}`~fishbonett.bath.spec.Bath`
object bundles that spectral density with the choices that control the mapping.

```python
from fishbonett import Bath

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5),   # spectral density J(w)
            domain=(-25, 36),                       # signed frequency window
            n_modes=40, phys_dim=20,                # modes, boson Hilbert dim
            temperature=1.0,                        # for thermalization
            discretization="legendre")              # or "tedopa"
```

- **`domain`** — the (signed) frequency window the spectral density is sampled on
  (optional — see *Automatic defaults*).
- **`n_modes`** — the number of discretized modes (optional — see below).
- **`phys_dim`** — the local boson Hilbert-space truncation per mode.
- **`temperature` / `beta`** — finite-temperature thermalization (below).
- **`discretization`** — `"legendre"` (default) or `"tedopa"`.

The bath does not need to own a system operator. `SystemBath(coupling=...)` owns
that part of the physical model. For a multi-site model, bind the operator that
connects a particular bath to its system site:

```python
coupled = bath.bind(sigma_z)
```

Representations receive `bath` directly and discretize it into their own finite
star or chain coefficients. Those coefficients are private implementation data,
not another public bath type. `CoupledBath` records only the physical association
between a bath and one or more model operators. The old `Bath(coupling=...)`
spelling emits `DeprecationWarning` and will be removed in a future major release.
If it duplicates `SystemBath(coupling=...)`, the values must agree.

## Automatic defaults

`domain` and `n_modes` can both be left unspecified; they are then derived from
the spectral density and the propagation time:

```python
bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), temperature=1.0, phys_dim=20)
# domain and n_modes chosen automatically at run() time
```

- **`domain`** defaults to the window that captures **99.9% of the reorganization
  energy** $\lambda = \tfrac{1}{\pi}\int_0^\infty J(\omega)/\omega\,d\omega$:
  $(0, \omega_{hi})$ at zero temperature.  With a temperature set, the thermofield
  density lives on both frequency halves, $J_\beta(+\omega) = J(\omega)(n_\beta+1)$
  and $J_\beta(-\omega) = J(\omega)\,n_\beta$; each half is truncated by **its own**
  reorganization-energy tail, so the window is **asymmetric**,
  $(-\omega_{lo}, \omega_{hi})$, with the thermally-suppressed negative edge much
  closer to zero (and widening with temperature).
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
from fishbonett import Bath
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
:width: 80%
:align: center

The automatic bath (markers) reproduces both the real and imaginary parts of the
exact correlation function (lines).  **Inset:** the relative error stays around
$10^{-2}$ for the automatic choice, but a too-small mode count develops a
finite-size recurrence and a too-narrow domain misses spectral weight — each wrong
by tens of percent.
```

So the reorganization-energy window and the light-cone mode count are each doing
real work: drop either and the bath correlation function is wrong by tens of
percent; keep both and it is faithful for the entire propagation.

A **structured** bath is handled the same way.  Take a more realistic density — a
weak Ohmic background plus two underdamped vibrational peaks,
$J(\omega) = 0.05\,\omega\,e^{-\omega/2.5} + \sum_{k} \frac{2\lambda_k\gamma_k
\Omega_k^2\,\omega}{(\Omega_k^2-\omega^2)^2 + \gamma_k^2\omega^2}$ with peaks at
$\Omega = 6, 13$.  Its correlation function is strongly oscillatory (the two peaks
beat against each other), yet the automatic construction covers **both** peaks —
the 99.9% reorganization-energy window reaches out to $\omega_{hi} \approx 29.5$ —
and the light cone asks for ~80 modes, which sample the peaks finely enough to
reproduce $C(t)$ to a few $\times 10^{-2}$ over the whole run:

```{figure} img/bath_structured.png
:alt: Left, a structured spectral density with two peaks, the star modes sampling it within the auto domain edge at 32.1; right, the strongly oscillatory correlation function with the auto-discretized bath (markers) on the exact curve and an inset error around 1e-3.
:width: 100%
:align: center

**Left:** a structured spectral density (Ohmic background + two vibrational peaks);
the automatic domain reaches past both peaks and the star modes (markers) sample
them — densely where $J$ has weight.  **Right:** the resulting oscillatory
correlation function, reproduced by the automatic bath to a few percent even
after several beat periods (inset).
```

The same holds at **finite temperature**.  There the correlation function carries
the detailed-balance factor,
$C(t) = \tfrac{1}{\pi}\int_0^\infty d\omega\, J(\omega)[\coth(\tfrac{\beta\omega}{2})
\cos\omega t - i\sin\omega t]$, and the automatic bath uses the **asymmetric signed
domain** above (thermofield / T-TEDOPA), each half sized by its own
reorganization-energy tail — for $k_B T = 1$ here, $(-4.3, 34.9)$ and 100 modes.
It reproduces the thermal $C(t)$ just as faithfully (peak error
$7.6 \times 10^{-3}$) at far fewer modes than a symmetric window would need, while
too few modes or too narrow a domain fail in the same way:

```{figure} img/bath_correlation_finiteT.png
:alt: Finite-temperature bath correlation function; the auto thermofield bath (156 modes on a signed domain) lands on the exact thermal C(t), with an inset showing the relative error stays ~1e-3 while too-few-modes and too-narrow-domain discretizations rise toward 1.
:width: 80%
:align: center

Finite temperature ($k_B T = 1$).  The automatic thermofield bath (100 modes on
the asymmetric signed domain $(-4.3, 34.9)$, markers) reproduces the thermal
correlation function (lines); the inset shows the same separation between the
faithful automatic choice and the degraded discretizations.
```

## Finite star data and star-to-chain mapping

`fishbonett` supports two equivalent computational routes to a finite harmonic
bath. A quadrature produces independent star modes directly. Alternatively,
orthogonal-polynomial recurrences produce chain coefficients, and diagonalizing
that finite chain recovers independent modes. The numerical route does not define
the Hamiltonian representation selected later.

In the recurrence route, $J$ is the weight function and the chain parameters are
read from the three-term recurrence,

$$
\omega\,p_n(\omega) = t_{n+1}\,p_{n+1}(\omega) + \epsilon_n\,p_n(\omega)
                      + t_n\,p_{n-1}(\omega),
$$

so $(\epsilon_n,t_n)$ are the chain on-site energies and hoppings. This is
{py:func}`fishbonett.bath.chain.get_bath_nn_paras`, built on
{py:mod}`fishbonett.bath.recurrence`.

```python
from fishbonett.bath.chain import get_bath_nn_paras

eps_i, t_i = get_bath_nn_paras(bath.spectral_density(), n=40, domain=(-25, 36))
```

A finite chain can be diagonalized to obtain frequencies $\omega_k$, couplings
$g_k$, and an orthogonal transform. This is a way to generate a finite star
discretization. The interaction construction then starts from that star:

1. absorb the free-star evolution into $g_ke^{-i\omega_kt}$;
2. keep those modes for `interaction-star`, or apply the inverse transform
   star-to-chain for `interaction-chain`.

Consequently, chain diagonalization is not the definition of
`interaction-chain`. It is one way to prepare the finite star from which the
interaction representation is built. The whole mapping is implemented in
NumPy/SciPy without an external ORTHPOL dependency.

## `legendre` vs `tedopa`

Because quadrature and chain mapping are two numerical views of one finite bath,
`fishbonett` lets you pick which measure the $n$ modes are placed against.

The default `"legendre"` puts them at the Gauss–Legendre nodes of the *uniform*
measure on the domain (de Vega & Bañuls 2015): frequencies $\omega_k$ are the nodes
and couplings $g_k=\sqrt{J(\omega_k)\,w_k/\pi}$ come from the quadrature weights.
The nodes ignore $J$, which makes them robust and — crucially — **shared across
channels**, which is why a {doc}`multichannel bath <models/composite_multichannel>`
requires this setting.  This is
{py:func}`fishbonett.bath.legendre.get_vn_squared`.

`"tedopa"` instead uses the measure $d\mu(\omega)=J(\omega)\,d\omega$ — the actual
TEDOPA weight function — and builds its $n$-point Gauss quadrature (RKPW Lanczos +
Golub–Welsch).  The nodes cluster where the density lives, including the infrared,
and the quadrature is exact for polynomials up to degree $2n-1$, so it reproduces
bath autocorrelation functions to near machine precision where the uniform grid
gives only a few digits.  Prefer it when $J$ is sharply peaked or
infrared-divergent (sub-Ohmic).

```python
peaked = Bath(J=my_peaked_density, domain=(0, 40), n_modes=40, phys_dim=20,
              discretization="tedopa")             # resolves the peak
```

```{note}
The name is `"tedopa"`, not `"tedopa"`: *both* settings are
orthogonal-polynomial methods (Legendre polynomials are orthogonal too), so that
would not name the difference.  What distinguishes this one is that it uses $J$ as
the weight function, which is exactly TEDOPA.  `ORTHPOL` is the external Fortran
package the scheme was originally taken from; `fishbonett` does not depend on it.
```

See {py:mod}`fishbonett.bath.tedopa`; the `discretization` choice is threaded all
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
transform is {py:func}`fishbonett.bath.spec.thermalize`, usable on its own:

```python
from fishbonett.bath import thermalize

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
