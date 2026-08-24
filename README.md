# fishbonett 🐟

[![CI](https://github.com/nominhanggai/fishbone-tensor-networks/actions/workflows/ci.yml/badge.svg)](https://github.com/nominhanggai/fishbone-tensor-networks/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**fishbonett** `<')+++<` is a Python package for propagating the dynamics of
multi-site vibronic model systems that have *fishbone*-like configurations —
electron-transfer and excitation-energy-transfer systems in which each electronic
or vibrational site is coupled to its own harmonic bath — using matrix-product-state
(MPS) and tree-tensor-network time-evolving-block-decimation (TEBD) methods.

The package is self-contained and low-dependency: the bath discretization / TEDOPA
chain mapping is done entirely in NumPy/SciPy (Gauss–Legendre discretization plus a
Lanczos tridiagonalization), so no hard-to-build Fortran packages are required.

## Features

- **High-level interface** (`fishbonett.models`):
  declare a bath and system as `Bath` / `SystemBath` / `Fishbone` objects and
  propagate with a single `run(dt=..., t_max=..., method=...)` call — no
  hand-written sweep loop — returning a `Result` with the time grid, observables,
  per-step reduced density matrix and peak bond dimension. The same call
  dispatches to the TEBD, MPO/TDVP and tree evolution paths below.
- **MPS TEBD engine** (`fishbonett.evolve.tebd`): nearest-neighbour and
  swap-network updates acting on a `fishbonett.states.mps.SystemBathMPS` state,
  with SVD-threshold truncation, an optional maximum bond dimension, adaptive
  rank search, **local basis optimization (LBO)**, and an optional **CuPy GPU**
  backend.
- **MPO / TDVP engine** (`fishbonett.evolve.tdvp`): a generic product-sum MPO
  compiler with exact QR/SVD bond minimization, fixed-bond 1-site TDVP, two-site
  TDVP that grows the bond by SVD truncation, and adaptive two-site tangent-space
  evolution — one driver (`run_mpo_hamiltonian`) taking a representation and a
  sweep as independent arguments.
- **Tree tensor-network TEBD**: `fishbonett.evolve.modetree` places one system and
  its bath modes on a balanced binary tree, while `fishbonett.evolve.sitetree`
  propagates multi-site and fishbone models on arbitrary loop-free trees. Both
  compress tree edges using their Schmidt spectra.
- **Self-contained bath discretization / TEDOPA chain mapping** from an arbitrary
  spectral density `J(ω)` — either the default Gauss–Legendre star or a
  **measure-adapted TEDOPA star** (`fishbonett.bath.tedopa`)
  whose nodes are placed by an N-point Gauss quadrature of the actual measure
  `J_β(ω)dω`, resolving infrared-divergent and sharply peaked baths. Selectable via
  `Bath(discretization="tedopa")`.
- **Six explicit Hamiltonian representations:** Schrödinger, interaction, and
  polaron transformations in star and chain forms. Interaction construction
  starts from the finite star, removes its free evolution, and only then applies
  the star-to-chain transform when requested.
- **Representation-independent propagation:** representations do not depend on
  TEBD or TDVP; representations produce `tebd_gates`, `tdvp_mpo`, or `trotter_mpo`.
- **Finite-temperature chain-cooling** state preparation.
- **Fishbone tree tensor network** (`Fishbone` / `TreeFishbone`) for dimers and
  multi-site vibronic models.
- **Rate theory & diabatization:** Fermi golden-rule / Marcus rates, multi-acceptor
  corrections, Metropolis integrators, the transfer-tensor method, and Boys
  localization.

## Installation

```bash
pip install -e .                 # from a checkout
# optional extras:
pip install -e ".[speed]"        # opt_einsum contraction backend (numpy fallback otherwise)
pip install -e ".[gpu-cuda12]"   # or gpu-cuda11, matching your CUDA runtime
pip install -e ".[rates]"        # vegas Monte-Carlo rate integrator
pip install -e ".[test,docs]"    # development
```

Core dependencies: `numpy` and `scipy` only. `opt_einsum` is an optional
contraction backend (`[speed]`). It is on par with `numpy.einsum` for the MPS /
MPO evolution paths but gives a large speedup for the **tree / TTN paths** (whose
multi-operand kernels numpy evaluates unoptimised), so install `[speed]` if you
use `tree-*` methods. Set `FISHBONETT_EINSUM=numpy` before importing `fishbonett`
to force the NumPy backend for benchmarking or backend comparisons. Requires
Python ≥ 3.10.

## Quick start

Declare a bath and a two-level system, then propagate with one call:

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5),   # spectral density J(w)
            domain=(-25, 36), temperature=1.0,       # T-TEDOPA thermalization
            n_modes=40, phys_dim=20,
            discretization="tedopa")                # or the default "legendre"
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)

result = model.run(dt=0.05, t_max=4.0, method="interaction-chain-tree-tebd", bond_dim=200,
                   observables={"sz": sigma_z, "sx": sigma_x})

result.t                 # time grid
result.expect["sz"]      # <sigma_z>(t)
result.max_bond          # peak bond dimension per step
```

Every `method` name begins with its Hamiltonian representation, followed
by the integrator; tree tensor-network methods also include `tree`. Examples include
`interaction-chain-tebd`, `schrodinger-chain-tdvp2`,
`polaron-chain-tdvp2`, and `interaction-chain-tree-tebd`. Every method uses the
same `dt`/`t_max` and returns the same `Result`.

A **fishbone** is a set of electronic sites, each with one or two baths. The
electronic sites can form *any* loop-free tree via `TreeFishbone` (an edge list);
the common 1D chain is a convenience specialization, `Fishbone` (a linear
backbone), that uses the same tree-state propagation path:

```python
from fishbonett.models import Fishbone

def bath(op):                                        # one bath, coupling operator op
    return Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(0, 40),
                n_modes=20, phys_dim=10).bind(op)

fb = Fishbone(sites=[0.5 * sigma_z + sigma_x] * 3,           # 3 electronic sites
              baths=[(bath(sigma_z), bath(sigma_x))] * 3,    # two baths per site
              backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2)  # nearest-neighbour
res = fb.run(dt=0.02, t_max=2.0, bond_dim=100, trunc_eps=1e-4,
             observables={"sz": sigma_z})
res.expect["sz"]         # shape (n_steps, n_sites): <sz> on every site (per-site)
```

For a non-chain topology, use `fishbonett.models.TreeFishbone` directly and pass
an edge list instead of a backbone.

The low-level state containers (`fishbonett.states.mps`,
`fishbonett.states.tree`) and evolution modules (`fishbonett.evolve.tebd`,
`fishbonett.evolve.tdvp`, `fishbonett.evolve.sitetree`,
`fishbonett.evolve.modetree`) remain available directly for finer control. More
runnable demonstrations are in
[`examples/`](examples/) — start with
[`examples/friendly_interface.py`](examples/friendly_interface.py).

## Documentation

Build the API documentation and tutorials locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Testing

```bash
pip install -e ".[test]"
pytest
```

## Citing

If `fishbonett` contributes to your research, please cite it via
[`CITATION.cff`](CITATION.cff) (once a software paper is published).

## References

- Bath discretization by Legendre polynomials — *Phys. Rev. B* **92**, 155126 (2015).
- TEDOPA chain mapping — *J. Math. Phys.* **51**, 092109 (2010).
- Boys localization for diabatization — *J. Chem. Phys.* **129**, 244101 (2008).
- Transfer-tensor method — *Phys. Rev. Lett.* **112**, 110401 (2014).

## License

Released under the [MIT License](LICENSE).

## Acknowledgment

The package name *fishbone* is by courtesy of Mr. Song Feng-Feng.
