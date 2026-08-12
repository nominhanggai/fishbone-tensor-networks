# Getting started

## Installation

```bash
pip install -e .                 # from a checkout
pip install -e ".[gpu]"          # optional CuPy GPU truncation
pip install -e ".[symbolic]"     # optional sympy-based spectral densities
pip install -e ".[rates]"        # optional vegas Monte-Carlo integrator
pip install -e ".[test,docs]"    # development
```

Core dependencies are `numpy`, `scipy` and `opt_einsum`; Python ≥ 3.10 is required.

## A first simulation

The following propagates the population $\langle\sigma_z\rangle(t)$ of a two-level
system coupled to a small discrete multichannel bath, in the interaction picture.

```python
import numpy as np
from fishbonett.backwardSpinBosonMultiChannel import SpinBoson
from fishbonett.mps import SpinBosonMPS
from fishbonett.stuff import sigma_x, sigma_z

freq = [10.0, 25.0, 40.0]
coup_mat = [np.diag(c) for c in [(5, -5), (-3, 3), (2, -2)]]
n_boson = 2 * len(freq)
pd = [10] * n_boson + [2]

eth = SpinBoson(pd, coup_mat=coup_mat, freq=freq, temp=100.0)
eth.h1e = 130.0 * sigma_x + np.diag([0.0, -200.0])
eth.build(n=0)

etn = SpinBosonMPS(pd)
etn.B[-1][0, 0, 0] = 1.0

dt, chi, eps = 1e-3, 40, 1e-6
for tn in range(30):
    u1, u2 = eth.get_u(2 * tn * dt, 2 * dt, factor=2)
    etn.U = u1
    for j in range(n_boson - 1, 0, -1):
        etn.update_bond(j, chi, eps, swap=1)
    etn.update_bond(0, chi, eps, swap=0)
    etn.update_bond(0, chi, eps, swap=0)
    etn.U = u2
    for j in range(1, n_boson):
        etn.update_bond(j, chi, eps, swap=1)
    theta = etn.get_theta1(n_boson)
    rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
    print(np.einsum('ij,ji', rho, sigma_z).real)
```

## Common workflow

Most simulations follow the same recipe:

1. Choose physical dimensions `pd = boson_dims + [system_dim]`.
2. Build a model / bath object and set its spectral density and coupling
   operators, then `build(...)` to discretize the bath (Gauss–Legendre
   discretization followed by a Lanczos chain mapping).
3. Construct the {py:class}`~fishbonett.mps.SpinBosonMPS` state and set the initial
   condition.
4. Obtain the Trotter gates with `get_u(...)` and sweep with `update_bond(...)`.
5. Read out observables from `get_theta1(...)`.

See the [`examples/`](https://github.com/nominhanggai/fishbone-tensor-networks/tree/main/examples)
directory for runnable scripts covering the interaction picture, the cooling
scheme, and rate theory.
