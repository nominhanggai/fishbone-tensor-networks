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

The high-level interface ({py:mod}`fishbonett.simulate`) propagates the population
$\langle\sigma_z\rangle(t)$ of a two-level system coupled to a bath with a single
call. Declare the bath and the system, then `run`:

```python
import numpy as np
from fishbonett.simulate import Bath, SpinBoson
from fishbonett.stuff import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5),   # spectral density J(w)
            domain=(-25, 36), temperature=1.0,       # T-TEDOPA thermalization
            n_modes=40, phys_dim=20,
            discretization="orthpol")                # or the default "legendre"
model = SpinBoson(h=sigma_x, coupling=sigma_z, bath=bath)

result = model.run(dt=0.05, t_max=4.0, method="tree-tdvp2", bond_dim=200,
                   observables={"sz": sigma_z})

result.t                 # time grid
result.expect["sz"]      # <sigma_z>(t)
result.max_bond          # peak bond dimension per step
```

`method` selects the engine — `"tebd"`, `"mpo-tdvp1" | "mpo-tdvp2" | "mpo-dtdvp"`,
or `"tree-tdvp" | "tree-tdvp2" | "tree-tebd"`. Every method uses the same
`dt`/`t_max` and returns the same {py:class}`~fishbonett.simulate.Result`, so
switching engines is a one-word change.

## The fishbone geometry

{py:class}`fishbonett.fishbone_sim.Fishbone` describes a 1D chain of electronic
sites, each coupled to one bath (a comb) or two baths — one on each side of the
site (the fishbone). It is declared the same way and returns per-site data:

```python
from fishbonett.fishbone_sim import Fishbone

def bath(op):                                        # one bath, coupling operator op
    return Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(0, 40),
                n_modes=20, phys_dim=10, coupling=op)

fb = Fishbone(sites=[0.5 * sigma_z + sigma_x] * 3,           # 3 electronic sites
              baths=[(bath(sigma_z), bath(sigma_x))] * 3,    # two baths per site
              backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2)  # nearest-neighbour
res = fb.run(dt=0.02, t_max=2.0, bond_dim=100, observables={"sz": sigma_z})
res.expect["sz"]         # shape (n_steps, n_sites)
res.rdm                  # shape (n_steps, n_sites, d, d)
```

## Low-level engines

For finer control the underlying engines are available directly: build a model /
bath object (for example {py:class}`~fishbonett.model.SpinBoson` or
{py:class}`~fishbonett.model.FishBoneH`), discretize with `build(...)`, construct
the {py:class}`~fishbonett.mps.SpinBosonMPS` (or {py:class}`~fishbonett.fishbone.FishBoneNet`)
state, obtain the Trotter gates with `get_u(...)`, sweep with `update_bond(...)`,
and read out observables from `get_theta1(...)`. The high-level interface above is
a thin wrapper over exactly this loop.

See the [`examples/`](https://github.com/nominhanggai/fishbone-tensor-networks/tree/main/examples)
directory for runnable scripts — start with `friendly_interface.py`, which also
covers the interaction picture, the cooling scheme, and rate theory.
