# fishbonett

[![CI](https://github.com/nominhanggai/fishbone-tensor-networks/actions/workflows/ci.yml/badge.svg)](https://github.com/nominhanggai/fishbone-tensor-networks/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

`fishbonett` simulates open quantum systems coupled to harmonic environments.
It turns continuous spectral densities into finite baths and propagates the
combined system-bath wavefunction with matrix-product-state (MPS) or tree tensor
network methods.

The high-level API covers a single system with one or several coupling channels
and multi-site models with baths attached to selected sites. Hamiltonians may use
Schrödinger, interaction, or polaron representations in star or chain modes.
Propagation uses TEBD, Trotter MPOs, or TDVP where supported.

## Install

The package is not yet released on PyPI. From a checkout:

```bash
python -m pip install -e ".[speed]"
```

The core package requires only NumPy and SciPy. The `speed` extra installs
`opt_einsum`, which is recommended for tree tensor networks. Other extras are:

```bash
python -m pip install -e ".[rates]"       # VEGAS rate integrator
python -m pip install -e ".[gpu-cuda12]"  # or gpu-cuda11
python -m pip install -e ".[test,docs]"   # tests and documentation
```

Set `FISHBONETT_EINSUM=numpy` before importing `fishbonett` to force the NumPy
contraction backend.

## First simulation

This example propagates a two-level system in a zero-temperature Ohmic bath.
Automatic bath resolution chooses the frequency interval and mode count for the
requested propagation time; TEDOPA then discretizes that interval.

```python
import numpy as np

from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(
    J=lambda omega: 0.15 * omega * np.exp(-omega / 3.0),
    phys_dim=8,
    discretization="tedopa",
)

model = SystemBath(
    h=0.5 * sigma_x,
    coupling=sigma_z,
    bath=bath,
)

result = model.run(
    dt=0.05,
    t_max=1.0,
    method="interaction-chain-trotter-mpo",
    trunc_eps=1e-4,
    bond_dim=None,
    observables={"sz": sigma_z},
)

print(result.t)
print(result.expect["sz"])
print(result.max_bond)
```

`trunc_eps` is the main tensor-network accuracy control. `bond_dim=None` leaves
the maximum bond dimension unrestricted; set a finite value only when a hard
memory limit is needed. Fixed-bond and dynamically expanding TDVP methods require
a finite cap. `Result` uses the same layout for every high-level method, so
representations and integrators can be compared without changing analysis code.

## Models and methods

| Physical model | Class | Typical use |
|---|---|---|
| One system, one bath | `SystemBath` | Spin-boson and molecular-vibration models |
| One system, shared bath modes with several coupling operators | `SystemBath` | Correlated multichannel noise |
| A one-dimensional backbone with baths on its sites | `Fishbone` | Electron and excitation-energy transfer |
| Any loop-free network of system sites and baths | `TreeFishbone` | Branched molecular or transport models |

For `Fishbone` and `TreeFishbone`, attach each bath explicitly with
`bath.bind(system_operator)` so the coupled site and operator are unambiguous.

A method name states the Hamiltonian representation and integrator, for example:

- `schrodinger-chain-tdvp2`
- `interaction-chain-tebd`
- `interaction-chain-trotter-mpo`
- `interaction-chain-tree-tebd`
- `polaron-chain-tdvp2`

The method registry rejects unsupported combinations and explains why they are
unavailable. Run the following to inspect the complete table for the installed
revision:

```python
from fishbonett.models.registry import describe_taxonomy

print(describe_taxonomy())
```

Baths may use explicit `domain` and `n_modes` values for convergence studies, or
leave both unset for automatic resolution. Finite temperature uses the T-TEDOPA
signed-frequency construction. Both measure-adapted TEDOPA and Gauss-Legendre
star discretizations are implemented in NumPy and SciPy.

## Documentation and examples

The [documentation](https://nominhanggai.github.io/fishbone-tensor-networks/)
starts with a complete simulation, then explains bath conventions,
representations, tensor-network geometries, and integrators. Paper-backed
tutorials cover:

- vibrationally assisted transfer in a molecular dimer;
- strong-coupling spin-boson dynamics;
- donor-bridge-acceptor electron transfer with transfer tensors; and
- heat flow through a two-bath molecular junction.

Runnable scripts are in [`examples/`](examples/). The low-level state and
evolution APIs remain available under `fishbonett.states`,
`fishbonett.representations`, and `fishbonett.evolve` for custom workflows.

To build the documentation locally:

```bash
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

## Development

```bash
python -m pip install -e ".[dev,docs]"
python -m ruff check src tests examples benchmarks docs/figures.py conftest.py
pytest
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The package is
distributed under the [MIT License](LICENSE).
