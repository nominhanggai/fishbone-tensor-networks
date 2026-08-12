# fishbonett

**fishbonett** propagates the dynamics of multi-site vibronic ("fishbone"-like)
open-quantum-system models — electron- and excitation-energy-transfer systems in
which each electronic or vibrational site is coupled to its own harmonic bath —
using matrix-product-state (MPS) and tree-tensor-network (TEBD) methods.

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
methods/index
systems/index
bath
api
```

## Highlights

- A declarative high-level interface ({py:mod}`fishbonett.simulate`,
  {py:mod}`fishbonett.treebone`): describe a bath and system as `Bath` /
  `SpinBoson` / `Fishbone` / `TreeFishbone` objects and propagate with one
  `run(dt=..., t_max=..., method=...)` call over any of the engines below.
- One canonical TEBD engine ({py:class}`fishbonett.states.mps.SpinBosonMPS`) with leg
  swaps, adaptive bond dimension, optional local basis optimization, and an
  optional CuPy GPU backend.
- Self-contained TEDOPA bath discretization / chain mapping (no external Fortran
  dependencies).
- Interaction-picture MPS/MPO/tree propagation schemes and chain cooling, the
  fishbone tree tensor network, rate theory, and Boys-localization diabatization.

## Indices

- {ref}`genindex`
- {ref}`modindex`
