# fishbonett

**fishbonett** propagates the dynamics of multi-site vibronic ("fishbone"-like)
open-quantum-system models — electron- and excitation-energy-transfer systems in
which each electronic or vibrational site is coupled to its own harmonic bath —
using matrix-product-state (MPS) and tree-tensor-network (TEBD) methods.

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
api
```

## Highlights

- One canonical TEBD engine ({py:class}`fishbonett.mps.SpinBosonMPS`) with leg
  swaps, adaptive bond dimension, optional local basis optimization, and an
  optional CuPy GPU backend.
- Self-contained TEDOPA bath discretization / chain mapping (no external Fortran
  dependencies).
- Interaction-picture, star-geometry and cooling propagation schemes, the fishbone
  tree tensor network, rate theory, and Boys-localization diabatization.

## Indices

- {ref}`genindex`
- {ref}`modindex`
