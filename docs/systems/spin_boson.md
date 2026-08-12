# Spin-boson: one system, one bath

{py:class}`~fishbonett.simulate.SpinBoson` is the basic model: a single system
coupled to one {py:class}`~fishbonett.simulate.Bath`.  Despite the name the
"spin" need not be two-level.

```python
import numpy as np
from fishbonett.simulate import Bath, SpinBoson
from fishbonett.stuff import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SpinBoson(h=sigma_x, coupling=sigma_z, bath=bath)
r = model.run(dt=0.02, t_max=2.0, method="mpo-tdvp1", bond_dim=100,
              observables={"sz": sigma_z})
```

- `h` — the system Hamiltonian, any `(d, d)` array.
- `coupling` — the system operator the bath couples to (a **list** makes it a
  multichannel bath; see {doc}`composite_multichannel`).
- `bath` — the {py:class}`~fishbonett.simulate.Bath` (see {doc}`../bath`).

`run(...)` dispatches to any {doc}`propagation method <../methods/index>` via
`method=`; they all return the same {py:class}`~fishbonett.simulate.Result`.

## System dimension and initial state

**Every** method — `tebd`, the MPO engines and the tree engines — supports an
arbitrary system dimension, a general Hermitian coupling and an arbitrary initial
state.  The `initial=` argument takes:

- `"up"` (default) — the first basis state $|0\rangle$;
- `"down"` — the second basis state $|1\rangle$;
- `"ground"` — the ground state of `h`;
- an explicit length-`d` vector (it is normalized for you).

```python
# a three-level system, started from an explicit superposition
h3 = np.diag([0.0, 1.0, 2.5])
coup3 = np.diag([1.0, 0.0, -1.0])
bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=30, phys_dim=15)
r = SpinBoson(h=h3, coupling=coup3, bath=bath).run(
        dt=0.02, t_max=1.0, method="tebd", bond_dim=80,
        initial=[1, 1, 0], observables={"n": np.diag([0, 1, 2])})
```

```{note}
`h` and `coupling` must be **Hermitian** and of the same dimension (the
interaction-picture engines diagonalize the coupling).  A *multichannel* bath (a
list of couplings) is the one case routed through the tree automatically — see
{doc}`composite_multichannel`.
```

## Result

```python
r.t                  # time grid, shape (n_steps,)
r.expect["sz"]       # each observable, over time
r.rdm                # system reduced density matrix per step, (n_steps, d, d)
r.max_bond           # peak bond per step (adaptive methods)
```

See {doc}`observables` for the full observable specification, and {doc}`fishbone`
for many-site generalizations of this same interface.
