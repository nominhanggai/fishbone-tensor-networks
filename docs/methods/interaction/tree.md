# Interaction-chain on a balanced mode tree

`interaction-chain-tree-tebd` uses the same `interaction-chain` Hamiltonian as
the 1D MPS methods, but stores the state on a binary tree tensor network.

The representation contains no mode--mode terms: after the free star bath is
removed, the time-dependent coupling is transformed star-to-chain and every
resulting mode couples only to the system. This makes a balanced tree possible
without introducing long-range bath interactions.

## Why the tensor-network geometry can help

On a 1D MPS, many cuts can separate the system from a large part of the bath. On
a binary tree tensor network, only the logarithmically many edges near the root separate large
subtrees. When system-mediated correlation dominates, this can reduce the number
of expensive bonds.

The tree is a state-storage and operator-application choice. It does not define a
different Hamiltonian representation.

## Method

`interaction-chain-tree-tebd` applies a second-order conditional-coupling tree
operator and truncates every edge by its relative Schmidt spectrum. There is no
public TDVP implementation for this balanced mode-tree state.

All return laboratory-system RDMs and report the maximum retained tree bond per
step.

```python
r = model.run(
    dt=0.02,
    t_max=2.0,
    representation="interaction-chain",
    state_geometry="binary-tree",
    integrator="tebd",
    bond_dim=120,
    trunc_eps=1e-5,
)
```

Use {doc}`/models/fishbone` for a different physical model in which multiple
system sites themselves form a tree and carry their own baths.
