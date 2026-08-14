# API reference

Start from `from fishbonett import Bath, SystemBath, Truncation`.

A calculation is four **independent** choices: **model** (what is coupled to what —
`models`), **frame** (how the Hamiltonian is written down — a picture *and* a mode
basis — `frames`), **geometry** (the graph the state lives on — `states`), and
**integrator** (how a step is taken — `evolve`).
{py:mod}`fishbonett.models.registry` records which combinations exist and why the
rest do not.

You can say them directly, which is usually clearer than remembering a name:

```python
sb.run(dt=0.02, t_max=2.0, frame="interaction-star", geometry="path",
       integrator="tdvp2")
sb.run(dt=0.02, t_max=2.0, method="mpo-ip-tdvp2")     # the same run, named
```

A frame is a picture × a mode basis, so there are six: `schrodinger-chain`,
`schrodinger-star`, `interaction-chain`, `interaction-star`, `polaron-chain` and
`polaron-star` (the last unimplemented — possible, but it dresses every mode at
once). The two bases are one orthogonal transform apart, so they are the same
physics at different cost.
`integrator` is `"tebd"`, `"tdvp1"`, `"tdvp2"`, `"dtdvp"` or `"trotter-mpo"`.  Omit
an axis and it is inferred when only one combination fits; when several do, the
error lists them.  `registry.describe_taxonomy()` prints the whole table.  `bath`
turns a spectral density into the chain parameters every
model starts from, and `linalg`/`operators` hold the shared numerics.  The system
sits at **site 0** throughout, with the bath modes following, nearest first.

## Models — the physical setups

The four models and the taxonomy that relates them to frames and propagators.  A
model's two inputs are a `Bath` (the environment, below) and a
{py:class}`~fishbonett.system.System` — any Hermitian `h`, any Hermitian coupling
and an initial state, validated once so that no frame re-derives it.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.models.registry
   fishbonett.models.system_bath
   fishbonett.models.fishbone
   fishbonett.system
   fishbonett.models.simulation
   fishbonett.models.propagate
   fishbonett.models.result
```

`simulation` is the orchestration boundary: it compiles a resolved registry row
into a `SimulationPlan` containing the prepared frame, state, stepping and
measurement policy. `propagate` owns the shared step/measure/collect loop. What
varies is explicit — the **integrator** supplies `step`, the **frame** supplies the
lab-frame RDM policy (dressed frames undress their own observable), and the
**state** supplies `peak_bond`.

## Geometry: the state ansätze

A chain is a tree, so both containers are the same object: tensors on a loop-free
graph.  `TensorNetwork` holds everything that follows from loop-freeness — topology,
the orthogonality centre, and the reduced density matrices read off it — and asks
each container for `tensor` / `set_tensor` / `neighbours` so it never has to know
whether the physical leg is stored in the middle (`vL, p, vR`) or last.  What stays
per-container is the gauge policy and the gate splitting: `SystemBathMPS` is in
Vidal form with LBO and a GPU path, `TreeTensorNetwork` is mixed-canonical.  These
hold tensors only; the models that drive them are above.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.states.network
   fishbonett.states.mps
   fishbonett.states.tree
```

## Propagators

The long-standing `tdvp` and `modetree` modules remain the public import paths,
but each is now a small compatibility façade. Internally their implementations
are split into tensor kernels/core topology, symmetric sweeps, and whole-run
drivers. Dependencies point only upward through those layers; bath resolution is
confined to drivers and model planning.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.evolve.tebd
   fishbonett.evolve.sitetree
   fishbonett.evolve.mpo_apply
   fishbonett.evolve.tdvp
   fishbonett.evolve.modetree
```

## Frames

One Hamiltonian, several representations — see {doc}`/methods/index` for which
propagator suits which frame.  Frames consume compiled bath coefficients and emit
the operator form their compatible integrators require:
{py:class}`~fishbonett.frames.terms.LocalTerms` for static tree TEBD, an
{py:class}`~fishbonett.frames.mpo.MPOFrame` for TDVP, or a time-dependent gate or
factorized-propagator builder.  The bath-compilation boundary is common even where
the operator representation necessarily differs.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.frames.terms
   fishbonett.frames.gates
   fishbonett.frames.schrodinger
   fishbonett.frames.interaction_picture
   fishbonett.frames.polaron
   fishbonett.frames.multichannel
   fishbonett.frames.coolingchain
   fishbonett.frames.mpo
```

## The bath: specification, discretization, chain mapping

`Bath` is environment physics plus resolution settings.  `CoupledBath` binds it to
model-owned system operators.  Compilation then produces immutable, operator-free
`StarBath` or `ChainBath` coefficients before a frame builds gates or an MPO.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.bath.spec
   fishbonett.bath.coupled
   fishbonett.bath.compiled
   fishbonett.bath.chain
   fishbonett.bath.legendre
   fishbonett.bath.tedopa
   fishbonett.bath.lanczos
   fishbonett.bath.recurrence
   fishbonett.bath.auto
```

## Operators, linear algebra and spectral densities

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.operators
   fishbonett.linalg
   fishbonett.spectral_densities
```

## Rate theory and diabatization

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.rates
   fishbonett.diabatization
```
