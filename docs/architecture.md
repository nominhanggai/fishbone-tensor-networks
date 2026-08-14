# Architecture

The package separates a simulation into a physical problem, compiled numerical
representations, a Hamiltonian frame, a tensor-network state and an integrator.
The dependency direction is one-way:

```text
Bath + System / site graph
        |
        v
CoupledBath                         model owns the system operators
        |
        v
StarBath / ChainBath / PolaronBath  immutable, operator-free coefficients
        |
        v
frame compiler                      LocalTerms, MPOFrame, gates, or factorized U
        |
        v
SimulationPlan                      prepared state + step/measure policies
        |
        v
integrator + tensor-network state   TEBD / TDVP on an MPS or tree
        |
        v
Result                              time, RDMs, expectations, bond diagnostics
```

## Ownership rules

- {py:class}`~fishbonett.bath.spec.Bath` owns environment physics and numerical
  resolution: spectral densities, temperature, domain, mode count and local Fock
  dimension.
- {py:class}`~fishbonett.bath.coupled.CoupledBath` owns the association between
  that environment and one or more system-space operators. Multiple operators are
  channels sharing one set of modes.
- {py:class}`~fishbonett.bath.compiled.StarBath` and
  {py:class}`~fishbonett.bath.compiled.ChainBath` own finite numerical
  coefficients. {py:class}`~fishbonett.bath.compiled.PolaronBath` holds the
  reweighted chain and reorganization shift required by the Lang--Firsov frame.
  They are immutable and deliberately contain no system operator. A bound bath
  caches each representation for reuse within a simulation setup.
- A frame owns transformations of the Hamiltonian and lowers compiled bath data
  into the form its compatible integrator consumes.
- {py:class}`~fishbonett.models.simulation.SimulationPlan` owns orchestration for
  one resolved method: prepared state, frame-specific lab-frame measurement, and
  either step policies or a native whole-run driver.
- An integrator advances tensors. High-level integrator paths consume compiled
  coefficients and do not decide how a spectral density is discretized.

`Bath.coupling` is deprecated. In the single-system API,
`SystemBath(coupling=...)` is authoritative; Fishbone inputs should be explicit
`bath.bind(operator)` objects. The compatibility spelling emits a
`DeprecationWarning`, and conflicting duplicate values are rejected.

## Why frames do not have one output type

Different representations expose different useful structure. A static local
Hamiltonian naturally becomes {py:class}`~fishbonett.frames.terms.LocalTerms`; a
TDVP path needs {py:class}`~fishbonett.frames.mpo.MPOFrame`; the interaction
picture can expose time-dependent gates or an exact conditional-displacement
propagator. Forcing all of these through one tensor container would hide useful
capabilities. The stable boundary is therefore the compiled bath input, while the
method registry records which frame output and integrator are compatible.

## Compatibility boundary

The high-level `run()` interface uses this pipeline. Low-level builders that
accept a spectral-density callable and a domain remain available for existing
research scripts; they compile internally and are compatibility entry points, not
the dependency direction for new high-level code.

{py:class}`~fishbonett.models.system_bath.SystemBath` now validates the physical
problem and user-facing run choices, resolves one registry row and executes its
compiled plan. Frame preparation, state construction, stepping and measurement
live in :mod:`fishbonett.models.simulation`; the former `_DRIVERS`, `_MPO_FRAMES`
and `_SWAP_FRAMES` maps no longer exist on the model.

The two former numerical monoliths are split behind stable compatibility façades:

```text
evolve.tdvp facade       -> _tdvp_driver -> _tdvp_sweeps -> _tdvp_kernels
evolve.modetree facade   -> _modetree_driver -> _modetree_sweeps
                                              -> _modetree_core
```

Kernel/core modules contain tensor algebra and topology only. The chain sweep
module owns symmetric projector splitting; the mode-tree operation module owns
TTNO application, canonicalization and edge truncation. Driver modules resolve
compatibility inputs, prepare a run and collect its trajectory. The documented
imports from `evolve.tdvp` and `evolve.modetree` remain stable.

Frame outputs are checked structurally through the runtime protocols in
{py:mod}`fishbonett.frames.capabilities`: MPO, static graph, static gates, swap
gates, and conditional displacement. A plan therefore fails during compilation
if an engine is paired with a frame that does not expose the required operations.

All three high-level model classes execute through `SimulationPlan`. The
single-system model and multi-site Fishbone models differ in their result
collector, not in where orchestration lives. `run(seed=...)` scopes randomized
linear algebra and fixed-bond padding to that one plan without changing NumPy's
global generator.
