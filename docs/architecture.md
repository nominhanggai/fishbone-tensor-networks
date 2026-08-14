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
StarBath / ChainBath                immutable, operator-free coefficients
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
  coefficients. They are immutable and deliberately contain no system operator.
- A frame owns transformations of the Hamiltonian and lowers compiled bath data
  into the form its compatible integrator consumes.
- {py:class}`~fishbonett.models.simulation.SimulationPlan` owns orchestration for
  one resolved method: prepared state, frame-specific lab-frame measurement, and
  either step policies or a native whole-run driver.
- An integrator advances tensors. High-level integrator paths consume compiled
  coefficients and do not decide how a spectral density is discretized.

`Bath.coupling` remains a compatibility field for Fishbone inputs. In the
single-system API, `SystemBath(coupling=...)` is authoritative; if both spellings
are present they must agree. New lower-level composition should use
`bath.bind(operator)` explicitly.

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

Kernel/core modules contain tensor algebra and topology only. Sweep modules own
one symmetric traversal. Driver modules resolve compatibility inputs, prepare a
run and collect its trajectory. Existing imports from `evolve.tdvp` and
`evolve.modetree` continue to work.
