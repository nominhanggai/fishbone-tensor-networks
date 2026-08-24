# Tutorials

Start with the molecular dimer if this is your first `fishbonett` calculation.
The tutorials then increase in numerical and conceptual difficulty:

| order | tutorial | main idea | method | expected cost |
|---:|---|---|---|---|
| 1 | {doc}`vibronic_dimer` | a structured vibration assists excitation transfer | interaction-chain Trotter MPO | four-step check: seconds; plotted calculation: 2 × 800 steps |
| 2 | {doc}`nonadiabatic_spin_boson` | strong, hot, nonadiabatic relaxation | interaction-chain Trotter MPO | four-step check: seconds; plotted calculation: 252 steps on 200 bath sites; full interval: 1,257 steps on 600 bath sites |
| 3 | {doc}`bridge_electron_transfer` | correlated non-Condon fluctuations accelerate electron transfer | interaction-chain Trotter MPO + transfer tensors | direct comparison: 3 × 100 steps; map tomography: 18 × 75 steps |
| 4 | {doc}`two_bath_heat_flow` | two reservoirs produce a nonequilibrium energy current | tree TEBD | four-step check: seconds; plotted calculation: 2 × 250 steps |

The first two pages are natural companions: the dimer emphasizes a structured
molecular vibration, while the spin--boson page emphasizes strong coupling and
finite temperature. The bridge tutorial then introduces reduced dynamical maps
and transfer tensors. Heat flow is last because it combines two baths, targeted
system--mode observables, a continuity equation, and a steady-state claim.

Before interpreting any quantitative result, read {doc}`convergence`. It gives
the shared refinement workflow for timestep, SVD threshold, local Fock space,
bath resolution, and finite-chain recurrence. Each tutorial adds only the
checks specific to its physical observable.

Each example script defaults to its `smoke` profile, a four-step API check. The
`docs` profile selects the numerical controls used for the plotted trajectory.
The `reference` profile extends the time horizon or performs the refinements
specified on that tutorial page. Wall time depends strongly on the resolved
mode count, retained bond dimensions, BLAS implementation, and hardware.

```{toctree}
:maxdepth: 1

convergence
vibronic_dimer
nonadiabatic_spin_boson
bridge_electron_transfer
bridge_electron_transfer_validation
two_bath_heat_flow
```
