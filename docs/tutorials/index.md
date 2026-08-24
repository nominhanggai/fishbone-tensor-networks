# Tutorials

Start with the molecular dimer if this is your first `fishbonett` calculation.
The tutorials then increase in numerical and conceptual difficulty:

| order | tutorial | main idea | method | expected cost |
|---:|---|---|---|---|
| 1 | {doc}`vibronic_dimer` | a structured vibration assists excitation transfer | interaction-chain Trotter MPO | smoke run: seconds; full comparison: tens of minutes or more |
| 2 | {doc}`nonadiabatic_spin_boson` | strong, hot, nonadiabatic relaxation | interaction-chain Trotter MPO | smoke run: seconds; paper-length profile: long production run |
| 3 | {doc}`bridge_electron_transfer` | correlated non-Condon fluctuations accelerate electron transfer | interaction-chain Trotter MPO + transfer tensors | early dynamics: minutes; map regeneration: many production runs |
| 4 | {doc}`two_bath_heat_flow` | two reservoirs produce a nonequilibrium energy current | tree TEBD | smoke run: seconds; steady-current study: tens of minutes or more |

The first two pages are natural companions: the dimer emphasizes a structured
molecular vibration, while the spin--boson page emphasizes strong coupling and
finite temperature. The bridge tutorial then introduces reduced dynamical maps
and transfer tensors. Heat flow is last because it combines two baths, targeted
system--mode observables, a continuity equation, and a steady-state claim.

Before interpreting any quantitative result, read {doc}`convergence`. It gives
the shared refinement workflow for timestep, SVD threshold, local Fock space,
bath resolution, and finite-chain recurrence. Each tutorial adds only the
checks specific to its physical observable.

The commands called without a profile use inexpensive smoke settings. Figures
on these pages use the documented `docs` profiles, and the `reference` profiles
are manual production calculations. Wall time depends strongly on the resolved
mode count, retained bond dimensions, BLAS implementation, and hardware.

Generated SVG figures and numerical summaries are build artifacts. They are
recomputed during documentation CI and are not stored in commits.

```{toctree}
:maxdepth: 1

convergence
vibronic_dimer
nonadiabatic_spin_boson
bridge_electron_transfer
bridge_electron_transfer_validation
two_bath_heat_flow
```
