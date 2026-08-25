# Converging open-system tensor-network dynamics

A smooth population curve is not evidence of convergence. A reliable result is
stable under independent refinements of the time integrator, tensor-network
compression, local oscillator basis, and finite bath. This page gives the
workflow shared by all tutorials.

## Start from the observable

Choose the quantity and time window needed for the physical conclusion before
refining numerical controls. Compare complete trajectories on a common time
grid and report a norm such as

$$
\epsilon_O=\max_{t\le t_{\max}}
|\langle O(t)\rangle_{\mathrm{refined}}-
 \langle O(t)\rangle_{\mathrm{base}}|.
$$

Endpoint agreement can hide errors in transient populations, currents, or
coherences. Normalization is a necessary invariant, but two inaccurate runs can
both preserve it.

## Use the SVD threshold as the primary cutoff

Set `bond_dim=None` or a generous safety cap and control
compression with `trunc_eps`. If the maximum bond repeatedly reaches a finite
cap, the result is bond-limited and tightening `trunc_eps` cannot help.

Compare at least two thresholds. Retained bond dimensions and discarded weight
are diagnostics; acceptance is based on stability of the physical observable.

## Refine timestep and truncation together

Halving `dt` doubles the number of propagation and compression steps over the
same physical interval. A fixed per-step SVD threshold can therefore introduce
more accumulated truncation error in the smaller-step calculation, masking the
reduction in integration error.

Use this sequence:

1. tighten `trunc_eps` at the original timestep until the observable is stable;
2. halve `dt` and use an equally tight or tighter threshold;
3. tighten the threshold once more at the smaller timestep;
4. accept timestep convergence only when both refined comparisons agree.

This is a coupled convergence test, not a rule that one fixed numerical factor
must relate `dt` and `trunc_eps`.

## Converge the local oscillator basis

Increase `phys_dim` while holding the bath discretization and propagation
controls fixed. Population near the highest Fock level is a warning, but a small
mean occupation alone does not guarantee convergence of a strongly displaced
or non-Gaussian mode.

Local-basis optimization changes how the basis is represented; it does not
remove the need to check the retained local dimension and its tolerance.

## Converge the finite bath

The frequency domain and number of modes answer different questions:

- expand `domain` to test missing spectral weight and frequency cutoffs;
- increase `n_modes` at fixed domain to test quadrature resolution;
- extend the chain when increasing `t_max`, so boundary reflections remain
  beyond the observation window; and
- with automatic resolution, record the resolved domain and mode count in the
  result metadata.

At finite temperature, negative thermofield frequencies encode thermal
absorption. They are part of the represented bath but are not additional
positive-frequency contributions to the physical reorganization energy.

## Add method-specific checks

The shared tests above do not replace physics-specific validation:

- population transfer: compare every state population and any paper reference,
  as in {doc}`vibronic_dimer`;
- transfer tensors: converge the direct dynamical maps and retained memory, as
  in {doc}`bridge_electron_transfer_validation`;
- heat current: check both directional currents, energy continuity, a
  zero-temperature-bias control, and absence of finite-chain recurrence, as in
  {doc}`two_bath_heat_flow`;
- physical coherences in transformed representations: transform the
  observable back before comparison, as illustrated by the representation
  comparison in {doc}`nonadiabatic_spin_boson`.

## Report enough information to reproduce the result

Record `dt`, `t_max`, `trunc_eps`, `bond_dim`, local dimensions, bath domain,
mode count, temperature convention, initial state, representation, state
geometry, and integrator. For a publication figure, retain the observable
difference from each refinement, not only the settings of the final run.
