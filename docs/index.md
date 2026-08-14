# fishbonett

**fishbonett** propagates the quantum dynamics of open systems coupled to harmonic
baths — from a single two-level system in a bath, to multi-site vibronic
("fishbone"-like) models for electron and excitation-energy transfer in which every
electronic or vibrational site carries its own bath — using matrix-product-state
(MPS) and tree-tensor-network methods.

The physical setting throughout is

$$
H = H_{\mathrm{sys}} + O \otimes \sum_k g_k (b_k + b_k^{\dagger})
    + \sum_k \omega_k b_k^{\dagger} b_k ,
$$

an arbitrary Hermitian system Hamiltonian $H_{\mathrm{sys}}$ coupled through an
arbitrary Hermitian operator $O$ to a bath specified only by its spectral density
$J(\omega)$ (and a temperature). Everything else — discretizing the bath, mapping
it to a chain, choosing a frame, propagating, reading observables — is what this
package does.

## How to read this

The main sections are meant to be read in this order, though each stands alone:

1. **{doc}`getting_started`** — install, then a complete first simulation in a
   dozen lines. Start here.
2. **{doc}`models/index`** — the four models: what you can express, from a single
   spin-boson system to arbitrary loop-free trees of sites and baths, plus how to
   define observables.  The model you pick decides which frames and propagators
   are available.
3. **{doc}`bath`** — how a continuous $J(\omega)$ becomes a finite chain of modes:
   discretization, the TEDOPA chain mapping, finite temperature (thermofield), and
   the automatic `domain` / `n_modes` choices, with numerical evidence that they
   are faithful.
4. **{doc}`methods/index`** — the propagation methods: for each model, the frame
   (interaction picture, polaron, Schrödinger) and the integrator (TEBD, exact
   MPO propagator, TDVP), with the theory behind each.
5. **{doc}`architecture`** — the ownership and dependency boundaries used by the
   implementation.

{doc}`api` is the generated reference for every public module.

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

getting_started
models/index
bath
methods/index
architecture
api
```

## Highlights

- **A declarative interface.** Describe a bath and a system as `Bath` /
  `SystemBath` / `Fishbone` / `TreeFishbone` objects and propagate with a single
  `run(dt=..., t_max=..., method=...)` call. Every method takes the same arguments
  and returns the same `Result`, so switching engines — or cross-validating one
  against another — is a one-word change.
- **Many frames, one model.** The same physical model can be propagated in the
  Schrödinger picture, the interaction picture, or the polaron frame; the frame is
  what determines how much entanglement the state has to carry. Which frames a
  model admits — and why the others are absent — is recorded in
  {py:mod}`fishbonett.models.registry`.
- **Sensible automation.** The bath `domain` and mode count can be derived from the
  spectral density and the propagation time; truncation is driven by an accuracy
  threshold, with the bond dimension left unbounded unless you cap it.
- **Self-contained.** TEDOPA discretization and chain mapping are implemented in
  the package — no external Fortran dependencies — with an optional CuPy GPU
  backend and an optional `opt_einsum` fast path.
- Also included: chain cooling, the fishbone tree tensor network, golden-rule /
  Marcus rate theory, and Boys-localization diabatization.

## Indices

- {ref}`genindex`
- {ref}`modindex`
