# Observables: per-site, single-site and multi-site

On the fishbone engines each entry of `observables` is one of three forms — mix
them freely in a single `run`:

```python
res = fb.run(dt=0.02, t_max=1.0, bond_dim=80, observables={
    "sz":   sigma_z,                             # bare op -> measured on EVERY site
    "sz2":  (sigma_z, 2),                        # (op, i) -> just site 2
    "zz13": (np.kron(sigma_z, sigma_z), (1, 3)), # (op, (i, j)) -> two-site correlation
})
res.expect["sz"]     # (n_steps, n_sites)   -- per-site
res.expect["sz2"]    # (n_steps,)           -- one site
res.expect["zz13"]   # (n_steps,)           -- <sigma_z(1) sigma_z(3)>
```

- **bare operator** `op` (a `(d, d)` array) — measured on **every** site whose
  dimension matches; `expect[name]` is `(n_steps, n_sites)`, with `NaN` where the
  operator dimension does not match a site.
- **`(op, i)`** — the operator on the single site `i`; `expect[name]` is
  `(n_steps,)`.
- **`(op, (i, j, ...))`** — a composite operator on those sites, where `op` is
  `(D, D)` with `D` the product of the site dimensions in the given order (e.g. a
  two-site correlation $\sigma_z \otimes \sigma_z$); `expect[name]` is
  `(n_steps,)`.

Site indices refer to the **electronic sites** `0 … n_sites − 1`, in the order you
passed them to `Fishbone`/`TreeFishbone`.

## How multi-site operators are evaluated

A multi-site expectation value needs the *joint* reduced density matrix of the
requested sites.  Contracting the whole tree would be exponentially expensive, so
the engine contracts only the **minimal subtree spanning the requested sites**.
With the orthogonality centre moved inside that subtree, every tensor outside it is
isometric, so the bonds leaving the subtree contract to the identity and never
enter the calculation — the cost depends only on the subtree, not on the size of
the bath.  This is exposed on the low-level state as
{py:meth}`TensorNetwork.joint_rdm <fishbonett.states.network.TensorNetwork.joint_rdm>` and
{py:meth}`TensorNetwork.expectation <fishbonett.states.network.TensorNetwork.expectation>`.

All three forms work identically through the 1D
{py:class}`~fishbonett.models.fishbone.Fishbone`, since it delegates to the same engine.
A 1D MPS has a loop-free tensor graph, so these methods are available on
{py:class}`~fishbonett.states.mps.SystemBathMPS` too — its spanning "subtree" is
just the stretch of chain between the outermost requested sites.

## Per-site reduced density matrices

Independently of `observables`, `result.rdm` always holds the single-site reduced
density matrix of every site at every step:

```python
res.rdm.shape        # (n_steps, n_sites, d, d) for uniform site dimension d
res.rdm[t, i]        # d x d reduced density matrix of site i at step t
```

For a fishbone whose sites have **different** dimensions (e.g. a spin site and a
vibration site, see {doc}`composite_multichannel`), `result.rdm` is instead an
object array of per-site matrices, and a bare per-site observable is filled only
where its dimension matches (`NaN` otherwise).
