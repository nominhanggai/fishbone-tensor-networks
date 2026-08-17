# Can a four-spin molecular junction rectify heat?

This tutorial studies a segmented XXZ spin chain between two independent bosonic
reservoirs. It needs an initially thermal interacting wire, two finite-temperature
bath purifications, local heat currents, forward and reverse temperature biases,
and an equal-temperature control.

The segmented Hamiltonian is adapted from the heat-diode mechanism studied by
[Balachandran *et al.*](https://arxiv.org/abs/1809.01917). Their reservoirs are
Markovian Lindblad baths. Here they are explicit T-TEDOPA chains, so this tests
finite-time unitary system-bath dynamics rather than reproducing their steady-state
numbers.

## The physical model

Four spin-$1/2$ sites form two different molecular segments,

$$
H_S=\sum_i B S_i^z
+(S_0^xS_1^x+S_0^yS_1^y+\Delta S_0^zS_1^z)
+J_c(S_1^xS_2^x+S_1^yS_2^y)
+S_2^xS_3^x+S_2^yS_3^y.
$$

We use $\Delta=4$, $J_c=0.5$, and $B=0.5$. The left segment is strongly
interacting, the right segment is an XX wire, and the interface is weaker than
either intrasegment bond. Identical endpoint operators $\sigma_x$ connect the
wire to identical Ohmic-Gaussian spectral densities. Only the bath temperatures
are exchanged between forward and reverse bias.

```{eval-rst}
.. literalinclude:: ../../examples/xxz_thermal_rectifier.py
   :language: python
   :pyobject: RectifierConfig

.. literalinclude:: ../../examples/xxz_thermal_rectifier.py
   :language: python
   :pyobject: make_rectifier
```

The bath frequency domains and mode counts are automatic and the measure-adapted
TEDOPA discretization is explicit. Production runs use `trunc_eps=1e-5`;
`max_bond=512` is a safety limit, not the convergence knob.

The package-wide default of `1e-4` is a sensible starting point, but it is not a
substitute for convergence. Here, halving the time step while retaining `1e-4`
applies the truncation twice as often and discards the smaller entanglement
increments. Tightening to `1e-5` restores time-step convergence, so that is the
threshold used for the reported trajectories.

## Purifying the interacting wire

A list of four local vectors cannot represent the Gibbs state of an interacting
chain. `GibbsPurification` diagonalizes the 16-dimensional physical Hamiltonian,
forms $|\sqrt{\rho_\beta}\rangle$, groups each physical spin with an inert ancilla,
and factorizes the result into an exact MPS. Hamiltonians and observables act as
$O\otimes I$ on each supersite.

This is exact for the isolated wire. Attaching the baths is still a contact quench:
a product of wire and bath Gibbs states is not the interacting global Gibbs state.
The equal-temperature trajectory is therefore a measured control, not something
assumed to have zero current at every time.

In `make_rectifier`, `lift_site_operator(sigma_x, 0)` constructs the coupling
matrix for purified site 0; it does not attach a bath. The keys in
`baths={0: ..., 3: ...}` perform the attachment, so the left bath acts on endpoint
0 and the right bath on endpoint 3. This separates the operator definition from
the model topology.

## Current from continuity

For a cut to the right of site $i$, let $H_{\le i}$ contain the site and bond terms
entirely to its left. The energy leaving that region is

$$
j_i=-\frac{dH_{\le i}}{dt}=i[H_{\le i},V_{i,i+1}].
$$

`energy_current_operator` reduces the commutator to two or three sites. In a
quasi-steady window the currents at all three cuts must agree. A single flux curve
is not sufficient: unequal cut currents mean energy is accumulating in the wire.

## Propagation and continuation

```{eval-rst}
.. literalinclude:: ../../examples/xxz_thermal_rectifier.py
   :language: python
   :pyobject: run_bias
```

`bath_horizon` resolves the automatic baths for the complete intended time before
the first step. `SimulationCheckpoint` can resume that same resolved Hamiltonian,
but cannot change a bath temperature or enlarge the bath. Expensive observables
are sampled every 0.1 time unit through `observe_every`; TEBD still steps by 0.02.

The documentation build runs forward bias, reverse bias, an equal-temperature
control, and both biases with half the time step, a tighter SVD threshold, a larger
boson Fock space, and baths resolved for a longer light cone.

![Forward, reverse, control, and convergence](../img/xxz_thermal_rectifier.png)

```{include} ../_generated/xxz_thermal_rectifier.md
```

## What may be concluded

A finite explicit bath does not create an infinite-time NESS. A current may be
reported only in a pre-recurrence interval where its drift is small, all internal
cuts agree, the equal-temperature current is negligible, and the time-step, SVD,
Fock-space, and bath-resolution checks agree. The generated result applies these
rules automatically. When they fail, the correct conclusion is transient heat
propagation—not a thermal-diode ratio.
