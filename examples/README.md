# Examples

Small, self-contained examples that use the installed `fishbonett` package.
The default or smoke calculation runs quickly unless a script says otherwise;
for example:

```bash
python examples/interaction_picture_spin_boson.py
```

| Example | Method |
|---|---|
| [`interaction_picture_spin_boson.py`](interaction_picture_spin_boson.py) | Interaction-picture spin-boson dynamics with a discrete multichannel bath (MPS swap-network TEBD). |
| [`golden_rule_rate.py`](golden_rule_rate.py) | Fermi golden-rule vs Marcus electron-transfer rate from a spectral density (`fishbonett.rates`). |
| [`vibronic_dimer.py`](vibronic_dimer.py) | Vibrationally assisted transfer in a biased molecular dimer, with one Brownian bath coupled to the molecular energy difference. |
| [`nonadiabatic_spin_boson.py`](nonadiabatic_spin_boson.py) | Strong-coupling interaction-chain spin--boson dynamics compared with the published Figure 8 population. |
| [`bridge_electron_transfer.py`](bridge_electron_transfer.py) | Donor--bridge--acceptor electron transfer with diagonal and non-Condon bath fluctuations. |
| [`two_bath_heat_flow.py`](two_bath_heat_flow.py) | Heat flow through a two-level junction, measured from explicit system--bath correlations. |
| [`multiset_holstein.py`](multiset_holstein.py) | Franck--Condon Holstein dynamics compared between multi-set and conventional MPS TDVP. |
| [`fmo_state_layouts.py`](fmo_state_layouts.py) | Seven-site FMO dynamics with system-first, interleaved, multi-set MPS, and multi-set tree layouts. |
| [`fmo_mps_methods.py`](fmo_mps_methods.py) | Checkpointed TEBD, Trotter-MPO, TDVP1, TDVP2, and dTDVP comparisons for the system-first and interleaved FMO MPS layouts. |

The vibronic-dimer, nonadiabatic-spin-boson, electron-transfer, and heat-flow
tutorials accept `--profile smoke|docs|reference`. Their default `smoke`
profile preserves the physical Hamiltonian but performs only a four-step API
check. The `docs` profile selects the controls used for the
plotted tutorial trajectory. The `reference` profile extends the time horizon
or refines the timestep, Fock dimension, SVD threshold, or parameter grid as
specified on the corresponding tutorial page.
The Holstein method comparison instead provides `smoke`, `quick`, and
`paper-scale` profiles; its default is `quick`.
The FMO layout comparison provides `smoke` and `quick`; add
`--layouts multi-set-tree` only when the coupled tree calculation is wanted.
The FMO MPS propagation comparison provides `smoke` and `200fs`. The latter is
a production calculation and resumes each method from its most recent segment.
