# Examples

Small, self-contained examples that use the installed `fishbonett` package. Each
demonstrates one method family and runs in seconds; e.g.

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

The four scientific tutorials accept `--profile smoke|docs|reference`. The
default `smoke` profile preserves the physical Hamiltonian but performs only a
four-step API check. The `docs` profile selects the controls used for the
plotted tutorial trajectory. The `reference` profile extends the time horizon
or refines the timestep, Fock dimension, SVD threshold, or parameter grid as
specified on the corresponding tutorial page.
