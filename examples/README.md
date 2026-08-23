# Examples

Small, self-contained examples that use the installed `fishbonett` package. Each
demonstrates one method family and runs in seconds; e.g.

```bash
python examples/interaction_picture_spin_boson.py
```

| Example | Method |
|---|---|
| [`interaction_picture_spin_boson.py`](interaction_picture_spin_boson.py) | Interaction-picture spin-boson dynamics with a discrete multichannel bath (unified TEBD engine + leg swaps). |
| [`cooling_spin_boson.py`](cooling_spin_boson.py) | Finite-temperature "cooling" scheme with a chain bath and a `get_rdm` readout. |
| [`golden_rule_rate.py`](golden_rule_rate.py) | Fermi golden-rule vs Marcus electron-transfer rate from a spectral density (`fishbonett.rates`). |
| [`vibronic_dimer.py`](vibronic_dimer.py) | Vibrationally assisted transfer in a biased molecular dimer, with a Brownian bath on each molecule. |
| [`nonadiabatic_spin_boson.py`](nonadiabatic_spin_boson.py) | Strong-coupling spin--boson dynamics compared between interaction- and Schrödinger-chain representations. |
| [`bridge_electron_transfer.py`](bridge_electron_transfer.py) | Donor--bridge--acceptor electron transfer with diagonal and non-Condon bath fluctuations. |
| [`two_bath_heat_flow.py`](two_bath_heat_flow.py) | Heat flow through a two-level junction, measured from explicit system--bath correlations. |

The four scientific tutorials accept `--profile smoke|docs|reference`. The
default smoke profile preserves the physical Hamiltonian but performs only a
four-step engine check. Documentation builds use the docs profile; reference
profiles are deliberately manual because they include convergence variants.

## Legacy examples

The original research scripts are preserved under [`legacy/`](legacy/). They are
kept for reference, cover many additional scenarios (star geometries,
multiple-acceptor donor–acceptor models, local basis optimization, Marcus-curve
comparisons, fishbone tree networks), and may use older per-module import paths.
The curated examples above are the recommended starting point.
