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
| [`xxz_thermal_rectifier.py`](xxz_thermal_rectifier.py) | A purified four-spin XXZ junction coupled to two independently thermalized T-TEDOPA baths, with heat-current and convergence controls. |

## Legacy examples

The original research scripts are preserved under [`legacy/`](legacy/). They are
kept for reference, cover many additional scenarios (star geometries,
multiple-acceptor donor–acceptor models, local basis optimization, Marcus-curve
comparisons, fishbone tree networks), and may use older per-module import paths.
The curated examples above are the recommended starting point.
