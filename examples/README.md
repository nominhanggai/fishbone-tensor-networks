# Examples

Small, self-contained examples that use the installed `fishbonett` package. Each
demonstrates one method family and runs in seconds; e.g.

```bash
python examples/interaction_picture_spin_boson.py
```

| Example | Method |
|---|---|
| [`interaction_picture_spin_boson.py`](interaction_picture_spin_boson.py) | Interaction-picture spin-boson dynamics with a discrete multichannel bath (unified TEBD engine + leg swaps). |
| [`hsb_interaction_picture.py`](hsb_interaction_picture.py) | The keyword-constructed `SpinBosonModel` — interaction picture with respect to H_SB. |
| [`cooling_spin_boson.py`](cooling_spin_boson.py) | Finite-temperature "cooling" scheme with a chain bath and a `get_rdm` readout. |
| [`golden_rule_rate.py`](golden_rule_rate.py) | Fermi golden-rule vs Marcus electron-transfer rate from a spectral density (`fishbonett.rates`). |

## Legacy examples

The original research scripts are preserved under [`legacy/`](legacy/). They are
kept for reference, cover many additional scenarios (star geometries,
multiple-acceptor donor–acceptor models, local basis optimization, Marcus-curve
comparisons, fishbone tree networks), and may use older per-module import paths.
The curated examples above are the recommended starting point.
