# Tutorials

These tutorials connect numerical controls to physical conclusions. Each page
contains a complete copy-and-run program, an equation-to-code mapping, output
diagnostics, a convergence procedure, and a physical interpretation. Install the
plotting dependency with `pip install "fishbonett[docs]"` before running them.

The embedded calculations use documentation-sized numerical settings so readers
can inspect real dynamics without starting a production job. Each page states
which controls must be converged before drawing quantitative conclusions.
Generated figures and numerical summaries are recomputed during the
documentation build rather than stored in the repository.

```{toctree}
:maxdepth: 1

vibronic_dimer
nonadiabatic_spin_boson
bridge_electron_transfer
two_bath_heat_flow
```
