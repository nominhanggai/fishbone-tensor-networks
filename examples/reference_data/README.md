# Tutorial comparison data

These files support the paper comparisons in the scientific tutorials. They
contain data derived from published graphics or calculations performed with
the public examples; none are raw trajectory data supplied by the authors.

`dijkstra_2015_fig5_quantum_dynamics.csv` contains 41 samples of the
$\omega_0/J=4$ and 8 quantum population curves in Fig. 5 of Dijkstra *et al.*,
*J. Phys. Chem. Lett.* **6**, 627--632 (2015),
<https://doi.org/10.1021/jz502701u>. The values were mapped from the vector paths
in `fig4.eps` in the authors' arXiv source, at intervals of $0.5/J$. The axes
give $0\leq tJ\leq20$ and $0\leq P_A\leq0.7$.

`nuomin_2022_fig8_ic10.csv` contains samples of the converged IC10 population
curve in Fig. 8 of Nuomin, Beratan, and Zhang, *Phys. Rev. A* **105**, 032422
(2022), <https://doi.org/10.1103/PhysRevA.105.032422>. The values were mapped
from the vector path in the authors' arXiv source figure. They are comparison
data derived from the published graphic, not raw trajectory data supplied by
the authors.

`acharyya_2021_fig2_populations.csv` contains the solid donor, bridge, and
acceptor curves from panels (a) and (b) of Fig. 2 in Acharyya, Ovcharenko, and
Fingerhut, *J. Chem. Phys.* **153**, 185101 (2020),
<https://doi.org/10.1063/5.0027976>. The values were extracted from the vector
paths in the authors' arXiv PDF, rather than estimated from raster pixels, and
sampled every 0.05 ps. The PDF axes give 0--15 ps horizontally and 0--1
population vertically. Values were not renormalized after extraction.

`bridge_electron_transfer_ttm_maps.npz` contains the two 75-step, nine-column
dynamical maps used by the transfer-tensor validation. Each map was generated
with the public example at `dt=0.002 ps`, a 0.15 ps direct-memory window, 95
automatically resolved TEDOPA modes, local Fock dimension 6, and SVD threshold
`1e-4`. The stored signed domain is about -871 to 4313 cm^-1, and the propagation
method is `interaction-chain-trotter-mpo`. Run

```bash
python examples/bridge_electron_transfer.py \
  --generate-reference-maps examples/output/dba_ttm_maps.npz
```

to recompute the maps. The output directory is ignored because this reference
calculation is intentionally too expensive for a documentation build. The
small checked-in maps allow CI to repeat the transfer-tensor extrapolation,
fits, and pointwise comparison without treating a pre-rendered figure as a
scientific result.
