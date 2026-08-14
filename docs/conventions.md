# Scientific conventions and reference checks

fishbonett uses natural units, $\hbar=1$, and evolves states with
$U(t)=\exp(-iHt)$. Site 0 is the system; bath sites follow in the order defined
by the selected frame and geometry.

## Spectral-density normalization

The package defines

$$
J(\omega)=\pi\sum_k |g_k|^2\delta(\omega-\omega_k).
$$

For a quadrature node $\omega_k$ with weight $q_k$, this gives
$g_k^2=J(\omega_k)q_k/\pi$. The reorganization energy is therefore

$$
\lambda=\frac{1}{\pi}\int \frac{J(\omega)}{\omega}\,d\omega.
$$

These factors live in {py:mod}`fishbonett.bath.conventions` and are checked by
analytic reference-value tests. Frame builders should consume those helpers or a
compiled bath instead of restating the convention.

## Interaction picture

The free bath is rotated out. An annihilation term carries
$g_k e^{-i\omega_k t}$, while its Hermitian conjugate carries the conjugate
phase. A gate over $[t,t+\Delta t]$ uses

$$
\int_t^{t+\Delta t} e^{-i\omega s} ds
=\Delta t\,e^{-i\omega(t+\Delta t/2)}
\operatorname{sinc}\!\left(\frac{\omega\Delta t}{2\pi}\right),
$$

which has the exact value $\Delta t$ at $\omega=0$. The implementation uses this
midpoint/sinc form rather than a quotient that loses precision near zero.

The sign, phase, and locality checks follow the interaction-picture Hamiltonians
in Hanggai Nuomin, David N. Beratan, and Peng Zhang,
[arXiv:2111.14308](https://arxiv.org/abs/2111.14308), and the multichannel
generalization by Hanggai Nuomin et al.,
[arXiv:2212.06099](https://arxiv.org/abs/2212.06099).

## Finite temperature

`Bath` uses the T-TEDOPA signed-frequency convention. A bare positive-frequency
density is extended to negative frequencies with Bose occupation $n_\beta$:

$$
J_\beta(\omega)=
\begin{cases}
J(|\omega|)(n_\beta+1),&\omega>0,\\
J(|\omega|)n_\beta,&\omega<0.
\end{cases}
$$

Thus compiled star and chain data already include temperature; downstream frames
must not apply a second thermofield factor.

## Validation layers

- Unit tests check factors of $\pi$, the zero-frequency phase limit, Hermiticity,
  and immutable compiled data.
- Small exact-diagonalization tests compare equivalent frames and integrators.
- `tests/characterization/all_methods_golden.py` checks every registry method
  before and after a structural change.
- `benchmarks/baseline_suite.py` records stable work metrics (Krylov calls and
  iterations, peak bond) and a reference observable; wall time is diagnostic.
