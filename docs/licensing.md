# License and numerical provenance

fishbonett is distributed under the [MIT License](../LICENSE). The package's
MPO, TDVP, balanced-tree, bath-discretization, and randomized linear-algebra
implementations are original implementations of published algorithms and
standard numerical constructions:

- Hamiltonian MPOs are compiled from sums of local operator products and reduced
  by exact QR/SVD bond minimization.
- Chain evolution uses reorthogonalized Arnoldi exponential actions and symmetric
  one- or two-site projector splitting.
- Balanced mode trees apply the commuting conditional-displacement exponential
  as a graph-generic tree tensor-network operator, followed by canonicalization
  and Schmidt truncation.
- Measure-adapted bath quadrature uses a composite positive measure, symmetric
  Lanczos recurrence coefficients, and Golub--Welsch diagonalization.
- Randomized SVD uses a randomized range finder with stabilized power iterations.

The implementation modules cite the scientific papers that define the numerical
methods. Those citations document the algorithms and conventions needed to
understand and reproduce the calculations.
