"""Hamiltonian representations of open-system models.

The six single-bath representations are ``schrodinger-chain``,
``schrodinger-star``, ``interaction-chain``, ``interaction-star``,
``polaron-chain``, and ``polaron-star``.  Each name completely specifies how the
Hamiltonian and state are transformed and which bath operators appear in the
result.  There is no second public category to combine with these names.

The six public representation builders contain physics only.  They do not
select TEBD, TDVP, an MPS, or a tree.  :mod:`fishbonett.encodings` converts a
representation into local terms, gates, MPOs, TTNOs, or factorized propagators,
and :mod:`fishbonett.evolve` advances the corresponding tensor-network state.
The exploratory ``coolingchain`` module is a legacy stateful utility outside
the public method registry.

For the interaction representations the construction is explicit: discretize
the bath as independent star modes, rotate with respect to their diagonal free
Hamiltonian, and either retain those operators (``interaction-star``) or apply
the star-to-chain transformation (``interaction-chain``).  Recovering the same
finite star by diagonalizing a truncated chain is only an equivalent numerical
route to the discretization.
"""
