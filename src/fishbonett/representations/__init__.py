"""Hamiltonian representations of open-system models.

The six single-bath representations are ``schrodinger-chain``,
``schrodinger-star``, ``interaction-chain``, ``interaction-star``,
``polaron-chain``, and ``polaron-star``.  Each name completely specifies how the
Hamiltonian and state are transformed and which bath operators appear in the
result.  There is no second public category to combine with these names.

The representation objects behind these six names own the represented
Hamiltonian and materialize the numerical products supported by it: ``tdvp_mpo``,
``trotter_mpo``, or ``tebd_gates``.  They do not advance an MPS or tree;
:mod:`fishbonett.evolve` consumes those products and advances the corresponding
tensor-network state.
The exploratory ``coolingchain`` module is a legacy stateful utility outside
the public method registry.

For the interaction representations the construction is explicit: discretize
the bath as independent star modes, rotate with respect to their diagonal free
Hamiltonian, and either retain those operators (``interaction-star``) or apply
the star-to-chain transformation (``interaction-chain``).  Recovering the same
finite star by diagonalizing a truncated chain is only an equivalent numerical
route to the discretization.
"""
