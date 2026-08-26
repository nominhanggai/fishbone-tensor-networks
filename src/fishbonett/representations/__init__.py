"""Hamiltonian representations of open-system models.

The five single-bath representations are ``schrodinger-chain``,
``schrodinger-star``, ``interaction-chain``, ``polaron-chain``, and
``polaron-star``. Representation objects define the
Hamiltonian and state transformation and the resulting bath operators.

The representation objects behind these five names own the represented
Hamiltonian and materialize the numerical products supported by it: ``tdvp_mpo``,
``trotter_mpo``, or ``tebd_gates``.  They do not advance an MPS or tree;
:mod:`fishbonett.evolve` consumes those products and advances the corresponding
tensor-network state.

For the interaction representation the construction is explicit: discretize
the bath as independent star modes, rotate with respect to their diagonal free
Hamiltonian, and apply the star-to-chain transformation
(``interaction-chain``). Recovering the same
finite star by diagonalizing a truncated chain is only an equivalent numerical
route to the discretization.
"""

from fishbonett.representations.interaction import InteractionRepresentation
from fishbonett.representations.exciton import ExcitonInteractionRepresentation
from fishbonett.representations.multichannel import (
    MultichannelInteractionRepresentation,
)
from fishbonett.representations.polaron import PolaronRepresentation
from fishbonett.representations.schrodinger import (
    LocalTerms,
    SchrodingerRepresentation,
)

__all__ = [
    "InteractionRepresentation",
    "ExcitonInteractionRepresentation",
    "MultichannelInteractionRepresentation",
    "PolaronRepresentation",
    "SchrodingerRepresentation",
    "LocalTerms",
]
