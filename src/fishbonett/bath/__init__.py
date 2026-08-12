"""Bath discretization and TEDOPA chain mapping.

* :mod:`fishbonett.bath.legendre` -- Gauss-Legendre star discretization
  (``get_vn_squared``);
* :mod:`fishbonett.bath.orthpol` -- measure-adapted ORTHPOL / Jacobi star;
* :mod:`fishbonett.bath.lanczos` -- Lanczos tridiagonalization (star -> chain);
* :mod:`fishbonett.bath.recurrence` -- orthogonal-polynomial recurrence
  coefficients (the historical ``py-orthpol`` replacement).
"""
from fishbonett.bath.legendre import get_vn_squared, get_legendre_recursion
from fishbonett.bath.orthpol import (
    get_vn_squared_orthpol, make_orthpol_discretizer,
)
from fishbonett.bath.lanczos import lanczos
from fishbonett.bath.recurrence import recurrenceCoefficients

__all__ = [
    "get_vn_squared", "get_legendre_recursion",
    "get_vn_squared_orthpol", "make_orthpol_discretizer",
    "lanczos", "recurrenceCoefficients",
]
