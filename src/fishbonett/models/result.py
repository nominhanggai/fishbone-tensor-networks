"""What every ``run`` returns.

One container for both shapes of result, because the *model* decides the shape:

===================  ==========================  ==============================
field                single-system models         multi-site models
===================  ==========================  ==============================
``expect[name]``     ``(n_steps,)``              ``(n_steps, n_sites)`` for a
                                                 per-site spec, ``(n_steps,)``
                                                 for a single- or multi-site one
``rdm``              ``(n_steps, d, d)``         ``(n_steps, n_sites, d, d)``,
                                                 or an object array when site
                                                 dimensions differ
``max_bond``         peak bond per step          same
``method``           the method that ran         same
``meta``             ``{}``                      ``{"n_sites": n}``
===================  ==========================  ==============================

``chain``, ``star``, ``mode-tree`` and ``multichannel`` are single-system;
``comb`` and ``site-tree`` are multi-site.  See
:mod:`fishbonett.models.registry`.
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Result"]


@dataclass
class Result:
    """Result of a propagation."""
    t: np.ndarray
    expect: dict                      # observable name -> array over time
    max_bond: np.ndarray = None       # peak bond dimension per step (adaptive)
    rdm: np.ndarray = None            # spin reduced density matrix per step (T,2,2)
    method: str = ""
    meta: dict = field(default_factory=dict)
