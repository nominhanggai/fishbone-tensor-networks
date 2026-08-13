"""Models: the physical setups you can propagate, and what each one admits.

A **model** says what is coupled to what -- how many system sites, how they are
wired, how the bath is represented.  It is the outermost of three nested
choices, and it constrains the other two:

    model  ->  frame  ->  propagator

Six models, two classes:

================  =====================  ==========================================
model             class                  what it is
================  =====================  ==========================================
``chain``         :class:`SystemBath`    1 system + 1 bath, modes chain-mapped to 1D
``star``          :class:`SystemBath`    1 system + 1 bath, no chain mapping
``mode-tree``     :class:`SystemBath`    1 system + 1 bath, modes on a binary tree
``multichannel``  :class:`SystemBath`    1 system + 1 bath, several shared couplings
``comb``          :class:`Fishbone`      N sites on a 1D backbone, baths per site
``site-tree``     :class:`TreeFishbone`  N sites in any loop-free tree, baths/site
================  =====================  ==========================================

The first four are one class because they differ only in how the bath is
represented, which ``run(method=...)`` selects.  ``multichannel`` is picked
automatically from the bath's shape rather than by a method name.

:mod:`fishbonett.models.registry` is the authority: which frames each model has,
which methods realize them, and -- for the combinations that are absent -- why.
``python -c "from fishbonett.models.registry import describe_taxonomy as d;
print(d())"`` prints the whole thing.

.. note::
   ``fishbonett.models`` previously meant what is now
   :mod:`fishbonett.frames` (the Hamiltonian builders).  In commits before that
   rename, ``models/`` is the frames package, not this one.
"""
from fishbonett.models.result import Result
from fishbonett.models.system_bath import SystemBath
from fishbonett.models.fishbone import Fishbone, TreeFishbone
from fishbonett.models.registry import (
    MODELS, FRAMES, Model, Frame, METHOD_FRAMES,
    models_of, frames_of, methods_of, all_methods, model,
    methods_by_frame, frame_label, describe_taxonomy,
)

__all__ = [
    # model classes
    "SystemBath", "Fishbone", "TreeFishbone", "Result",
    # the taxonomy
    "MODELS", "FRAMES", "Model", "Frame", "METHOD_FRAMES",
    "models_of", "frames_of", "methods_of", "all_methods", "model",
    "methods_by_frame", "frame_label", "describe_taxonomy",
]
