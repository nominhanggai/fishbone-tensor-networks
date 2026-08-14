"""Models: the physical setups you can propagate, and what each one admits.

A **model** says what is coupled to what -- how many system sites and how they are
wired.  Only that: how the *bath* is represented is two separate axes, the mode
``basis`` and the state ``geometry``, because those are choices of representation
rather than of physics.  A run is five of them:

    model -> frame -> basis -> geometry -> integrator

Four models, three classes:

================  =====================  ==========================================
model             class                  what it is
================  =====================  ==========================================
``system-bath``   :class:`SystemBath`    1 system + 1 bath + 1 coupling operator
``multichannel``  :class:`SystemBath`    1 system + 1 bath, several shared couplings
``comb``          :class:`Fishbone`      N sites on a 1D backbone, baths per site
``site-tree``     :class:`TreeFishbone`  N sites in any loop-free tree, baths/site
================  =====================  ==========================================

``chain``, ``star`` and ``mode-tree`` used to be listed here as models.  They were
not: the first two name a bath *basis* and the third a state *geometry*, and all
three are the same one-system/one-bath problem.  ``multichannel`` is picked
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
