"""Models: the physical setups you can propagate, and what each one admits.

A **model** says what is coupled to what -- how many system sites and how they are
wired. Only that: the complete Hamiltonian ``representation`` and the tensor-network
``state_geometry`` are separate choices. A run has four public axes:

    model -> representation -> state_geometry -> integrator

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
not: the first two belong inside a complete representation name and the third is
a tensor-network state geometry. ``multichannel`` is picked
automatically from a list of model coupling operators rather than by a method name.

:mod:`fishbonett.models.registry` is the authority: which representations each model has,
which methods realize them, and -- for the combinations that are absent -- why.
``python -c "from fishbonett.models.registry import describe_taxonomy as d;
print(d())"`` prints the whole thing.

.. note::
   ``fishbonett.models`` previously meant what is now
   :mod:`fishbonett.representations` (the Hamiltonian builders).  In commits before that
   rename, ``models/`` is the representations package, not this one.
"""
from fishbonett.models.result import Result
from fishbonett.models.system_bath import SystemBath
from fishbonett.models.fishbone import Fishbone, TreeFishbone
from fishbonett.models.simulation import SimulationPlan, compile_plan
from fishbonett.models.registry import (
    MODELS, REPRESENTATIONS, Model, Representation, METHOD_REPRESENTATIONS,
    models_of, representations_of, methods_of, all_methods, model,
    methods_by_representation, representation_label, describe_taxonomy,
)

__all__ = [
    # model classes
    "SystemBath", "Fishbone", "TreeFishbone", "Result",
    "SimulationPlan", "compile_plan",
    # the taxonomy
    "MODELS", "REPRESENTATIONS", "Model", "Representation", "METHOD_REPRESENTATIONS",
    "models_of", "representations_of", "methods_of", "all_methods", "model",
    "methods_by_representation", "representation_label", "describe_taxonomy",
]
