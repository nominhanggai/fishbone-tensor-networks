"""Models: the physical setups you can propagate, and what each one admits.

A **model** specifies how many system sites exist, how they are connected, and
which baths couple to them. The Hamiltonian ``representation`` and tensor-network
``state_geometry`` are selected separately. A run has four public axes:

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

``multichannel`` is selected automatically when the model coupling contains a
list of operators.

:mod:`fishbonett.models.registry` lists the representations and methods available
for each model, including reasons for unavailable combinations.
``python -c "from fishbonett.models.registry import describe_taxonomy as d;
print(d())"`` prints the whole thing.

"""
from fishbonett.models.result import Result, SimulationCheckpoint
from fishbonett.models.system_bath import SystemBath
from fishbonett.models.fishbone import Fishbone, TreeFishbone
from fishbonett.targets import BathMode
from fishbonett.models.simulation import SimulationPlan, compile_plan
from fishbonett.models.registry import (
    MODELS, REPRESENTATIONS, Model, RepresentationSpec, MethodSpec,
    METHOD_REPRESENTATIONS,
    models_of, representations_of, methods_of, all_methods, model,
    methods_by_representation, representation_label, describe_taxonomy,
)

__all__ = [
    # model classes
    "SystemBath", "Fishbone", "TreeFishbone", "BathMode", "Result", "SimulationCheckpoint",
    "SimulationPlan", "compile_plan",
    # the taxonomy
    "MODELS", "REPRESENTATIONS", "Model", "RepresentationSpec", "MethodSpec",
    "METHOD_REPRESENTATIONS",
    "models_of", "representations_of", "methods_of", "all_methods", "model",
    "methods_by_representation", "representation_label", "describe_taxonomy",
]
