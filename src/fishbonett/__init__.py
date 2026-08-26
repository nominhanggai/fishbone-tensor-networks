"""fishbonett: tensor-network propagation of vibronic open-quantum-system dynamics.

``fishbonett`` propagates the dynamics of multi-site vibronic model systems with
"fishbone"-like configurations -- electron- and excitation-energy-transfer models
in which each electronic or vibrational site is coupled to its own bath -- using
matrix-product-state (MPS) and tree-tensor-network (TTN) ansaetze.

Start here
----------
Three names cover the usual case::

    from fishbonett import Bath, SystemBath, Truncation

:class:`~fishbonett.bath.spec.Bath`
    *What* the bath is: a spectral density, a frequency window, how many modes,
    and how big each mode's Fock space. Coupling operators belong to the model.
:class:`~fishbonett.models.system_bath.SystemBath`
    *What* to propagate: a system Hamiltonian ``h`` plus that bath.  Call
    ``model.run(dt=..., t_max=..., method=...)`` and read the :class:`Result`.
:class:`~fishbonett.linalg.Truncation`
    *How accurately*: ``eps`` (the accuracy knob) and ``max_bond`` (an optional
    hard cap, ``None`` = unlimited).

How the package is laid out
---------------------------
A calculation has four public selection axes:

* **model** -- what is coupled to what and how the system sites are wired;
  see :mod:`fishbonett.models`.
* **representation** -- how the Hamiltonian is written and which terms are
  rotated away; see :mod:`fishbonett.representations`.
* **state_geometry** -- the tensor-state layout; see :mod:`fishbonett.states`.
* **integrator** -- how a step is taken, using Trotter gates, an MPO, or TDVP;
  see :mod:`fishbonett.evolve`.

They do not combine freely; the registry records compatible combinations. A
static ``H`` admits TDVP on an MPO built once; a time-dependent one must rebuild
every step; and only the single-channel interaction representation has commuting
mode-coupling terms,
which lets ``interaction-chain-trotter-mpo`` write that propagator in closed
form.

:mod:`fishbonett.models.registry` lists supported combinations and reasons for
unavailable combinations::

    from fishbonett.models.registry import describe_taxonomy
    print(describe_taxonomy())

The two **inputs** to a model are a :class:`~fishbonett.bath.spec.Bath` (what the
environment is) and a :class:`~fishbonett.system.System` (what is coupled to it --
any Hermitian ``h`` of any dimension, any Hermitian coupling, and an initial state).
:mod:`fishbonett.representations` discretizes a resolved bath into the finite
coefficients its Hamiltonian requires. A
:class:`~fishbonett.bath.coupled.CoupledBath` associates a model's bath with its
system operator;
:mod:`fishbonett.linalg` and :mod:`fishbonett.operators` hold the shared numerics.

Five models, four classes: :class:`~fishbonett.models.system_bath.SystemBath` for one
system and one bath (``system-bath``, ``multichannel``),
:class:`~fishbonett.models.exciton.ExcitonBath` for a single excitation with
independent local baths (``exciton-bath``),
:class:`~fishbonett.models.fishbone.Fishbone` for a 1D chain of sites (``comb``), and
:class:`~fishbonett.models.fishbone.TreeFishbone` for any loop-free tree of sites
(``site-tree``). Hamiltonian ``representation`` and ``state_geometry`` are
selected through ``run``.

:class:`~fishbonett.states.mps.SystemBathMPS` and
:class:`~fishbonett.states.tree.TreeTensorNetwork` derive from
:class:`~fishbonett.states.network.TensorNetwork`, which manages topology,
orthogonality centres, and reduced density matrices. The MPS orders its legs as
``(vL, p, vR)`` and the tree as ``(bonds..., p)``. See :mod:`fishbonett.states`
for canonical-form and gate-splitting details.
:class:`~fishbonett.states.multiset.MultiSetMPS` instead holds one independent
environmental MPS per exact system-basis state.
:class:`~fishbonett.states.multiset_tree.MultiSetTreeTensorNetwork` applies the
same outer expansion to independently truncated bath trees.

Site ordering: the system is **site 0** and the bath modes follow, nearest
first.

Public API
----------
* **High-level interface:** :class:`~fishbonett.bath.spec.Bath`,
  :class:`~fishbonett.models.system_bath.SystemBath`,
  :class:`~fishbonett.models.exciton.ExcitonBath`,
  :class:`~fishbonett.models.result.Result`, and
  :class:`~fishbonett.linalg.Truncation`. For several system sites use
  :class:`~fishbonett.models.fishbone.Fishbone` or
  :class:`~fishbonett.models.fishbone.TreeFishbone`.
* **State ansaetze:** :class:`~fishbonett.states.mps.SystemBathMPS` for a 1D MPS,
  :class:`~fishbonett.states.multiset.MultiSetMPS` for a multi-set MPS,
  :class:`~fishbonett.states.multiset_tree.MultiSetTreeTensorNetwork` for
  multi-set bath trees, and
  :class:`~fishbonett.states.tree.TreeTensorNetwork` for a general tree.
* **Representations and numerical products:** the Schrödinger, interaction,
  polaron, and multichannel builders in :mod:`fishbonett.representations`.
  See :doc:`the methods guide </methods/index>` for compatible propagators.
* **Bath discretization:**
  :func:`~fishbonett.bath.chain.get_bath_nn_parameters`,
  :func:`~fishbonett.bath.chain.get_coupling`,
  :func:`~fishbonett.bath.legendre.get_vn_squared`,
  :func:`~fishbonett.bath.lanczos.lanczos`, and
  :func:`~fishbonett.bath.recurrence.recurrence_coefficients`.
* **Operators and spectral densities:** :mod:`fishbonett.operators` and
  :mod:`fishbonett.spectral_densities`.
"""
from importlib.metadata import PackageNotFoundError, version as _version

from fishbonett.states.mps import SystemBathMPS
from fishbonett.states.multiset import MultiSetMPS
from fishbonett.states.multiset_tree import MultiSetTreeTensorNetwork
from fishbonett.evolve.tdvp import run_mpo_hamiltonian
from fishbonett.evolve.modetree import run_tree_tebd
from fishbonett.evolve.multiset import run_multiset_mpo_hamiltonian
from fishbonett.evolve.multiset_tree import run_multiset_tree_hamiltonian
from fishbonett.bath import (
    Bath, CoupledBath, thermalize,
    integrated_free_phase, reorganization_energy, star_coupling_squared,
    get_bath_nn_parameters, get_coupling,
    get_vn_squared, get_legendre_recursion,
    get_vn_squared_tedopa, make_tedopa_discretizer,
    lanczos, recurrence_coefficients,
)
from fishbonett.linalg import Truncation
from fishbonett.targets import BathMode
from fishbonett.system import System
from fishbonett.models import (
    Result, SimulationCheckpoint, ExcitonBath, Fishbone, SystemBath,
    TreeFishbone,
)
from fishbonett.states.tree import TreeTensorNetwork
from fishbonett.states.thermal import GibbsPurification
from fishbonett.operators import (
    sigma_0, sigma_1, sigma_x, sigma_y, sigma_z, sigma_p, sigma_m,
    temp_factor, entang, energy_current_operator,
)
from fishbonett.spectral_densities import (
    drude, drude1, brownian, lorentzian, natphys, lemmer,
    sd_back, sd_high, sd_zero_temp,
)

try:
    __version__ = _version("fishbonett")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    # low-level state and evolution entry points
    "SystemBathMPS",
    "MultiSetMPS",
    "MultiSetTreeTensorNetwork",
    "run_mpo_hamiltonian",
    "run_multiset_mpo_hamiltonian",
    "run_multiset_tree_hamiltonian",
    "run_tree_tebd",
    # high-level interface
    "Bath", "CoupledBath",
    "System", "SystemBath", "ExcitonBath",
    "Result", "SimulationCheckpoint", "Truncation", "thermalize", "Fishbone",
    "BathMode",
    "TreeFishbone", "TreeTensorNetwork", "GibbsPurification",
    # discretization / chain mapping
    "integrated_free_phase", "reorganization_energy", "star_coupling_squared",
    "get_bath_nn_parameters", "get_coupling", "get_vn_squared",
    "get_legendre_recursion", "lanczos", "recurrence_coefficients",
    "get_vn_squared_tedopa", "make_tedopa_discretizer",
    # operators
    "sigma_0", "sigma_1", "sigma_x", "sigma_y", "sigma_z", "sigma_p", "sigma_m",
    "temp_factor", "entang", "energy_current_operator",
    # spectral densities
    "drude", "drude1", "brownian", "lorentzian", "natphys", "lemmer",
    "sd_back", "sd_high", "sd_zero_temp",
]
