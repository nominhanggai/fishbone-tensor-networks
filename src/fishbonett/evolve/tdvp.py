"""Public chain-TDVP API.

The implementation is split by responsibility:

* :mod:`fishbonett.evolve._tdvp_kernels` contains local contractions,
  canonicalization and Krylov evolution;
* :mod:`fishbonett.evolve._tdvp_sweeps` contains one-site, two-site and adaptive
  projector-splitting sweeps;
* :mod:`fishbonett.evolve._tdvp_driver` owns the whole-run loop for a prepared
  representation object exposing ``tdvp_mpo(t)``.

Applications may import the documented TDVP operations from this module. Internal
modules import their lowest required implementation layer.
"""
from fishbonett.evolve._tdvp_kernels import (
    SX, SZ, applyH0 as _apply_h0, applyH1 as _apply_h1,
    evolveAC as _evolve_site_tensor, evolveC as _evolve_bond_tensor,
    expmv_lanczos, init_mps, init_right_envs, krylov_statistics, left_qr,
    right_canonicalize, right_lq,
    updateleftenv as _update_left_environment,
    updaterightenv as _update_right_environment,
)
from fishbonett.evolve._tdvp_sweeps import (
    applyH2 as _apply_h2, bonddims, measure_rdm, measure_sz, tdvp1sweep,
    a1tdvp_sweep, tdvp2sweep,
)
from fishbonett.evolve._tdvp_driver import run_mpo_hamiltonian

# Public PEP 8 names for the internal contraction kernels.
apply_h0 = _apply_h0
apply_h1 = _apply_h1
apply_h2 = _apply_h2
evolve_site_tensor = _evolve_site_tensor
evolve_bond_tensor = _evolve_bond_tensor
update_left_environment = _update_left_environment
update_right_environment = _update_right_environment

__all__ = [
    "SX", "SZ", "init_mps", "apply_h1", "apply_h0",
    "update_left_environment", "update_right_environment",
    "right_canonicalize", "init_right_envs", "left_qr",
    "right_lq", "expmv_lanczos", "evolve_site_tensor",
    "evolve_bond_tensor", "tdvp1sweep", "measure_sz", "measure_rdm",
    "apply_h2", "tdvp2sweep", "bonddims",
    "a1tdvp_sweep", "run_mpo_hamiltonian", "krylov_statistics",
]
