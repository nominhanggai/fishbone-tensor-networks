"""Rate theory for electron / excitation-energy transfer.

Fermi golden-rule and Marcus rates from a discretized spectral density
(:mod:`~fishbonett.rates.golden_rule`), multi-acceptor golden-rule corrections
(:mod:`~fishbonett.rates.golden_rule_multi`), Metropolis integrators
(:mod:`~fishbonett.rates.mcmc`), and the transfer-tensor method for long-time
dynamics (:mod:`~fishbonett.rates.transfer_tensor`).
"""
from fishbonett.rates.golden_rule import (
    fgr_rate, fgr_decay_profile, fgr_rate_by_order, marcus_rate,
)
from fishbonett.rates.mcmc import mcmc1d, mcmc2d, mcmc_time_ordered
from fishbonett.rates.transfer_tensor import (
    transfer_mat, predict_density_mat, dynamical_maps,
)

__all__ = [
    "fgr_rate", "fgr_decay_profile", "fgr_rate_by_order", "marcus_rate",
    "mcmc1d", "mcmc2d", "mcmc_time_ordered",
    "transfer_mat", "predict_density_mat", "dynamical_maps",
]
