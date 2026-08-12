"""Tensor-contraction backend.

Provides :func:`contract`, a drop-in for ``opt_einsum.contract`` /
``numpy.einsum``.  ``opt_einsum`` is used when installed (it caches contraction
paths and is a little faster on the hot tensor-network kernels), otherwise this
falls back to ``numpy.einsum`` with greedy path optimization -- so ``opt_einsum``
is an optional dependency, not a hard requirement.

Set the environment variable ``FISHBONETT_EINSUM=numpy`` to force the NumPy
backend even when ``opt_einsum`` is installed (used for benchmarking).
"""
import os

import numpy as _np

__all__ = ["contract", "BACKEND"]

_force = os.environ.get("FISHBONETT_EINSUM", "").lower()


def _numpy_contract(subscripts, *operands, **kwargs):
    kwargs.setdefault("optimize", "greedy")
    return _np.einsum(subscripts, *operands, **kwargs)


if _force == "numpy":
    contract = _numpy_contract
    BACKEND = "numpy"
else:
    try:
        from opt_einsum import contract as _oe_contract  # noqa: F401
        contract = _oe_contract
        BACKEND = "opt_einsum"
    except ImportError:
        contract = _numpy_contract
        BACKEND = "numpy"
