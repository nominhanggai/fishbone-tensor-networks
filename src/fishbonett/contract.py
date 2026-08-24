"""Tensor contractions used by the tensor-network engines.

The package requires :mod:`opt_einsum` so that its MPS and tree engines use the
same contraction-path implementation on every supported installation. This
module keeps that dependency behind one stable package-level function.
"""
from opt_einsum import contract as _contract_impl

__all__ = ["contract"]


def contract(subscripts, *operands, **kwargs):
    """Contract tensors with :func:`opt_einsum.contract`.

    The accepted arguments are those of :func:`numpy.einsum`, including the
    interleaved integer-label form.
    """
    return _contract_impl(subscripts, *operands, **kwargs)
