"""Tensor contractions used by the tensor-network engines.

The package requires :mod:`opt_einsum` so that its MPS and tree engines use the
same contraction-path implementation on every supported installation. This
module keeps that dependency behind one stable package-level function.
"""
from functools import lru_cache

from opt_einsum import contract as _contract_impl
from opt_einsum import contract_expression as _contract_expression_impl
from opt_einsum import get_symbol

__all__ = ["contract"]


def contract(subscripts, *operands, **kwargs):
    """Contract tensors with :func:`opt_einsum.contract`.

    The accepted arguments are those of :func:`numpy.einsum`, including the
    interleaved integer-label form.
    """
    return _contract_impl(subscripts, *operands, **kwargs)


@lru_cache(maxsize=4096)
def _cached_expression(pairs, output):
    """Compile an interleaved-label contraction for fixed tensor shapes."""
    inputs = [
        "".join(get_symbol(label) for label in labels)
        for _shape, labels in pairs
    ]
    result = "".join(get_symbol(label) for label in output)
    equation = f"{','.join(inputs)}->{result}"
    return _contract_expression_impl(
        equation,
        *(shape for shape, _labels in pairs),
        optimize="greedy",
    )


def _contract_cached(*operands):
    """Contract interleaved-label operands with a shape-cached expression."""
    pairs = tuple(
        (tuple(operands[index].shape), tuple(operands[index + 1]))
        for index in range(0, len(operands) - 1, 2)
    )
    output = tuple(operands[-1])
    tensors = operands[:-1:2]
    return _cached_expression(pairs, output)(*tensors)


@lru_cache(maxsize=4096)
def _cached_subscript_expression(subscripts, shapes):
    """Compile a conventional Einstein-subscript contraction by shape."""
    return _contract_expression_impl(subscripts, *shapes, optimize="greedy")


def _einsum_cached(subscripts, *operands):
    """Evaluate a fixed-subscript contraction through cached opt_einsum."""
    shapes = tuple(tuple(operand.shape) for operand in operands)
    return _cached_subscript_expression(subscripts, shapes)(*operands)
