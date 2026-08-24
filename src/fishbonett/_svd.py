"""Internal singular-value decomposition helpers."""
import warnings

from numpy.linalg import LinAlgError
import scipy.linalg


def robust_svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
               check_finite=True, lapack_driver="gesdd", warn=True):
    """Evaluate an SVD and retry with the robust LAPACK driver if necessary.

    ``gesdd`` is normally faster, while ``gesvd`` is a useful fallback for the
    rare matrices on which the divide-and-conquer iteration does not converge.
    The return convention is the same as :func:`scipy.linalg.svd`.
    """
    try:
        return scipy.linalg.svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            overwrite_a=overwrite_a,
            check_finite=check_finite,
            lapack_driver=lapack_driver,
        )
    except LinAlgError:
        if lapack_driver == "gesvd":
            raise
        if warn:
            warnings.warn(
                "gesdd SVD did not converge; retrying with gesvd.",
                stacklevel=2,
            )
        return scipy.linalg.svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            overwrite_a=overwrite_a,
            check_finite=check_finite,
            lapack_driver="gesvd",
        )
