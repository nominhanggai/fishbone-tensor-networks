"""Metropolis-Hastings integrators for the high-order golden-rule corrections.

The order-``k`` correction in :mod:`fishbonett.rates.golden_rule_multi` is a
``k``-dimensional *time-ordered* integral ``t > t_1 > ... > t_k > 0``.  Beyond
about four dimensions deterministic quadrature is impractical, so these routines
sample it instead.

.. rubric:: API

==============================  ===============================================
:func:`mcmc_time_ordered`       the one that matters: time-ordered ``dim``-D
:func:`mcmc1d`, :func:`mcmc2d`  1D and 2D versions, for testing the machinery
==============================  ===============================================

.. rubric:: The oscillatory-integrand problem

The integrands here are complex and oscillatory, so they cannot be used directly
as a sampling density.  :func:`mcmc_time_ordered` samples ``|f|`` and carries the
phase separately, recombining as ``<e^{i arg f}> / <1/|f|>``.  This is a
reweighting estimator: when the phase oscillates rapidly the numerator nearly
cancels and the variance grows -- the usual sign problem.  Check convergence by
watching the running mean rather than trusting a single ``N``, and note that the
estimator is *biased* at finite ``N`` (it is a ratio of means), which is why
``burn_in`` samples are discarded.

Each function accepts a local ``seed`` and never mutates NumPy's global random
state.
"""
import math

import numpy as np


def _validated_inputs(interval, N, seed):
    if (not isinstance(N, (int, np.integer)) or isinstance(N, (bool, np.bool_))
            or N < 1):
        raise ValueError("N must be a positive integer")
    if len(interval) != 2:
        raise ValueError("interval must be (lower, upper)")
    lower, upper = map(float, interval)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("interval must contain finite values with lower < upper")
    return lower, upper, np.random.default_rng(seed)


def _positive_weight(value):
    value = np.asarray(value)
    if value.ndim != 0 or np.iscomplexobj(value):
        raise ValueError("Metropolis sampling weights must be real scalars")
    weight = float(value)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Metropolis sampling weights must be finite and non-negative")
    return weight


def _acceptance_ratio(new_weight, old_weight, proposal_ratio=1.0):
    """Stable Metropolis ratio, including zero-density states."""
    if old_weight == 0:
        return 1.0 if new_weight > 0 else 0.0
    return min(1.0, new_weight / old_weight * proposal_ratio)


def mcmc2d(func, interval, N, *, seed=None):
    """Metropolis sample of ``func(x, y)`` over the ordered wedge ``x < y``.

    Two-dimensional test case for the machinery in :func:`mcmc_time_ordered`;
    returns the array of ``N+1`` accepted samples (not yet normalized by the
    domain volume).
    """
    lower, upper, rng = _validated_inputs(interval, N, seed)
    y = rng.uniform(lower, upper)
    x = rng.uniform(lower, y)
    samples = [_positive_weight(func(x, y))]
    mc_points = [(x, y)]

    def Omega(X1, X2):
        xa1, yb1 = X1
        xa2, yb2 = X2
        return 1 / (yb2 - lower)

    for i in range(N):
        y = rng.uniform(lower, upper)
        x = rng.uniform(lower, y)

        new_sample = _positive_weight(func(x, y))
        ratio = _acceptance_ratio(
            new_sample, samples[i],
            Omega((x, y), mc_points[i]) / Omega(mc_points[i], (x, y)),
        )

        r = rng.uniform(0, 1)

        if r < ratio:
            samples.append(new_sample)
            mc_points.append((x, y))
        else:
            samples.append(samples[i])
            mc_points.append(mc_points[i])

    return np.array(samples)


def mcmc1d(func, interval, N, *, seed=None):
    """Metropolis sample of ``func(x)`` over ``interval`` with a uniform proposal.

    The one-dimensional reference case: with a uniform proposal the
    Metropolis ratio reduces to ``f(new)/f(old)``.  Returns the ``N+1`` accepted
    samples.
    """
    lower, upper, rng = _validated_inputs(interval, N, seed)
    x = rng.uniform(lower, upper)
    d = upper - lower
    samples = [_positive_weight(func(x))]
    mc_points = [x]

    def Omega(X1, X2):
        return 1 / d

    for i in range(N):
        x = rng.uniform(lower, upper)

        new_sample = _positive_weight(func(x))
        ratio = _acceptance_ratio(new_sample, samples[i])

        r = rng.uniform(0, 1)

        if r < ratio:
            samples.append(new_sample)
            mc_points.append(x)
        else:
            samples.append(samples[i])
            mc_points.append(mc_points[i])

    return np.array(samples)


def mcmc_time_ordered(func, dim, interval, N, burn_in=1000, *, seed=None):
    """Time-ordered ``dim``-dimensional integral of a complex oscillatory ``func``.

    Samples the simplex ``t_max > t_1 > t_2 > ... > t_dim > t_min`` by drawing
    each ``t_i`` uniformly below its predecessor, with the Metropolis ratio
    correcting for the resulting non-uniform proposal density.

    Because ``func`` is complex, ``|func|`` is used as the sampling weight and the
    phase is accumulated separately; the estimate is then
    ``<e^{i arg f}> / <1/|f|>``, normalized by the simplex volume
    ``(t_max - t_min)^dim / dim!``.

    Parameters
    ----------
    func : callable
        Takes ``dim`` time arguments, returns a complex value.
    dim : int
        Integral dimension (the perturbative order).
    interval : (float, float)
        ``(t_min, t_max)``.
    N : int
        Number of Metropolis proposals.
    burn_in : int
        Samples discarded from the denominator average.

    Returns
    -------
    (estimate, phases, samples)
        The complex estimate, plus the raw phase and magnitude arrays so
        convergence can be inspected -- see the module docstring on why a single
        ``N`` should not be trusted.
    """
    t_min, t_max, rng = _validated_inputs(interval, N, seed)
    if (not isinstance(dim, (int, np.integer)) or isinstance(dim, (bool, np.bool_))
            or dim < 1):
        raise ValueError("dim must be a positive integer")
    if (not isinstance(burn_in, (int, np.integer))
            or isinstance(burn_in, (bool, np.bool_))
            or burn_in < 0 or burn_in >= N + 1):
        raise ValueError("burn_in must satisfy 0 <= burn_in < N + 1")

    def generate_t():
        t_list = np.array([t_max] * (dim + 1), dtype=np.float64)
        for i in range(1, dim + 1):
            t_list[i] = rng.uniform(t_min, t_list[i - 1])

        t_list = t_list[1:]
        return t_list

    mc_points = [generate_t()]
    samples = [_positive_weight(np.abs(func(*mc_points[-1])))]

    def omega(X1, X2):
        return 1 / np.prod(X2[:-1] - t_min)

    for i in range(N):
        t_list = generate_t()

        new_sample = _positive_weight(np.abs(func(*t_list)))
        ratio = _acceptance_ratio(
            new_sample, samples[i],
            omega(t_list, mc_points[i]) / omega(mc_points[i], t_list),
        )

        r = rng.uniform(0, 1)

        if r < ratio:
            samples.append(new_sample)
            mc_points.append(t_list)
        else:
            samples.append(samples[i])
            mc_points.append(mc_points[i])
    numerator = np.array([np.exp(1j * np.angle(func(*p))) for p in mc_points])
    samples = np.array(samples) / math.factorial(dim) * (t_max - t_min) ** dim
    retained_weights = samples[burn_in:]
    if np.any(retained_weights <= 0):
        raise ValueError("retained samples contain zero target weight")
    retained_phase = numerator[burn_in:]
    estimate = np.mean(retained_phase) / np.mean(1 / retained_weights)
    return estimate, numerator, samples
