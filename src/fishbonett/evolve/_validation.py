"""Validation shared by public low-level propagation drivers."""

import numpy as np


def time_steps(dt, nsteps):
    if (isinstance(dt, (bool, np.bool_))
            or not isinstance(dt, (int, float, np.number))
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be a finite positive number")
    if (isinstance(nsteps, (bool, np.bool_))
            or not isinstance(nsteps, (int, np.integer)) or nsteps < 1):
        raise ValueError("nsteps must be a positive integer")
    return float(dt), int(nsteps)


def positive_integer(value, name, *, allow_none=False):
    if value is None and allow_none:
        return None
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return int(value)


def nonnegative_finite(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
