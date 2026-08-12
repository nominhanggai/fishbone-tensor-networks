"""Back-compat shim: moved to :mod:`fishbonett.bath.legendre`."""
from fishbonett.bath.legendre import *          # noqa: F401,F403
from fishbonett.bath import legendre as _m
globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith("__")})
