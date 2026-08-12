"""Back-compat shim: moved to :mod:`fishbonett.bath.lanczos`."""
from fishbonett.bath.lanczos import *           # noqa: F401,F403
from fishbonett.bath import lanczos as _m
globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith("__")})
