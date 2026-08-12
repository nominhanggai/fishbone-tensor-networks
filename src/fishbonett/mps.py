"""Back-compat shim.  The MPS *state* moved to :mod:`fishbonett.states.mps` and
the swap-network TEBD *sweep* to :func:`fishbonett.evolve.tebd.update_bond`."""
from fishbonett.states.mps import (  # noqa: F401
    BosonicBathMPS, BosonicBath1D, SpinBosonMPS, SpinBoson1D,  # last two deprecated
)
