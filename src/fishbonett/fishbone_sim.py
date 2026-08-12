"""Back-compat shim: the high-level 1D :class:`Fishbone` interface now lives in
:mod:`fishbonett.simulate` (alongside :class:`Bath` and :class:`BosonicBath`)."""
from fishbonett.simulate import Fishbone      # noqa: F401
