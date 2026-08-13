"""Frames: the same physical model, written down in different ways.

A *frame* is a choice of what the Hamiltonian looks like -- which terms have been
rotated or transformed away, and what is left to propagate.  It is one of three
independent choices you make (see :mod:`fishbonett` for the other two: the
geometry, and the propagator).

============================================  =================================
:mod:`~fishbonett.frames.schrodinger`         nothing rotated out; ``H`` static
:mod:`~fishbonett.frames.interaction_picture` free bath rotated out; ``H(t)``
:mod:`~fishbonett.frames.polaron`             Lang-Firsov; static, low entanglement
:mod:`~fishbonett.frames.multichannel`        interaction picture, several couplings
:mod:`~fishbonett.frames.coolingchain`        finite ``T`` by a non-unitary gauge
:mod:`~fishbonett.frames.mpo`                 the chain/star MPOs the frames emit
============================================  =================================

Each module produces either Trotter gates or an MPO; the state lives in
:mod:`fishbonett.states` and the propagation algorithm in
:mod:`fishbonett.evolve`.  Which propagators a frame admits is not arbitrary --
a static ``H`` can drive TDVP with a once-built MPO, a time-dependent one must
rebuild each step, and only the interaction picture makes all the coupling terms
commute (which is what ``trotter-mpo`` exploits).  See :doc:`/methods/index`.

Site ordering: the system is **site 0** and the bath modes follow, nearest first
(``[system, c_0, c_1, ...]``).
"""
