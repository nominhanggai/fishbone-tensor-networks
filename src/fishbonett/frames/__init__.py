"""Frames: the same physical model, written down in different ways.

A *frame* is a choice of what the Hamiltonian looks like -- which terms have been
rotated or transformed away, and what is left to propagate.  It is the **middle**
of three nested choices: the :mod:`model <fishbonett.models>` comes first and
decides which frames are even available, and the frame in turn decides which
propagators (:mod:`fishbonett.evolve`) apply.

=============================================  =====================================
:mod:`~fishbonett.frames.interaction_picture`  free bath rotated out; ``H(t)``
:mod:`~fishbonett.frames.polaron`              Lang-Firsov; static, low entanglement
:mod:`~fishbonett.frames.multichannel`         interaction picture, several couplings
:mod:`~fishbonett.frames.coolingchain`         finite ``T`` by a non-unitary gauge
:mod:`~fishbonett.frames.mpo`                  the chain/star MPOs the frames emit
=============================================  =====================================

Each module produces either Trotter gates or an MPO; the state lives in
:mod:`fishbonett.states` and the propagation algorithm in
:mod:`fishbonett.evolve`.  Which propagators a frame admits is not arbitrary --
a static ``H`` can drive TDVP with a once-built MPO, a time-dependent one must
rebuild each step, and only the interaction picture makes all the coupling terms
commute (which is what ``trotter-mpo`` exploits).  See :doc:`/methods/index`.

The **Schroedinger picture has no builder class here**, which is the clearest sign
that this package is organized by (model, frame) pair rather than by frame: its
chain and star MPOs are :mod:`fishbonett.frames.mpo` functions called from
:mod:`fishbonett.evolve.tdvp`, and the multi-site models build their static
Hamiltonian inline in :meth:`fishbonett.models.fishbone.TreeFishbone.hamiltonians`.

Not every frame is available for every model: the multi-site models have only the
Schroedinger picture, and ``coolingchain`` sits outside the taxonomy entirely
(there the frame *is* the state).  :mod:`fishbonett.models.registry` records which
combinations exist and why the rest do not.

Site ordering: the system is **site 0** and the bath modes follow, nearest first
(``[system, c_0, c_1, ...]``).
"""
