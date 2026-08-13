"""Frames: the same physical model, written down in different ways.

A *frame* is a choice of what the Hamiltonian looks like -- which terms have been
rotated or transformed away, and what is left to propagate.  It is the **middle**
of three nested choices: the :mod:`model <fishbonett.models>` comes first and
decides which frames are even available, and the frame in turn decides which
propagators (:mod:`fishbonett.evolve`) apply.

=============================================  =====================================
:mod:`~fishbonett.frames.schrodinger`          nothing rotated out; ``H`` static
:mod:`~fishbonett.frames.interaction_picture`  free bath rotated out; ``H(t)``
:mod:`~fishbonett.frames.polaron`              Lang-Firsov; static, low entanglement
:mod:`~fishbonett.frames.multichannel`         interaction picture, several couplings
:mod:`~fishbonett.frames.coolingchain`         finite ``T`` by a non-unitary gauge
:mod:`~fishbonett.frames.terms`                ``LocalTerms``, what a static frame emits
:mod:`~fishbonett.frames.gates`                two-site gates, what a ``H(t)`` frame emits
:mod:`~fishbonett.frames.mpo`                  the chain/star MPOs the frames emit
=============================================  =====================================

Each module produces either Trotter gates or an MPO; the state lives in
:mod:`fishbonett.states` and the propagation algorithm in
:mod:`fishbonett.evolve`.  Which propagators a frame admits is not arbitrary --
a static ``H`` can drive TDVP with a once-built MPO, a time-dependent one must
rebuild each step, and only the interaction picture makes all the coupling terms
commute (which is what ``trotter-mpo`` exploits).  See :doc:`/methods/index`.

The **Schroedinger picture** is the one frame that is geometry-independent: it
rotates nothing away, so its Hamiltonian is just the chain-mapped one written down,
and :func:`fishbonett.frames.schrodinger.terms` emits it as a
:class:`~fishbonett.frames.terms.LocalTerms` graph for *any* topology -- one system
with a chain of modes, a comb, or an arbitrary tree of sites.  That construction used
to live inside :class:`fishbonett.models.fishbone.TreeFishbone`, so the multi-site
models bypassed this package entirely; they now go through it like everything else.

The time-dependent frames do **not** produce ``LocalTerms``, and that is the honest
shape of the physics rather than an omission: the interaction picture has no on-site
bath terms at all and a coupling that must be rebuilt every step, and the polaron
frame folds the coupling into a displacement on a single bond.  They emit a list of
two-site Hamiltonians instead, which
:func:`fishbonett.frames.gates.swap_gate_pairs` exponentiates -- so ``SystemBathIP``
and ``SystemBathMultiChannel`` differ in how ``H(t)`` is *built*, not in how it
becomes gates.

Not every frame is available for every model: the multi-site models have only the
Schroedinger picture, and ``coolingchain`` sits outside the taxonomy entirely
(there the frame *is* the state).  :mod:`fishbonett.models.registry` records which
combinations exist and why the rest do not.

Site ordering: the system is **site 0** and the bath modes follow, nearest first
(``[system, c_0, c_1, ...]``).
"""
