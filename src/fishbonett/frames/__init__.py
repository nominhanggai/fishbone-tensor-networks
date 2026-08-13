"""Frames / representations of the same system + harmonic-bath Hamiltonian.

Each module builds the *same* physical model in a different frame -- Schroedinger
picture (:mod:`~fishbonett.frames.schrodinger`), interaction picture
(:mod:`~fishbonett.frames.interaction_picture`, :mod:`~fishbonett.frames.multichannel`),
polaron / Lang-Firsov frame (:mod:`~fishbonett.frames.polaron`), or a cooling
ansatz (:mod:`~fishbonett.frames.coolingchain`) -- as a Hamiltonian or as Trotter
gates, and drives the unified engines in :mod:`fishbonett.states` /
:mod:`fishbonett.evolve`.
"""
