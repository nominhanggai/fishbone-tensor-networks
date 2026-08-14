"""Schroedinger picture: nothing rotated out, ``H`` static and strictly local.

The frame with no transformation.  The chain-mapped Hamiltonian is written down as
it stands -- bath frequencies on their own sites, nearest-neighbour hoppings between
them -- so ``H`` is time-independent and its gates and MPO are built **once**.  The
price is entanglement: nothing has been removed, so the state carries the full
system-bath correlation and the bond dimensions are the largest of any frame.

This module is the frame for *any* topology: one system site with a chain of modes,
a comb, or an arbitrary loop-free tree of sites each with its own bath(s).  They
differ only in the edge list, which is why :func:`terms` returns a
:class:`~fishbonett.frames.terms.LocalTerms` graph rather than anything
geometry-specific.

.. rubric:: What's here

===================  ===========================================================
:func:`terms`        a model's systems + baths -> static ``LocalTerms``
:func:`chain_terms`  the nodes and edges one bath chain contributes
:func:`star_terms`   the same for a shared-mode multichannel star
===================  ===========================================================

The MPO form of this frame for the single-system models lives in
:mod:`fishbonett.frames.mpo` (``build_chain_mpo``, ``build_static_star_mpo``) and is
driven by :mod:`fishbonett.evolve.tdvp`; the gate form is driven by
:mod:`fishbonett.evolve.sitetree`.
"""
import numpy as np

from fishbonett.bath.coupled import bind_bath
from fishbonett.frames.terms import LocalTerms
from fishbonett.operators import annihilate, sigma_z

__all__ = ["terms", "chain_terms", "star_terms", "bath_ops"]


def bath_ops(d):
    """``(a, a_dag, x, n)`` on a ``d``-dimensional boson site, with ``x = a + a^dag``."""
    a = annihilate(d)
    ad = a.T
    return a, ad, a + ad, ad @ a


def chain_terms(bath, site, next_node, dims, edges, site_H, edge_H):
    """Append one bath as a TEDOPA **chain** hanging off ``site``.

    Mode ``m`` gets its frequency ``w[m]`` on its own node; the system-bath coupling
    ``k[0]`` sits on the ``(site, c_0)`` edge and the mode-mode hoppings ``k[m]`` on
    the edges after it.  Returns the next free node id.
    """
    coupled = bind_bath(bath, default_operator=sigma_z)
    compiled = coupled.compiled_chain()
    d = compiled.phys_dim
    a, ad, x, numb = bath_ops(d)
    w = compiled.frequencies
    cop = coupled.operator
    prev = site
    node = next_node
    for m in range(compiled.n_modes):
        dims.append(d)
        site_H.append(w[m] * numb)
        edges.append((prev, node))
        if m == 0:
            edge_H[(prev, node)] = compiled.system_coupling * np.kron(cop, x)
        else:
            edge_H[(prev, node)] = compiled.hoppings[m - 1] * (
                np.kron(ad, a) + np.kron(a, ad))
        prev = node
        node += 1
    return node


def star_terms(bath, site, next_node, dims, edges, site_H, edge_H):
    """Append one **multichannel** bath as a shared-mode star on ``site``.

    Every channel is discretized on the *same* Gauss-Legendre nodes ``omega_k``, so
    the channels are cross-correlated rather than independent, and mode ``k`` couples
    through the combined operator ``M_k = sum_c g_{c,k} O_c`` with
    ``g_{c,k} = sqrt(J_c(omega_k) w_k / pi)``.  Returns the next free node id.
    """
    coupled = bind_bath(bath)
    star = coupled.compiled_star()
    coup_mat = star.combine(coupled.operators)
    _a, _ad, x, numb = bath_ops(star.phys_dim)
    node = next_node
    for k in range(star.n_modes):
        dims.append(star.phys_dim)
        site_H.append(star.frequencies[k] * numb)
        edges.append((site, node))
        # (site op M_k) (x) (a + a^dag)
        edge_H[(site, node)] = np.kron(coup_mat[k], x)
        node += 1
    return node


def terms(sites, edges, baths, t_max=None):
    """The static Hamiltonian of a multi-site model, as :class:`LocalTerms`.

    Parameters
    ----------
    sites : list of (d, d) array
        The system-site Hamiltonians.  Nodes ``0..len(sites)-1``.
    edges : list of (i, j, C)
        System-system couplings, forming a tree over the sites; ``C`` is a
        ``(d_i*d_j, d_i*d_j)`` operator on the pair.
    baths : list
        One entry per site: a list of :class:`~fishbonett.bath.spec.Bath` (possibly
        empty).  Each bath becomes a chain -- or, if multichannel, a star -- of nodes
        hanging off its site.
    t_max : float, optional
        Propagation time, needed only to size a bath whose ``n_modes`` is automatic
        (see :meth:`fishbonett.bath.spec.Bath.resolved`).
    """
    ns = len(sites)
    dims = [np.asarray(h).shape[0] for h in sites]
    edge_list = [(i, j) for (i, j, _) in edges]
    site_H = [np.asarray(sites[i], complex).copy() for i in range(ns)]
    edge_H = {(i, j): C for (i, j, C) in edges}
    node = ns
    for i in range(ns):
        for bath in baths[i]:
            bath = bath.resolved(t_max)          # fill automatic domain / n_modes
            coupled = bind_bath(bath, default_operator=sigma_z)
            build = (star_terms if coupled.is_multichannel
                     else chain_terms)
            node = build(coupled, i, node, dims, edge_list, site_H, edge_H)
    return LocalTerms(dims=dims, edges=edge_list, site=site_H, bond=edge_H)
