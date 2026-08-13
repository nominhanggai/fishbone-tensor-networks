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

==================  ============================================================
:func:`terms`       a model's systems + baths -> static ``LocalTerms``
:func:`chain_terms` the nodes and edges one bath chain contributes
:func:`star_terms`  the same for a shared-mode multichannel star
==================  ============================================================

The MPO form of this frame for the single-system models lives in
:mod:`fishbonett.frames.mpo` (``build_chain_mpo``, ``build_static_star_mpo``) and is
driven by :mod:`fishbonett.evolve.tdvp`; the gate form is driven by
:mod:`fishbonett.evolve.sitetree`.
"""
import numpy as np

from fishbonett.bath.chain import get_bath_nn_paras
from fishbonett.bath.legendre import get_vn_squared
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
    d = bath.phys_dim
    a, ad, x, numb = bath_ops(d)
    w, k = get_bath_nn_paras(bath.spectral_density(), bath.n_modes,
                             list(bath.domain), discretizer=bath.discretizer())
    w = np.asarray(w, float)
    k = np.asarray(k, float)
    cop = np.asarray(bath.coupling if bath.coupling is not None else sigma_z, complex)
    prev = site
    node = next_node
    for m in range(bath.n_modes):
        dims.append(d)
        site_H.append(w[m] * numb)
        edges.append((prev, node))
        if m == 0:
            edge_H[(prev, node)] = k[0] * np.kron(cop, x)
        else:
            edge_H[(prev, node)] = k[m] * (np.kron(ad, a) + np.kron(a, ad))
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
    if bath.discretization != "legendre":
        raise ValueError("a multichannel bath must use the 'legendre' "
                         "discretization: its Gauss nodes are shared across "
                         "channels, whereas measure-adapted TEDOPA nodes are not")
    _a, _ad, x, numb = bath_ops(bath.phys_dim)
    channels = bath.channels()
    freq, g = None, []
    for Jc, _op in channels:
        f, v_sq = get_vn_squared(Jc, bath.n_modes, list(bath.domain))
        f = np.asarray(f, float)
        g.append(np.sqrt(np.asarray(v_sq, float) / np.pi))
        if freq is None:
            freq = f
        elif not np.allclose(freq, f):        # nodes are shared, so unreachable
            raise ValueError("multichannel channels do not share the mode grid")
    node = next_node
    for k in range(bath.n_modes):
        dims.append(bath.phys_dim)
        site_H.append(freq[k] * numb)
        M = sum(g[c][k] * channels[c][1] for c in range(len(channels)))
        edges.append((site, node))
        edge_H[(site, node)] = np.kron(M, x)          # (site op M) (x) (a + a^dag)
        node += 1
    return node


def terms(sites, edges, baths, t_max=None):
    """The static Hamiltonian of a multi-site model, as :class:`LocalTerms`.

    Parameters
    ----------
    sites : list of (d, d) array
        The system-site Hamiltonians.  Nodes ``0..len(sites)-1``.
    edges : list of (i, j, coupling)
        System-system couplings, forming a tree over the sites.
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
            build = (star_terms if getattr(bath, "is_multichannel", False)
                     else chain_terms)
            node = build(bath, i, node, dims, edge_list, site_H, edge_H)
    return LocalTerms(dims=dims, edges=edge_list, site=site_H, bond=edge_H)
