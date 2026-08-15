"""Transfer-tensor method (TTM): extrapolate short simulations to long times.

Propagating to long times is expensive; the bath memory kernel is usually much
shorter than the dynamics you want.  TTM exploits that.  Run ``d^2`` short
simulations -- one per initial condition in a basis of the Liouville space -- to
build the *dynamical maps* ``E(t_n)``, deconvolve them into *transfer tensors*
``T_n`` that isolate the memory kernel, and then iterate those tensors forward
indefinitely at negligible cost.

The workflow::

    dynamical_maps(t, d, basis_map)   # d^2 short runs -> E(t_n)
        -> transfer_mat([E(t_1), ...])   # deconvolve  -> T_n
        -> predict_density_mat(t, T, rho_short)   # iterate to long t

It works when ``T_n`` decays: once ``T_norm`` has fallen to your tolerance the
kernel is exhausted, and truncating there is exact to that tolerance.  If it has
*not* decayed by the end of the short runs, extrapolation is unjustified -- watch
the ``T_norm`` returned by :func:`transfer_mat`.

.. rubric:: API

===============================  ================================================
:func:`dynamical_maps`           assemble ``E(t)`` from the stored short runs
:func:`transfer_mat`             deconvolve ``E`` into transfer tensors + norms
:func:`predict_density_mat`      iterate the tensors to long times
:func:`map_basis_op`             one Liouville basis element from stored runs
:func:`read_rho`                 load a stored density matrix
===============================  ================================================

.. rubric:: Input convention

Density matrices are read from ``output/density_mat_<label>.npy`` in the
*current working directory* (one file per short run, indexed by time step), and
``basis_map`` maps a matrix index ``(i, j)`` to the label(s) that produced it.
Off-diagonal elements need two real runs each, combined as::

    rho_ij = r1 + i r2 - (1 + i)(r_ii + r_jj)/2      (i < j)
"""
import numpy as np
import copy
import itertools as it


def read_rho(label, t):
    """Density matrix at time index ``t`` from ``output/density_mat_<label>.npy``."""
    r = np.load(f"output/density_mat_{label}.npy")
    return r[t]


def map_basis_op(index, t, basis_map):
    """The Liouville basis element ``|i><j|`` propagated to time index ``t``.

    Diagonal elements come from a single stored run.  Off-diagonal ones are
    reconstructed from two real-valued runs plus the two diagonals, using the
    combination in the module docstring; ``index[0] > index[1]`` is the conjugate
    of the transposed case.
    """
    # print(index, t)
    if index[0] == index[1]:
        id_ = basis_map[index]
        return read_rho(id_, t)
    if index[0] < index[1]:
        id1 = basis_map[index][0]
        id2 = basis_map[index][1]
        r1 = read_rho(id1, t)
        r2 = read_rho(id2, t)
        id3 = basis_map[(index[0], index[0])]
        id4 = basis_map[(index[1], index[1])]
        r3 = read_rho(id3, t)
        r4 = read_rho(id4, t)
        return r1 + 1j * r2 - (1 + 1j) * (r3 + r4) / 2
    if index[0] > index[1]:
        index_ = (index[1], index[0])
        id1 = basis_map[index_][0]
        id2 = basis_map[index_][1]
        r1 = read_rho(id1, t)
        r2 = read_rho(id2, t)
        id3 = basis_map[(index_[0], index_[0])]
        id4 = basis_map[(index_[1], index_[1])]
        r3 = read_rho(id3, t)
        r4 = read_rho(id4, t)
        return r1 - 1j * r2 - (1 - 1j) * (r3 + r4) / 2


def transfer_mat(lt_map):
    """
    Args:
        lt_map (): a list of dynamical maps in the Liouville space, i.e., the
            basis is ``{|n>|m>}``.

    Returns:
        T: a list of same number of transfer tensors as the dynamical maps.
        T_norm: the corresponding matrix norm of elements in T
    """
    T1 = lt_map[0]
    T = [T1]
    T_norm = [np.linalg.norm(T1)]
    for N in range(1, len(lt_map)):
        TN = lt_map[N] - np.einsum('Nij,Njk->ik', T, lt_map[0:N][::-1])
        T.append(TN)
        T_norm.append(np.linalg.norm(TN))
    return T, T_norm


def dynamical_maps(t, d, basis_map):
    """The dynamical map ``E(t)`` as a ``(d^2, d^2)`` Liouville-space matrix.

    Column ``(i, j)`` is the propagated basis operator ``|i><j|``, so the whole
    matrix maps an initial density matrix (flattened) to its value at time index
    ``t``.  Assemble one per time step and hand the list to
    :func:`transfer_mat`.
    """
    r = np.zeros([d * d, d * d], dtype=np.complex128)
    for col, index in enumerate(it.product(range(d), repeat=2)):
        r[:, col] = map_basis_op(index, t, basis_map).reshape(d * d)
    return r


def predict_density_mat(t, T, r_init):
    """Extrapolate the density matrix to time index ``t`` with transfer tensors ``T``.

    Each new step is ``rho_n = sum_k T_k rho_{n-k}`` over the retained memory
    depth ``len(T)``, seeded with the directly-simulated ``r_init``.  Cost per
    step is a few matrix products regardless of how far out ``t`` is -- that is
    the whole point of the method.

    Requires ``t >= len(T) == len(r_init) > 0``: you need at least as much
    simulated history as the memory depth you kept.  Returns the full trajectory
    including the seed.
    """
    assert t >= len(T) == len(r_init) and len(T) > 0
    r = copy.deepcopy(r_init)
    diff = t - len(r_init)
    for i in range(diff):
        r_relevant = r[:-len(T) - 1:-1]
        rho = np.einsum('Nij,Njk->ik', T, r_relevant)
        r = np.append(r, [rho], axis=0)
    return r
