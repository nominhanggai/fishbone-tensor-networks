import numpy as np
import copy
import itertools as it


def read_rho(label, t):
    r = np.load(f"output/density_mat_{label}.npy")
    return r[t]


def map_basis_op(index, t, dict):
    # print(index, t)
    if index[0] == index[1]:
        id_ = dict[index]
        return read_rho(id_, t)
    if index[0] < index[1]:
        id1 = dict[index][0]
        id2 = dict[index][1]
        r1 = read_rho(id1, t)
        r2 = read_rho(id2, t)
        id3 = dict[(index[0], index[0])]
        id4 = dict[(index[1], index[1])]
        r3 = read_rho(id3, t)
        r4 = read_rho(id4, t)
        return r1 + 1j * r2 - (1 + 1j) * (r3 + r4) / 2
    if index[0] > index[1]:
        index_ = (index[1], index[0])
        id1 = dict[index_][0]
        id2 = dict[index_][1]
        r1 = read_rho(id1, t)
        r2 = read_rho(id2, t)
        id3 = dict[(index_[0], index_[0])]
        id4 = dict[(index_[1], index_[1])]
        r3 = read_rho(id3, t)
        r4 = read_rho(id4, t)
        return r1 - 1j * r2 - (1 - 1j) * (r3 + r4) / 2


def transfer_mat(lt_map):
    """
    Args:
        lt_map (): a list of dynamical maps in the Liouville space, i.e., the basis is {|n>|m>}.

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


def dynamical_maps(t, d):
    r = np.zeros([d * d, d * d], dtype=np.complex128)
    for n, index in it.product(range(d), repeat=2):
        r[:, n] = map_basis_op(index, t, dict).reshape(d * d)
    return r


def predict_density_mat(t, T, r_init):
    assert t >= len(T) == len(r_init) and len(T) > 0
    r = copy.deepcopy(r_init)
    diff = t - len(r_init)
    for i in range(diff):
        r_relevant = r[:-len(T) - 1:-1]
        rho = np.einsum('Nij,Njk->ik', T, r_relevant)
        r = np.append(r, [rho], axis=0)
    return r
