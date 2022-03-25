import numpy as np

dict = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): (3, 4), (1, 2): (5, 6), (0, 2): (7, 8)}


def read_rho(id, t):
    r = np.fromfile(f"pop1_50_10_0.0005_{id}.dat")
    return np.array(r[t])


def map_basis_op(index, t, dict):
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
        return map_basis_op(index_, t, dict).conj()

def B_mat(d):
    r = np.eye(d**2).reshape(d, d, d**2)
    r = np.transpose(r, [2,0,1])
    print(r.shape)
    B = np.einsum('Lih,Mij,Kjk,Nhk->MNKL', r.conj(), r, r, r.conj())
    B = B.reshape(d ** 4, d ** 4)
    print(B)
    C = np.einsum('Mij,Kjk,Nkl->MNKil', r, r, r.conj())
    C = C.reshape(d ** 4, d ** 4)
    print(C)
    return B, C

if __name__ == "__main__":
    B, C = B_mat(2)
    print(np.linalg.norm(B-C))
    d = 3
    r = np.eye(d ** 2).reshape(d, d, d ** 2)
    r = np.transpose(r, [2, 0, 1])
    print(np.einsum('Mij,Nji', r, r))




