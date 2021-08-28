import numpy as np
from fishbonett.backwardSpinBosonMultiChannel import SpinBoson
import scipy as sp
from examples.multipleAcceptor.electronicParametersAndVibronicCouplingDA import coupMol2_CT, coupMol2_LE, freqMol1_GR, \
    freqMol2_LE, coupMol1_LE, coupMol1_CT
from fishbonett.lanczos import lanczos

bath_length = 162 * 2
phys_dim = 20
bond_dim = 1000

a = [phys_dim] * bath_length
pd = a[::-1] + [2]

coup_num_LE = np.array(coupMol2_LE)
coup_num_CT = np.array(coupMol2_CT)

freq_num = np.array(freqMol2_LE)

# coup_num_LE = coup_num_LE * freq_num / np.sqrt(2)  # + list([1.15*x for x in back_coup])
# coup_num_CT = coup_num_CT * freq_num / np.sqrt(2)  # + list([-1.15*x for x in back_coup])

coup_mat = [np.diag([x, y]) for x, y in zip(coup_num_LE, coup_num_CT)]
reorg1 = sum([(coup_num_LE[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg2 = sum([(coup_num_CT[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
print("Reorg", reorg1, reorg2)
print(f"Len {len(coup_mat)}")

temp = 200
eth = SpinBoson(pd, coup_mat=coup_mat, freq=freq_num, temp=temp)
np.set_printoptions(suppress=True)

coup = np.random.rand(400)
freq = np.random.rand(400)
d = 162 + 162
coup = eth.coup_mat_np[:, 0, 0][:d]
# coup[0] = 0.0001
# coup[1] = 0
print(f"Coup {coup}")
freq = eth.freq[:d]
# print(freq,coup)
# freq.sort()
# coup = [1,1.5]*10
# freq = [2,10]*10
# print(f"Original Freq: {freq}")
tri, Q = lanczos(A=np.diag(freq), p=coup)

phase_func = lambda lam, t: np.exp(-1j * lam * (t))


def dn(t):
    phase_factor = np.array([phase_func(w, t) for w in freq])
    d_nt_mat = [np.einsum('k,k,k', coup, Q[:, n], phase_factor) for n in range(len(freq))]
    return d_nt_mat

#
# dk = np.array([np.abs(dn(0.002*t/10)) for t in range(0, 400)])
# dk.astype('float32').tofile('./dk.dat')

res = np.diagonal(Q.T @ Q - np.eye(Q.shape[0]))
res = res @ res
print(res)

eig = np.linalg.eigvals(tri)
eig.sort()
# print(f"Lanczos Eig {eig}")
print(f"diff {np.sqrt(eig @ freq) / np.linalg.norm(freq)} {list(eig - freq)}")
print(f"Residual: {res}")
