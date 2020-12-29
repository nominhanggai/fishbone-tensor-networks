from model import SpinBoson, kron, _c
from fishbone import SpinBoson1D
import numpy as np
from numpy import exp, tanh
from numpy.linalg import norm
from scipy.linalg import expm, svd
from scipy.integrate import quad as integrate
def coth(x):
    return (exp(2*x)-1)/(exp(2*x)+1)

def sigmaz(d=2):
    z = np.zeros([d, d])
    z[0, 0] = 1
    z[1, 1] = -1
    return z

def sigmax(d=2):
    z = np.zeros([d, d])
    z[0, 1] = 1
    z[1, 0] = 1
    return z


def temp_factor(temp, w):
    beta = 1/(0.6950348009119888*temp)
    return 0.5 * (1. + 1. / tanh(beta * w / 2.))


bath_length = 20
a = [2]*bath_length

pd = a + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
etn.B[-1][0,1,0] = 1/np.sqrt(2)
etn.B[-1][0,0,0] = 1./np.sqrt(2)
# electronic couplings
# tda = 100.0
e = 1000.

# spectral density parameters
g = 2500
eth.domain = [0, g]
ncap = 20000
lambd = 75
Omega = 150
s = 2
j = lambda x: lambd * ((x / Omega) ** s) * np.exp(-x / Omega)

eth.sd = j
# eth.sd = j
eth.he_dy = (np.eye(2) + sigmaz(2))/4
# eth.he_dy = np.eye(2)
eth.h1e = 1000/2 *sigmaz() #+ 500*sigmax()


eth.build(g)
etn.U = eth.get_u(dt=0.0001)


p = []
be = etn.B[-1]
s = etn.S[-1]
c = np.einsum('Ibc,IJ->Jbc', be, np.diag(s))
c1 = c[0, 0, 0]
c2 = c[0, 1, 0]
p.append(c1*c2)

for tn in range(1000):
    print("ni complete", tn)
    for j in range(0, bath_length):
        print("j==", j)
        etn.update_bond(j, 50, 1e-15)
    # print(etn.B[1][:,:,0] )
    # print(etn.S[1])
    # print([x.shape for x in etn.U])
    be = etn.B[-1]
    s = etn.S[-1]
    c = np.einsum('Ibc, IJ->Jbc', be, np.diag(s))
    print(c[:,:,0])
    print("S", etn.S)
    cc = c.conj()
    c1 = c[0,0,0]
    c2 = c[0,1,0]
    c3 = c[1,0,0]
    c4 = c[1,1,0]
    p.append(c1*c2.conj()+c3*c4.conj())

print("population", [np.abs(x) for x in p])

print(eth.w_list)
print(eth.k_list)
dim = 2
c = _c(dim)
hb1 = np.kron(np.eye(2), np.kron(449.97993819*c.T@c, np.eye(2)) )
hb2 = np.kron(np.kron(748.75012497*c.T@c, np.eye(2)) , np.eye(2))
hs = np.kron(np.eye(2), np.kron(np.eye(dim), eth.h1e))
hi1 = np.kron(np.eye(dim), 84.62805478*np.kron(c.T+c, eth.he_dy))
hi2 = np.kron(259.72266674*(np.kron(c.T,c) + np.kron(c,c.T)), np.eye(dim))
h = hb1+hb2+hs+hi1 + hi2
def U(dt):
    return expm(-1j*h*dt)
# u = U.reshape(10,2,10,2)
state = np.kron(np.array([1]+[0]), np.kron(np.array([1]+[0]), np.array([1/np.sqrt(2)]+[1/np.sqrt(2)])))
# print(u.shape,state.shape)
p = []
for dt in np.linspace(0,0.1,50):
    s = U(dt)@state
    s = s.reshape(4,2)
    r = np.einsum("IJ, Ij->Jj ", s.conj(), s)
    p.append(np.abs(r[0,1]))
print(p)
# s = np.einsum('ijkl,kl->ij', u, state)
# Mini TEBD
h1 = np.kron(748.75012497*c.T@c, np.eye(2)) + 259.72266674*(np.kron(c.T,c) + np.kron(c,c.T))
h2 = np.kron(449.97993819*c.T@c, np.eye(2)) + 84.62805478*np.kron(c.T+c, eth.he_dy) + np.kron(np.eye(dim), eth.h1e)

dt = 0.001
U1 = [exp(-1j*h*dt).reshape(2,2,2,2) for h in [h1,h2]]
U = eth.get_u(0.001)

def g_state(dim):
    tensor = np.zeros(dim)
    tensor[(0,) * len(dim)] = 1.
    return tensor


Bs = [g_state([1,d,1]) for d in [2,2,2]]
Ss = [np.ones([1], np.float) for d in [2,2,2]]
Bs[-1][0,0,0] = 1/np.sqrt(2)
Bs[-1][0,1,0] = 1/np.sqrt(2)

def get_theta1(i):
    return np.tensordot(np.diag(Ss[i]), Bs[i], [1, 0])

def get_theta2(i):
    j = (i + 1)
    return np.tensordot(get_theta1(i), Bs[j], [2, 0])

print("A", get_theta2(1))


def split_truncate_theta(theta, i, chi_max, eps):
    (chi_left_on_left, phys_left,
     phys_right, chi_right_on_right) = theta.shape
    theta = np.reshape(theta, [chi_left_on_left * phys_left,
                               phys_right * chi_right_on_right])
    A, S, B = svd(theta, full_matrices=False)
    chivC = min(chi_max, np.sum(S > eps))
    # keep the largest `chivC` singular values
    piv = np.argsort(S)[::-1][:chivC]
    A, S, B = A[:, piv], S[piv], B[piv, :]
    S = S / np.linalg.norm(S)
    A = np.reshape(A, [chi_left_on_left, phys_left, chivC])  # A: {vL*i, chivC} -> vL i vR=chivC
    B = np.reshape(B, [chivC, phys_right, chi_right_on_right])  # B: {chivC, j*vR} -> vL==chivC j vR
    A = np.tensordot(np.diag(Ss[i] ** (-1)), A, [1, 0])  # vL [vL'] * [vL] i vR -> vL i vR
    A = np.tensordot(A, np.diag(S), [2, 0])  # vL i [vR] * [vR] vR -> vL i vR
    Ss[i + 1] = S
    Bs[i] = A
    Bs[i + 1] = B


def update_bond(i, chi_max, eps):
    theta = get_theta2(i)
    U_bond = U[i]
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
    split_truncate_theta(Utheta, i, chi_max, eps)

# p = []
# be = Bs[-1]
# s = Ss[-1]
# c = np.einsum('Ibc,IJ->Jbc', be, np.diag(s))
# c1 = c[0, 0, 0]
# c2 = c[0, 1, 0]
# p.append(c1*c2)
#
# for tn in range(20):
#     print("ni complete", tn)
#     for j in range(0, bath_length):
#         print("j==", j)
#         update_bond(j, 50, 1e-15)
#     # print(etn.B[1][:,:,0] )
#     # print(etn.S[1])
#     # print([x.shape for x in etn.U])
#     be = Bs[-1]
#     s = Ss[-1]
#     c = np.einsum('Ibc, IJ->Jbc', be, np.diag(s))
#     print("be", be)
#     print("S", Ss)
#     cc = c.conj()
#     c1 = c[0,0,0]
#     c2 = c[0,1,0]
#     c3 = c[1,0,0]
#     c4 = c[1,1,0]
#     p.append(c1*c2.conj()+c3*c4.conj())
#
# print("population", [np.abs(x) for x in p])

# h1 = np.kron(748.75012497*c.T@c, np.eye(2)) + 259.72266674*(np.kron(c.T,c) + np.kron(c,c.T))
# h2 = np.kron(449.97993819*c.T@c, np.eye(2)) + 84.62805478*np.kron(c.T+c, eth.he_dy) + np.kron(np.eye(dim), eth.h1e)
# h1o = 259.72266674*(np.kron(c.T,c) + np.kron(c,c.T))
# h2o = 84.62805478*np.kron(c.T+c, eth.he_dy)
#
# print("h2oh", h1-eth.get_h2()[0][0], h2-eth.get_h2()[1][0])
# print(eth.get_h1()[1])
# print(eth.w_list,eth.k_list)
# print(U1)