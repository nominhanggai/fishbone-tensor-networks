from fishbonett.stuff import sd_zero_temp,drude1, temp_factor, sigma_z
from scipy.integrate import *
from fishbonett.model import _c
import numpy as np
import fishbonett.recurrence_coefficients as rc
import sympy
from opt_einsum import contract as einsum
from sympy.utilities.lambdify import lambdify
from scipy.linalg import expm
import itertools

def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    return expm(-dt * 1j * H)

f = lambda w: sd_zero_temp(w)
leng = 2
domain = [0,350]

# def get_coupling(j, domain, g, ncap=20000, n=leng):
#     alphaL, betaL = rc.recurrenceCoefficients(
#         n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
#     )
#     w_list = g * np.array(alphaL)
#     k_list = g * np.sqrt(np.array(betaL))
#     k_list[0] = k_list[0] / g
#     return w_list, k_list

# w, k= get_coupling(j=f, domain=domain, g=1, ncap=50000, n = leng)
w = np.array([147.99138037, 150.89261887])
k = np.array([74.76990594, 76.06207614])
k[1] = 0
print(w,k)


coup = np.diag(w) + np.diag(k[1:],1)+ np.diag(k[1:],-1)
freq, coef= np.linalg.eig(coup)

print(coef, freq)
print(w,k)
he_dy = (np.eye(2) + sigma_z) / 2
h1e = 0.5 * sigma_z
dim =20
c = _c(dim)
e = lambda lam, t, delta: (np.exp(-1j*lam*(t+delta)) - np.exp(-1j*lam*t))/(-1j*lam)
e = lambda lam, t, delta: np.exp(-1j * lam * (t+delta/2)) *delta
phase_factor = np.array([e(w, 0, 0) for w in freq])

k0 = k[0]
j0 = k0 * coef.T[0, :]  # interaction strength in the diagonal representation
delta = 0.002
num_steps = 100
p_osc = [1] + [0]*(dim-1)
psi = np.kron(np.kron(p_osc,p_osc),
               [1/np.sqrt(2),1/np.sqrt(2)]
               )
p= []
p1 = []

phase_factor = np.array([e(w, 1, 0.002) for w in freq])
# print(f'phase_factor {phase_factor}')
d_nt = [einsum('k,k,k', j0, coef[:, n], phase_factor) for n in range(len(freq))]
print(f'd_nt {d_nt}')
d_nt = d_nt[::-1]
h1 = np.kron(d_nt[1] * c + d_nt[1].conjugate() * c.T, he_dy) + np.kron(np.eye(dim),h1e) * delta
print(h1)
print(calc_U(h1, 1))

for tn in range(num_steps):
    phase_factor = np.array([e(w, tn*delta, delta) for w in freq])
    d_nt = [einsum('k,k,k', j0, coef[:, n], phase_factor) for n in range(len(freq))]
    d_nt = d_nt[::-1]
    h0 = np.kron(d_nt[0]*c + d_nt[0].conjugate()*c.T, np.eye(dim))
    h0 = np.kron(h0, he_dy)
    h1 = np.kron(np.eye(dim), d_nt[1]*c + d_nt[1].conjugate()*c.T)
    h1 = np.kron(h1, he_dy)
    print(h0.shape, h1.shape)
    H = h0+h1 +\
        np.kron(np.kron(np.eye(dim),np.eye(dim)),
                h1e) * delta
    u = calc_U(H, 1)
    print(f'u is {u}')
    psi = u@psi
    rho = np.kron(psi.conj(), psi)
    rho = rho.reshape(dim**2,2,dim**2,2)
    rho = np.trace(rho, axis1=0, axis2=2)
    p.append(np.abs(rho[0,1]))
    print(f'tn: {tn}')

# psi = np.kron(np.kron(p_osc,p_osc),
#                [1/np.sqrt(2),1/np.sqrt(2)]
#                )

# for tn in range(num_steps):
#     h0 = k[1]* (np.kron(c, c.T) + np.kron(c.T,c)) + w[1]* np.kron(c.T@c, np.eye(dim))
#     h0 = np.kron(h0, np.eye(2))
#     h1 = k[0]* np.kron(np.eye(dim), c.T+c) + w[1]* np.kron(np.eye(dim), c.T@c)
#     h1 = np.kron(h1, he_dy)
#     print(h0.shape, h1.shape)
#     H = h0+h1 +\
#         np.kron(np.kron(np.eye(dim),np. eye(dim)),
#                 h1e)
#     u = calc_U(H, delta)
#     psi = u@psi
#     rho = np.kron(psi.conj(), psi)
#     rho = rho.reshape(dim**2,2,dim**2,2)
#     rho = np.trace(rho, axis1=0, axis2=2)
#     p1.append(np.abs(rho[0,1]))
#     print(f'tn: {tn}')

print(p)
# print(p1)
