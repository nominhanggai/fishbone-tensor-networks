from model import FishBoneH, kron, _c
from fishbone import init
import numpy as np
from numpy import exp, tanh
from scipy.linalg import expm

# [[array([1., 3.]), array([1., 3.])], [array([1., 3.]), array([1., 3.])]]
# [[array([0.56418958, 1. ]), array([0.56418958, 1. ])], [array([0.56418958, 1.  ]), array([0.56418958, 1. ])]]

def sigmaz(d=2):
    z = np.zeros([d, d])
    z[0, 0] = 0
    z[1, 1] = 0
    return z

def sigmax(d=2):
    z = np.zeros([d, d])
    z[0, 1] = 1
    z[1, 0] = 1
    return z
sys = sigmax() + sigmaz()

intit = np.array([[1/np.sqrt(2)],[1/np.sqrt(2)]])

for n in range(0,201):
    final = expm(-1j * n* sys * 0.001) @ intit
    print([x*x.conj() for x in final])