import numpy as np
from scipy.linalg import svd as csvd
from numpy import exp
import fishbonett.recurrence_coefficients as rc
from copy import deepcopy as dcopy
import sympy
import scipy
from sympy.utilities.lambdify import lambdify
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, drude, _c
import multiprocessing as mp




class SpinBoson:

    def __init__(self, pd):
        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
        self.domain = [0, 1]
        self.he_dy = np.eye(self.pd_spin)
        self.h1e = np.eye(self.pd_spin)
        self.k_list = []
        self.w_lsit = []
        self.H = []
        self.coef= []
        self.freq = []
        self.phase = lambda lam, t, delta: (np.exp(-1j*lam*(t+delta)) - np.exp(-1j*lam*t))/(-1j*lam)
        self.phase = lambda lam, t, delta: np.exp(-1j * lam * (t+delta/2)) * delta

    def get_coupling(self, n, j, domain, g, ncap=20000):
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        _, _, self.h_squared = rc._j_to_hsquared(func=j, lb=domain[0], rb=domain[1], g=g)
        self.domain = domain
        return w_list, k_list

    def build_coupling(self, g, ncap):
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g, ncap)

    def poly(self):
        k = self.k_list
        w = self.w_list
        pn_list = [0, 1/k[0]]
        x = sympy.symbols("x")
        for i in range(1, len(k)):
            pi_1 = pn_list[i]
            pi_2 = pn_list[i - 1]
            pi = ((1 / k[i] * x - w[i - 1] / k[i]) * pi_1 - k[i - 1] / k[i] * pi_2).expand()
            pn_list.append(pi)
        pn_list = pn_list[1:]
        return [lambdify(x,pn) for pn in pn_list]

    def diag(self):
        w= self.w_list
        k = self.k_list
        self.coup = np.diag(w) + np.diag(k[1:], 1) + np.diag(k[1:], -1)
        freq, coef = np.linalg.eig(self.coup)
        return freq, coef

    def get_h2(self, t, delta):
        print("Geting h2")
        freq = self.freq
        coef = self.coef
        e = self.phase
        k0 = self.k_list[0]
        j0 = k0 * coef[0,:] # interaction strength in the diagonal representation
        print("Geting d's")
        # Permutation
        indexes = np.abs(freq).argsort()
        freq = freq[indexes]
        j0 = j0[indexes]
        # END Permutation
        d_nt = j0
        d_nt = d_nt[::-1]
        freq = freq[::-1]
        h2 = []
        for i, k in enumerate(d_nt):
            d1 = self.pd_boson[i]
            d2 = self.pd_spin
            c1 = _c(d1)
            kc = k.conjugate()
            f = freq[i]
            coup = np.kron(k*c1 + kc* c1.T, self.he_dy)
            site = np.kron(f*c1.T@c1, np.eye(d2))
            h2.append((delta*(coup+site), d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin
        site = delta*np.kron(np.eye(d1), self.h1e)
        h2[-1] = (h2[-1][0] + site, d1, d2)
        return h2

    def build(self, g, ncap=20000):
        self.build_coupling(g, ncap)
        print("Coupling Over")
        self.freq, self.coef = self.diag()

    def get_u(self, t, dt, mode='normal', factor=2):
        hee = self.get_h2(t, dt)
        self.H = [h[0] for h in hee]
        # print(print(self.H[0].shape), self.H[0], type(self.H[0]))

        # with mp.Pool(5) as p:
        #     U1 = p.map(calc_U, h)
        import time
        def wait_process_done(f, wait_time=0.001):
            # Monitor the status of another process
            if not f.is_alive():
                time.sleep(0.001)
            print('foo is done.')

        with mp.Manager() as manager:

            h_list = manager.list(self.H)
            print(type(h_list))
            # u_list = manager.list([i for i in range(len(h_list))])
            # idx_list = manager.list([i for i in range(len(h_list))])
            with mp.Pool(processes=4) as pool:
                U1 = pool.map(calc_U, h_list)
            U2 = U1
                # process = [0] * len(h_list)
            # for i in range(len(h_list)):
            #     process[i] = mp.Process(target=calc_U, args=(i, h_list, u_list))
            #     process[i].start()
            # for i in range(len(h_list)):
            #     process[i].join()
                # p.join()

        # U1 = [u for u in u_list]
        # U2 = [u for u in u_list]
        # for i, h_d1_d2 in enumerate(self.H):
        #     h, d1, d2 = h_d1_d2
        #     u = calc_U(h/factor, 1)
        #     r0 = r1 = d1  # physical dimension for site A
        #     s0 = s1 = d2  # physical dimension for site B
        #     u1 = u.reshape([r0, s0, r1, s1])
        #     u2 = np.transpose(u1, [1,0,3,2])
        #     U1[i] = u1
        #     U2[i] = u2
        #     print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2

def calc_U(H):
    # print(f"EXP {i}")
    # print(H[i].shape)
    print(type(H))
    return scipy.linalg.expm(- 1j * H)

if __name__ == '__main__':
    bath_length = 20
    phys_dim = 500
    a = [phys_dim] * bath_length
    pd = a[::-1] + [10]
    eth = SpinBoson(pd)

    # spectral density parameters
    g = 350
    eth.domain = [-g, g]
    temp = 1000
    j = lambda w: drude(w, 400) * temp_factor(temp, w)

    eth.sd = j

    eth.he_dy = np.eye(pd[-1])
    eth.h1e = eth.he_dy
    eth.build(g=350., ncap=20000)
    dt = 0.0005
    num_steps = 80
    # def get_u(t, dt, mode='normal', factor=2):
    factor = 2
    U1, U2 = eth.get_u(2*dt, dt,mode='normal')