import numpy as np

'''
Pauli matrices
'''

# S^+
sigma_p = np.float64([[0, 1], [0, 0]])
# S^-
sigma_m = np.float64([[0, 0], [1, 0]])

sigma_x = np.float64([[0, 1], [1, 0]])
sigma_y = np.complex64([[0, -1j], [1j, 0]])
sigma_z = np.float64([[1, 0], [0, -1]])

# zero matrix
sigma_0 = np.zeros((2, 2))
# identity matrix
sigma_1 = np.eye(2)

'''
Boson creation operators
'''

'''Temperature factor'''


def temp_factor(temp, w):
    beta = 1 / (0.6950348009119888 * temp)
    return 0.5 * (1. + 1. / np.tanh(beta * w / 2.))


'''Spectral densities'''


def sd_back(Sk, sk, w, wk):
    return Sk / (sk * np.sqrt(2 / np.pi)) * w * \
           np.exp(-np.log(np.abs(w) / wk) ** 2 / (2 * sk ** 2))


def sd_high(gamma_m, Omega_m, g_m, w):
    return 4 * gamma_m * Omega_m * g_m * (Omega_m ** 2 + gamma_m ** 2) * w \
           / ((gamma_m ** 2 + (w + Omega_m) ** 2) * (gamma_m ** 2 + (w - Omega_m) ** 2))


# zero-temperature spectral density
def sd_zero_temp(w):
    gamma = 5.
    Omega_1 = 181
    Omega_2 = 221
    Omega_3 = 240
    g1 = 0.0173
    g2 = 0.0246
    g3 = 0.0182
    S1 = 0.39
    S2 = 0.23
    S3 = 0.23
    s1 = 0.4
    s2 = 0.25
    s3 = 0.2
    w1 = 26
    w2 = 51
    w3 = 85
    return sd_back(S1, s1, w, w1) + sd_back(S2, s2, w, w2) \
           + sd_back(S3, s3, w, w3) + sd_high(gamma, Omega_1, g1, w) \
           + sd_high(gamma, Omega_2, g2, w) \
           + sd_high(gamma, Omega_3, g3, w)


def sd_zero_temp_prime(w):
    S1 = 0.39
    S2 = 0.23
    S3 = 0.23
    s1 = 0.4
    s2 = 0.25
    s3 = 0.2
    w1 = 26
    w2 = 51
    w3 = 85
    return sd_back(S1, s1, w, w1) + sd_back(S2, s2, w, w2) \
           + sd_back(S3, s3, w, w3)


# The parameters in lorentzian() are from  dx.doi.org/10.1021/jp400462f


def lorentzian(eta, w, lambd = 5245.,Omega = 77.):
    return 0.5 * lambd * (Omega ** 2) * eta * w / ((w ** 2 - Omega ** 2) ** 2 + (eta ** 2) * (w ** 2))


def drude1(w, lam):
    gam = 100. / 1.8836515673088531  # T^-1
    return 2 * lam * gam * w / (w ** 2 + gam ** 2)

def brownian(w, lam, gam, w0=1):
    return 2 * lam * gam * w0 ** 2 * w / ((w0 ** 2 - w ** 2) + gam ** 2 * w ** 2)
