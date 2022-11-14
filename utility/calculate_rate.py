
"""
Example calculation. Reproduce Figure 2 in dx.doi.org/10.1021/jp400462f | J. Phys. Chem. A 2013, 117, 6196−6204
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from legendre_discretization import get_vn_squared, get_approx_func
from golden_rule_rate import fgr_rate, fgr_rate_by_order
import itertools as it
from golden_rule_rate_3 import *

# Lorentzian spectral density parameters. Atomic units.
reorg_e = 2.39e-2
Omega = 3.5e-4
kbT = 9.5e-4/100
eta = 1.2e-3
domain = [0, 5e-3]
C_DA = 5e-4
j = lambda w: 0.5 * (4 * reorg_e) * Omega ** 2 * eta * w / ((Omega ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2)

w, v_sq = get_vn_squared(j, 100, domain)
v = np.sqrt(v_sq)
print("Discrete Reorganization E", np.sum(v_sq / w / np.pi))

aa_coupling = 5e-3
e = np.linspace(0.015, 0.03, 30)
print(len(e))

fgr_rate3 = np.vectorize(
lambda ei: fgr_rate3_correction_order_quad([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
                                           2000, 0)
)(e)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(e, fgr_rate3* 2 / (2.418884326509e-5), 'd-', label=f'Quadrature {1}')
# ax.plot(e, fgr_rate3_correction_1_vegas, 'd-', label=f'vegas {1}')
# ax.plot(e, fgr_rate3_correction_1_mcmc, 'd-', label=f'MCMC {1}')

ax.legend()
ax.set_xlabel('E (a.u.)')
ax.set_ylabel('Rate (a.u.)')
ax.set_xlim(0.015, 0.03)

x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.4)

fig.savefig('test.png')

print("finished")
