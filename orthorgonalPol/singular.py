from fishbonett.stuff import sd_zero_temp
from scipy.integrate import *
from fishbonett.model import _c
from fishbonett.stuff import drude1
import numpy as np
import fishbonett.recurrence_coefficients as rc
import sympy
from sympy.utilities.lambdify import lambdify
# f = lambda w: sd_zero_temp(w)/w
# g=1000
# a = quad(f, 0, g)



from fishbonett.stuff import sigma_z, sigma_x, natphys
f = lambda w: drude1(w, 100)
leng = 10
domain = [0,350]

def get_coupling(j, domain, g, ncap=20000, n=leng):
    alphaL, betaL = rc.recurrenceCoefficients(
        n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
    )
    w_list = g * np.array(alphaL)
    k_list = g * np.sqrt(np.array(betaL))
    k_list[0] = k_list[0] / g
    _, _, h_squared = rc._j_to_hsquared(func=j, lb=domain[0], rb=domain[1], g=g)
    return w_list, k_list, h_squared

w, k, h_squared= get_coupling(j=f, domain=domain, g=1, ncap=20000, n = leng)
print(len(w),len(k))

x = sympy.symbols("x")

pn_list = [0, 1/k[0]]
print(w,k)
for i in range(1, len(k)):
    pi_1 = pn_list[i]
    pi_2 = pn_list[i-1]
    pi = (1 / k[i] * x - w[i-1] / k[i]) * pi_1 - k[i-1] / k[i] * pi_2
    pn_list.append(pi)
    print(f"i = {i}")

pn_list = pn_list[1:]
print(pn_list)

print(len(pn_list), len(k))


p2 = lambdify(x, pn_list[1])
p3 = lambdify(x, pn_list[5])
j = lambda x: h_squared(x)*p3(x)*p3(x)
a = quad(j, *domain)
print(a)