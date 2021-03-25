from fishbonett.stuff import sd_zero_temp
from scipy.integrate import *
from fishbonett.model import _c
from fishbonett.stuff import drude1
import numpy as np
# eta = 4.11
# f = lambda w: sd_zero_temp(w)/w
# g=1000
# a = quad(f, 0, g)
# print(a)


from fishbonett.stuff import sigma_z, sigma_x, natphys
f = lambda w: drude1(w, 100)* np.cos(w*0.2)*w
a = quad(f, 0, 600)
print(a)

