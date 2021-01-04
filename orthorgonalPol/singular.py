from fishbonett.stuff import lorentzian, temp_factor
from scipy.integrate import *

eta = 4.11
f = lambda w: lorentzian(4.11, w)*temp_factor(300.,w)
a = quad(f, 0, 600)
print(a)

