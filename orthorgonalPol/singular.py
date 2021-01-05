from fishbonett.stuff import lorentzian, temp_factor
from scipy.integrate import *

eta = 4.11
f = lambda w: lorentzian(4.11, w)*temp_factor(300.,w)
g=1000
a = quad(f, -g, g)
print(a)

