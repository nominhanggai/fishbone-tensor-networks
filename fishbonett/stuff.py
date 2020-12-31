import numpy as np


'''
Pauli matrices
'''

# S^+
sp = np.float64([[0, 1], [0, 0]])
# S^-
sm = np.float64([[0, 0], [1, 0]])

sx = np.float64([[0, 1], [1, 0]])
sz = np.float64([[1, 0], [0, -1]])

# zero matrix block
s0 = np.zeros((2, 2))
# identity matrix block
s1 = np.eye(2)

'''
Boson creation operators
'''


'''Temperature factor'''
def temp_factor(temp, w):
    beta = 1/(0.6950348009119888*temp)
    return 0.5 * (1. + 1. / np.tanh(beta * w / 2.))
