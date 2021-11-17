#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:20:02 2021

@author: michaelwu
"""

import numpy as np
import fishbonett.recurrence_coefficients as rc
import matplotlib.pyplot as plt
def J(x):
    J = x ** 2 * np.exp(-x)
    return J

def get_coupling(n, j, domain, g, ncap=20000):#j=weight function
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        return w_list, k_list # k=beta, w=alpha, beta[0] drop
def delta(x, epsilon = 0.1):
    delta = epsilon / (epsilon ** 2 + x ** 2)
    return delta
j = lambda x : 1
alpha, beta = get_coupling(100, j, [0, 2], 1)
beta = beta[1:]
M = np.diag(alpha) + np.diag(np.sqrt(beta), -1) + np.diag(np.sqrt(beta), 1)
lamb, eigen = np.linalg.eig(M)
W = eigen[0, :] ** 2
V = W * J(lamb)
x = np.linspace(0, 2, 1000)
disc = np.zeros(len(x))
for i in range(len(lamb)):
    disc += V[i] * delta(x - lamb[i])
plt.plot(x, J(x), 'r-')
plt.plot(x, disc, 'k-')
plt.show()





