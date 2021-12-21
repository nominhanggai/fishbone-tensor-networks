#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:20:02 2021

@author: michaelwu, Mulliken
"""

import numpy as np
import fishbonett.recurrence_coefficients as rc
import matplotlib.pyplot as plt


def get_coupling(n, j, domain, g=1, ncap=20000):  # j=weight function
    alphaL, betaL = rc.recurrenceCoefficients(
        n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
    )
    j = lambda x: j(x) * np.pi
    alphaL = g * np.array(alphaL)
    betaL = g * np.sqrt(np.array(betaL))
    betaL[0] = betaL[0] / g
    return alphaL, betaL  # k=beta, w=alpha, beta[0] drop


def get_Vn_squared(J, n: int, domain, ncap=60000):
    weight = lambda x: 1 * np.pi
    alpha, beta = get_coupling(n, weight, domain=domain, ncap=ncap)
    M = np.diag(alpha) + np.diag(beta[1:], -1) + np.diag(beta[1:], 1)
    eigval, eigvec = np.linalg.eig(M)
    W = (eigvec[0, :]) ** 2 * (domain[1] - domain[0])
    V_squared = [J(w) * W[i] for i, w in enumerate(eigval)]
    return eigval, np.array(V_squared)


def get_approx_func(J, n, domain, epsilon, ncap=20000):
    delta = lambda x: 1 / np.pi * epsilon / (epsilon ** 2 + x ** 2)
    w, V_squared = get_Vn_squared(J, n, domain, ncap)
    J_approx = lambda x: np.sum([vi * delta(x - wi) for wi, vi in zip(w, V_squared)])
    return J_approx


if __name__ == '__main__':
    x = np.linspace(0, 20, 10000)

    drude = lambda x, gam, lam: 2 * lam * gam * x / (x ** 2 + gam ** 2)
    J = lambda x: drude(x, 5, 100)
    J_approx = get_approx_func(J, 1000, [0, 20], 0.05)
    disc = []
    for i in range(len(x)):
        disc += [J_approx(x[i])]

    plt.plot(x, J(x), 'r-')
    plt.plot(x, disc, 'k-')
    plt.show()
