#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:59:19 2021

@author: michaelwu
"""

import numpy as np
import fishbonett.recurrence_coefficients as rc
import matplotlib.pyplot as plt
from math import fsum
import scipy.linalg as lin

def get_coupling(n, j, domain, g=1, ncap=20000):#j=weight function
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        j = lambda x: j(x)*np.pi
        alphaL = g * np.array(alphaL)
        betaL = g * np.sqrt(np.array(betaL))
        betaL[0] = betaL[0] / g
        return alphaL, betaL # k=beta, w=alpha, beta[0] drop


def get_Vn(J, n, domain, ncap=60000):
    weight = lambda x : 1 * np.pi
    alpha, beta = get_coupling(n, weight, domain=domain, ncap=ncap)
    M = np.diag(alpha) + np.diag(beta[1:], -1) + np.diag(beta[1:], 1)
    eigval, eigvec = np.linalg.eig(M)
    W = (eigvec[0, :]) **2 * (domain[1] - domain[0])
    V_squared = [J(w)*W[i] for i, w in enumerate(eigval)]
    return eigval, V_squared


def get_approx_func(J, n, domain, epsilon, ncap=20000):
    delta = lambda x: 1/np.pi * epsilon / (epsilon ** 2 + x ** 2)
    w, V = get_Vn(J, n, domain, ncap)
    J_approx = lambda x: np.sum([vi * delta(x-wi) for wi, vi in zip(w,V)])
    return J_approx, w, V

def lanczos(A, p):
    A = np.array(A)
    q = np.array(p).copy()
    n = k = A.shape[0]
    Q = np.zeros((n, k + 1))
    Q[:, 0] = q / np.linalg.norm(q)
    # print(Q[:,0])
    alpha = 0
    beta = 0

    for i in range(k):
        if i == 0:
            q = np.dot(A, Q[:, i])
            # print(f"q1 {q}")
        else:
            q = np.dot(A, Q[:, i]) - beta * Q[:, i - 1]
            # print(f"q1 {q}")
        alpha = np.dot(q.T, Q[:, i])
        # print(f"alpha {alpha}")
        q = q - Q[:, i] * alpha
        # print(f"q2 {q}")
        q = q - np.dot(Q[:, :i], np.dot(Q[:, :i].T, q))  # full re-orthogonalization
        # print(f"q3 {q}")
        beta = np.linalg.norm(q)
        # print(f"beta {beta}")
        Q[:, i + 1] = q / beta
        # print(beta)

    Q = Q[:, :k]

    Sigma = np.dot(Q.T, np.dot(A, Q))
    return Sigma, Q

if __name__ == '__main__':
    x = np.linspace(0, 20, 10000)
    
    drude = lambda x, gam, lam: 2 * lam * gam * x / (x ** 2 + gam ** 2)
    ohmic = lambda x, eta, omegac: eta * x * np.exp(-x / omegac)
    J1 = lambda x: drude(x, 5, 100)
    J2 = lambda x: ohmic(x, 1, 3)
    J_approx1, w1, V1= get_approx_func(J1, 1000, [0, 20], 0.05)
    J_approx2, w2, V2 = get_approx_func(J2, 1000, [0, 20], 0.05)
    
    _, Q1 = lanczos(np.diag(w1), V1)
    
    color = ['darkred','red','darkorange','gold','yellow','lawngreen','green',\
             'darkgreen','deepskyblue','blue','darkblue','purple','indigo']
    im = 1j
    for t in range(len(color)):
        time = t * 0.1
        exp = np.exp(im * w1 * time)
        d = np.einsum("i, i, ij", V1, exp, Q1)
        plt.plot(np.arange(1, len(d) + 1), np.abs(d), color[t], label = t)
    plt.xlim(0, 40)
    plt.legend(loc = 'upper right')
    plt.show()
    
    for t in range(len(color)):
        time = t * 0.1
        exp = np.exp(im * w1 * time)
        d = np.einsum("i, i, ij", V2, exp, Q1)
        plt.plot(np.arange(1, len(d) + 1), np.abs(d), color[t], label = t)
    plt.xlim(0, 40)
    plt.legend(loc = 'upper right')
    plt.show()
    '''    
    disc = []
    for i in range(len(x)):
        disc += [J_approx1(x[i])]

    plt.plot(x, J1(x), 'r-')
    plt.plot(x, disc, 'k-')
    plt.show()
    '''
