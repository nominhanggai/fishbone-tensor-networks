import numpy as np
import itertools as it
from numpy.linalg import norm

"""
Boyslocalization to obtain diabatic couplings
The Journal of chemical physics, 129(24), 244101.
Jacobi sweeps are used to find maximum of the Boys function.
"""

s3_mu = [2.2471, 0.2322, -0.3287]
s2_mu = [27.1776,0.2494,-1.4186]
s1_mu = [28.7418,0.0702,-1.3821]
s3s2_mu = np.array([-1.42466459, -0.11795590, 0.10055223])*2.5412
s3s1_mu = np.array([0.22303506,0.40015346,-0.17319997])*2.5412
s2s1_mu = np.array([0.15068858,-2.56353252,0.98092125])*2.5412
# Above are dipole moments at the LET geometry. 2.5412 Debye = 1 a.u. diople
s3s2_elem = np.zeros([3,3]); s3s2_elem[0, 1]=1; s3s2_elem=s3s2_elem+s3s2_elem.T
s3s1_elem = np.zeros([3,3]); s3s1_elem[0, 2]=1; s3s1_elem=s3s1_elem+s3s1_elem.T
s2s1_elem = np.zeros([3,3]); s2s1_elem[1, 2]=1; s2s1_elem=s2s1_elem+s2s1_elem.T
s3_elem = np.zeros([3,3]); s3_elem[0, 0]=1
s2_elem = np.zeros([3,3]); s2_elem[1, 1]=1
s1_elem = np.zeros([3,3]); s1_elem[2, 2]=1

mu_mat_3 = np.kron(s3s2_elem, s3s2_mu) + np.kron(s3s1_elem, s3s1_mu) + np.kron(s2s1_elem, s2s1_mu) + \
         np.kron(s3_elem, s3_mu) + np.kron(s2_elem, s2_mu) + np.kron(s1_elem, s1_mu)

mu_mat_3 = mu_mat_3.reshape(3,3,3)

test_mat_3 = np.array(mu_mat_3)

mu_mat_2 = np.array([
    [[0.0958,-0.7722,0.1177],[1.238206485004, 0.22711212639999998, -0.08586836777599999]],
    [[1.238206485004, 0.22711212639999998, -0.08586836777599999], [-27.6663,-5.9275,1.2562]]
])

mu_mat_2_config1 = np.array([
    [[-1.1410,-0.9247,-0.6088], [-2.84510477626, -0.715646213116, -0.409709899928]],
    [[-2.84510477626, -0.715646213116, -0.409709899928], [-25.3102, -6.2400, -3.7840]]
])
# print(mu_mat[:,:,0])
# print(s2s1_elem, s3s2_elem, s3s1_elem)





def boys_func(mat_mu):
    dim = mat_mu.shape[0]
    comb = it.combinations(range(dim), 2)
    comb = list(comb)
    boys_l = []
    for p in comb:
        i = p[0]
        j = p[1]
        boys_l.append(
            norm(mat_mu[i,i,:] - mat_mu[j,j,:])**2
        )
    return sum(boys_l)



def boys_loc(mat_mu, u_final):
    dim = mat_mu.shape[0]
    comb = it.combinations(range(dim), 2)
    comb = list(comb)
    u_final = u_final
    mat_mu_after = mat_mu.copy()
    boys_value_0 = boys_func(mat_mu)
    for p in comb:
        print(p)
        i = p[0]
        j = p[1]

        mu_ij = mat_mu_after[i,j]
        mu_ii = mat_mu_after[i,i]
        mu_jj = mat_mu_after[j,j]
        F = norm(mu_ij)**2 - .25*norm(mu_ii-mu_jj)**2
        G = mu_ij@(mu_ii-mu_jj)
        theta1 = np.arccos(-F/np.sqrt(F**2+G**2))
        theta2 = np.arcsin(G/np.sqrt(F**2+G**2))
        # See Joseph's paper for the theory.
        t1_l = [theta1 + 2*k*np.pi for k in range(-2,3)] + [2*k*np.pi -theta1 for k in range(-1,1)]
        t2_l = [theta2 + 2*k*np.pi for k in range(-2,3)] + [(2*k+1)*np.pi - theta2 for k in range(-1,1)]
        # print("theta 1", theta1, np.sort(t1_l))
        # print("theta 2", theta2, np.sort(t2_l))
        # print(-F/np.sqrt(F**2+G**2), [np.cos(x) for x in t1_l])
        # print(G / np.sqrt(F ** 2 + G ** 2), [np.sin(x) for x in t2_l])
        val_l = [i for i, j in it.product(t1_l, t2_l) if np.abs(i-j) <=0.0001]
        key = np.argmin(np.abs(val_l))
        theta = .25*val_l[key]
        # print(val_l[key])
        u = np.eye(dim)
        u[i, j] = np.sin(theta)
        u[j, i] = -np.sin(theta)
        u[i, i] = u[j, j] = np.cos(theta)
        mat_mu_after = np.einsum('ij,jkX,kl->ilX', u, mat_mu_after, u.T)
        u_final = u@u_final
    boys_value = boys_func(mat_mu_after)
    return u_final,mat_mu_after,boys_value, boys_value_0

u_final = np.eye(3)
boys_value =0; boys_value_0 = 1;
while(np.abs(boys_value - boys_value_0)>0.001):
    u_final, mu_mat_3, boys_value, boys_value_0 = boys_loc(mu_mat_3, u_final)
    print(boys_value, boys_value_0)


u = u_final
H = np.diag([0, -0.1331, -0.2734])
# H = np.diag([0,  -0.0964])
# H = np.diag([0, -0.1288])
print(u.T@H@u)
print(repr(u@H@u.T*8065.54429))

# print(u@test_mat_3[:,:,2]@u.T)
# print(test_mat_3[:,:,0])

"""
 [[-0.00284977 -0.01795822 -0.00842691]
 [-0.01795822 -0.19976618  0.07138905]
 [-0.00842691  0.07138905 -0.20388406]]
"""