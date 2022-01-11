#####################
# Linear Time Iteration
#####################

import numpy as np
import matplotlib.pyplot as plt
from PB_Assignment2 import sol, params
from Assignment3 import plots

np.random.seed(123)

A = np.array([[1,0,1/params['sigma']],[-params['kappa'],1, 0],[0,0,1]])

M = np.array([[1, 1/params['sigma'], 0], [0, params['beta'],0],[0,0,0]])

D = np.array([[0,0,0],[0,0,0],[params['phi_2'],params['phi_1'], 0]])

print(A)
print(M)
print(D)

# 1

F_init = np.random.randn(3,3)

def F_update(F, A = A, M = M, D = D):
    return np.linalg.inv(A - M @ F) @ D

Fnew = F_update(F_init)
print(Fnew)

# 2
F = F_update(Fnew)
print(F)

# 3

def get_Q(A,M,F):
    return np.linalg.inv(A - M @ F)


# 4

rho_e = 0.8

Anew = np.array([[1,0,0,0],[0,1,0,1/params['sigma']],[-1,-params['kappa'],1, 0],[0,0,0,1]])

Mnew = np.array([[0,0,0,0],[0,1, 1/params['sigma'], 0], [0,0, params['beta'],0],[0,0,0,0]])

Dnew = np.array([[rho_e,0,0,0],[0,0,0,0],[0,0,0,0],[0,params['phi_2'],params['phi_1'], 0]])

def F_update(F, A = Anew, M = Mnew, D = Dnew):
    return np.linalg.inv(A - M @ F) @ D

Fnew = F_update(np.random.randn(4,4))

# 5
i = 0
while (abs(F - Fnew) > np.ones((4,4)) *1e-10).any():
    F = Fnew
    Fnew = F_update(F)
    #i += 1
    #print(i)

F_final = Fnew
print(F_final)

#6 

Q = get_Q(Anew, Mnew, F_final)
print(Q)

# 8
N = 20
eps = 0.01

z = [Q * eps]
e = [eps]

for i in range(N):
    et = rho_e * e[-1]
    zt = F @ np.array([[et],[0],[0],[0]], dtype=float)
    e.append(et)
    z.append(zt)

error = [x[0][0] for x in z]
output = [x[1][0] for x in z]
inflation = [x[2][0] for x in z]
interest = [x[3][0] for x in z]


# 9

plots(paths = [error,output, inflation, interest], path_names=['error', 'output', 'inflation', 'interest rate'])

