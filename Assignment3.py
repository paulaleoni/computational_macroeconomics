#####################
# Adaptive Learning
#####################

import numpy as np
import matplotlib.pyplot as plt
import warnings
from PB_Assignment2 import sol, params


np.random.seed(123)

def NKM(input, params = params, out_shock = False, int_shock=False, shock = None):
    '''
    input: input vector to model
    params: dictionary of parameters
    output: zt
    '''
    # add zeros to vector in case it has a smaller lenght than 9
    while len(input) < 9: 
        input = np.append(input,0)
    # extract C0 which are expected values of zt
    ey, epi, ei, = input[:3]
    # define A
    A = np.array([[1,0,1/params['sigma']],[-params['kappa'],1, 0],[0,0,1]])
    # get inverse of A
    A_inv = np.linalg.inv(A)
    # calculate B
    b1 = ey - 1/params['sigma'] *(-epi)
    b2 = params['beta'] *epi
    b3 = params['phi_1'] * epi + params['phi_2'] * ey
    B = np.array([[b1], [b2], [b3]]) +  np.array([[int(out_shock)],[0],[int(int_shock)]])
    if shock != None:
        B = np.array([[b1], [b2], [b3]]) +  np.array([[shock],[0],[int(int_shock)]])
    # get zt
    zt = np.dot(A_inv, B) 
    return zt.flatten()

def Rnew(Rold, gamma, v):
    v_prime = np.transpose(v)
    return Rold + gamma * ( v @ v_prime - Rold)

def Cnew(Cold, gamma, v, Rnew, z):
    v_prime = np.transpose(v)
    R_inv = np.linalg.inv(Rnew)
    return Cold + gamma * R_inv @ v @ (z - v_prime @ Cold)

def updating(gamma, shocks, initialC, initialR):
    '''
    use function Rnew and Cnew
    '''
    Ct = [initialC]
    Z = []
    R = initialR

    for e in shocks:
        v = np.array([[1],[e]])
        z = NKM(input = Ct[-1], shock = e)    

        rnew = Rnew(Rold = R, gamma = gamma, v = v)

        cnew = Cnew(Cold = Ct[-1], gamma = gamma, v = v, Rnew = rnew, z = z.reshape(1,3))

        Ct.append(cnew)
        Z.append(z)
        R = rnew

    return Ct, Z

def plots(paths, path_names, size = (10,10)):

    if len(paths) != len(path_names): warnings.warn('length of path and pathnames do not match')
    
    n = len(paths)
    fig, axes = plt.subplots(n, figsize = size)
    for i in range(n):
        axes[i].plot(paths[i])
        axes[i].set_title(path_names[i])
        axes[i].set
    fig.tight_layout()
    plt.show()



# 1
init = np.random.random(size = (2,3))

# 2
n = 100000
rand = np.random.randn(n)

# 3
gamma = 0.05
R_init = np.array([[1,1],[1,1]])


Cs, Z = updating(gamma = gamma, shocks = rand, initialC=init, initialR=R_init)


#4 
print('initial C:', Cs[0])
print('final C:', Cs[-1])
print('MSV solution:', np.array(sol).reshape(2,3).round(3))


# 5
output = [x[0] for x in Z]
inflation = [x[1] for x in Z]
interest = [x[2] for x in Z]

exp_output = [c[0][0] for c in Cs]
exp_inflation = [c[0][1] for c in Cs]

paths = [output, inflation, interest, exp_output, exp_inflation, rand]
path_names = ['output', 'inflation', 'interest rate', 'expexted output', 'expected inflation', 'shocks']

plots(paths, path_names)






