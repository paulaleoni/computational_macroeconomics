#########################
# Computational Macroeconomics
# Assignment 2
# author: Paula Beck
########################


import numpy as np
from scipy.optimize import fsolve

np.random.seed(123)

# input parameters
params = {'sigma': 2,'kappa':0.3, 'beta':0.99, 'phi_1':1.5, 'phi_2':0.2}

'''
Task 1: 
Change your function from last week to accommodate the new model, as described above.
'''

def NKM(input, params = params, out_shock = False, int_shock=False):
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
    # get zt
    zt = np.dot(A_inv, B) 
    return zt.flatten()

def get_Cs(input, out_shock = False, int_shock=False):
    '''
    uses function NKM
    returns vector of length 9 with C0,C1 and C2
    '''
    C0 = NKM(input)
    C = C0
    if out_shock == True:
        C0C1 = NKM(input, out_shock=True)
        C1 = C0C1 - C0
        C = np.append(C,C1)
    if int_shock == True:
        C0C2 = NKM(input, int_shock=True)
        C2 = C0C2 - C0
        if out_shock==False: C = np.append(C,[0,0,0]) # zero vector
        C = np.append(C,C2)
    return C.flatten()

def NKM_diff(input, out_shock=False, int_shock=False):
    '''
    calculate the difference between the input and the vector implied by the model
    uses function get_Cs
    '''
    #input = guess
    implied = get_Cs(input, out_shock=out_shock, int_shock=int_shock)
    diff = input - implied[:len(input)]      
    return diff


'''
Task 2:
Run the fsolve() function on the function with the extended model and print the solution. Are the values in C0 the same as in the model from last week? Explain this.
'''

# random vector as first guess drawn from uniform U(-1,1)
guess = np.random.uniform(low = -1, high = 1, size = 6)


sol = fsolve(NKM_diff, guess, args=(True, False))

print('---output shock-----------------------')
print(f'C0 is: {[np.round(x,3) for x in sol[:3]]}') # same as last week
print(f'C1 is: {[np.round(x,3) for x in sol[3:6]]}')
print('C0 has the same values as in the model from last week (very close to zero, numbers displayed are rounded values). This is due to the fact that C0 captures the steady state part of the model which is not influenced by any shocks. C1, however, gives the reaction of a shock in output.')

'''
Task 3:
Now add a non-auto-correlated shock also to the nominal interest rate and adjust your function (which will now get an input vector of length 9) to calculate the MSV solution.
'''
# random vector as first guess drawn from uniform U(-1,1)
guess = np.random.uniform(low = -1, high = 1, size = 9)

sol2 = fsolve(NKM_diff, guess, args=(True, True))

print('---output and interest rate shock-----')
print(f'C0 is: {np.round(sol2[:3],2)}')
print(f'C1 is: {np.round(sol2[3:6],2)}')
print(f'C2 is: {np.round(sol2[6:9],2)}')