#########################
# Assignment 1
######################## 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

'''
Simulate the model for 1000 periods under naive (or static) expectations:
EtYt+1 = Yt−1, (5)
Etπt+1 = πt−1. (6)
To do this, start with initial values for Yt, πt and it for period 0. Then
make a for loop that, for each period, calculates the outcomes of Yt, πt and
it, given the model parameters and values of expectations. The values of
expectations you can take from the realizations of inflation and output in
the previous period, using the above 2 equations
'''

sigma= 2
kappa=0.3
beta=0.99
phi_1=1.5
phi_2=0.2

Y = np.array([100])
pi = np.array([.02])
int = np.array([.04])

for i in range(1,1000+1):
    # expectations
    EY = Y[i-1]
    Epi = pi[i-1]
    # calculate interest rate
    int_new = phi_1 * Epi + phi_2 * EY
    int = np.append(int, int_new)
    # calculate ouput
    y_new = EY - 1/sigma * (int_new - Epi)
    Y = np.append(Y,y_new)
    # calculate inflation
    pi_new = beta * Epi + kappa * y_new
    pi = np.append(pi, pi_new)

convergence = [Y[1000],pi[1000],int[1000]]
print(convergence)

'''
Plot the results using matplotlib.pyplot package. Does the economy converge to some value?
'''
fig,ax = plt.subplots()
ax.plot(int, label = 'i')
ax.plot(Y, label = 'Y')
ax.plot(pi,label = 'pi')
plt.legend()

'''
 Write the model in the form
Azt = B
with zt = [Yt, πt, it]
• A is a matrix relating these three variables according to equation (1),
(2) and (3).
• B is a vector containing the other arguments showing up in the above
three equations (in this week’s model specification, only the expectation terms).

Define the matrix A in your script as a numpy array, and calculate its
inverse.
'''

A = np.array([[1,0,1/sigma],[-kappa,1, 0],[0,0,1]])
A_inv = np.linalg.inv(A)

'''
Write a function that has as input a vector of expectation values
(EtYt+1, Etπt+1, Etit+1) and as output the vector zt defined above, containing the period t values of the three variables that are in our system of
equations. (note that we do not care about interest rate expectations but
that we need to take this third argument of the function in order to use
fsolve later).
'''
# zt = A_inv * B

def func(exp,A = A):
    '''
    input: list of 3 ; Y, pi and interest in expectations
    output: zt = (Yt, πt, it)
    '''
    # unpack input vector
    ey, epi, ei = exp
    # get inverse of A
    A_inv = np.linalg.inv(A)
    # calculate B
    b1 = ey + 1/sigma *(epi)
    b2 = beta *epi
    b3 = phi_1 * epi + phi_2 * ey
    # solve for zt
    B = np.array([[b1], [b2], [b3]])
    zt = np.dot(A_inv, B)
    return zt

'''
What happens when you take as an input the values that your simulation
converged to?
'''

print(func(convergence))

'''
Modify your function such that it returns the difference between the input
values of your function (Etzt+1) and the resulting values of zt.
'''

def func_new(exp, A = A):
    '''
    input: list of 3 ; Y, pi and interest in expectations
    output: Ez - zt 
    '''
    # unpack input vector
    ey, epi, ei = exp
    # get inverse of A
    A_inv = np.linalg.inv(A)
    # calculate B
    b1 = ey + 1/sigma *(epi)
    b2 = beta *epi
    b3 = phi_1 * epi + phi_2 * ey
    B = np.array([[b1], [b2], [b3]])
    # solve for zt
    zt = np.dot(A_inv, B)
    Ez = np.array(exp).reshape(3,1)
    # get difference from input
    res = Ez-zt
    return res.reshape(3).tolist()

'''
We now want to find those values of expectations for which the realizations
in the current period are the same as the expectations about the next
period. When agents expect these values in every period, then they will
also be realized in every period. In this case the model will be in a steady
state, where agents make perfect predictions. This is the minimum state
variable solution of our simple model. In order to find this solution we
can apply the fsolve() function to the modified function that we defined
above.
Run the fsolve() function on the function that we have made. The fsolve()
function will numerically find values for our input, that result in output
values of zero. In our case, that means that it finds values of Ez for which
Ez-z=0 and hence Ez=z, which is what we were looking for.
'''   

solution = fsolve(func_new, convergence)

'''
What solution do you find? Compare it with your simulations
'''
print(solution)
