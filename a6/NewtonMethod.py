import numpy as np
from numpy import zeros
import time as clock
from my_functions import *

# Newton Method
def newton(model, options,  maxiter, check):
    """
    This function implements Newton's method  for solving the  
    optimization problem in exercise 4 of the given exercise sheet. 

    The optimization problem involved is 
    
                    min_w f(w)                                             
    
    with the update

        w^{k+1} = w^{k} - tau_k*(\nabla^2f(x^{k}))^{-1}\nabla f(x^{k}),

    where tau_k is typically set to 1. For full details 
    regarding the algorithm, see Algorithm 12 in the lecture notes.


    Parameter:
    ----------
    model:      dictionary
                (Defined in the main-file!)
    options:    dictionary
                (see below)
    tol:        postive real number
                If the residual drops below 'tol', the algorithm has found a 
                solution.
    maxiter:    positive integer
                Maximal number of iterations of the algorithm.
    check:      positve integer
                After 'check' many iterations the current objective value and 
                residual is printed to the screen.

    """

    # store options
    if 'storeResidual' not in options:
        options['storeResidual'] = False
    if 'storeTime' not in options:
        options['storeTime'] = False
    if 'storePoints' not in options:
        options['storePoints'] = False


    # load data 

    X = model['X']
    y = model['y']

    # initialization
    x_kp1 = options['init']

    fun_val = f_func(X, y, x_kp1)

    # recording
    if options['storeFuncs'] == True:
        seq_funcs = zeros(maxiter+1)
        seq_funcs[0] = fun_val
    if options['storeResidual'] == True:
        seq_res = zeros(maxiter+1)
        seq_res[0] = np.linalg.norm(grad_func(X,  y, x_kp1))**2
    if options['storeTime'] == True:
        seq_time = zeros(maxiter+1)
        seq_time[0] = 0
    if options['storePoints'] == True:
        seq_x = zeros((N, maxiter+1))
        seq_x[:, 0] = x_kp1

    time = 0

    for iter in range(1, maxiter+1):

        stime = clock.time()

        # todo: update variables

        # todo: compute gradient using grad_func function in my_functions.py file

        # todo: compute step size

        # todo: compute the Hessian matrix

        # todo: implement the Newton's method update

        # todo: compute the function value 
        fun_val = 

        # todo: check breaking condition by computing residual which is
        # squared gradient norm. Make sure to store the residual
        #  into the variable 'res'


        ################
        if res < 1e-12:
            break
        else:
            breakvalue = 0

        # tape residual
        time = time + (clock.time() - stime)
        if options['storeFuncs'] == True:
            seq_funcs[iter] = fun_val
        if options['storeResidual'] == True:
            seq_res[iter] = res
        if options['storeTime'] == True:
            seq_time[iter] = time
        if options['storePoints'] == True:
            seq_x[:, iter] = x_kp1

        # print info
        if (iter % check == 0):
            print('iter: %d, time: %5f, tau: %f, fun: %f' %
                  (iter, time, tau,  fun_val))


    # return results
    output = {
        'sol': x_kp1,
        'iter': iter
    }
    if options['storeFuncs'] == True:
        output['seq_funcs'] = seq_funcs[0:iter+1]
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res[0:iter+1]
    if options['storeTime'] == True:
        output['seq_time'] = seq_time[0:iter+1]
    if options['storePoints'] == True:
        output['seq_x'] = seq_x[:, 0:iter+1]

    return output
