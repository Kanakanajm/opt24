import numpy as np
from numpy import zeros
import time as clock
from my_functions import *

# Gradient Descent
def gd(model, options,  maxiter, check):
    """
    This function implements Gradient descent  algorithm for solving the  
    optimization problem in exercise 4 of the given exercise sheet. 

    The optimization problem involved is 
    
                    min_w f(w)                                             
    
    with the update

        w^{k+1} = w^{k} - tau_k*\nabla f(x^{k}),

    where tau_k <= (2/L), where  L is the maximum 
    eigenvalue of $\nabla^2 f(w)$ uniformly over all w. For full details 
    regarding the algorithm, see Algorithm 4 in the lecture notes.

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
    if 'storeFuncs' not in options:
        options['storeFuncs'] = False
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

    # Upper bound on Hessian matrix, Note: this is not the Hessian
    m = y.shape[0]
    tent_hess = 0
    for i in range(m):
        tent_hess += (1/m)*np.outer(X[i, :], X[i, :])

    # maximum eigen value
    w, v = np.linalg.eig(tent_hess)
    L = np.max(w) # Lipschitz constant of the gradient

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

        # update variables
        x_k = x_kp1.copy()

        # compute gradient
        grad_k = grad_func(X,  y, x_k)

        # compute step size
        tau =2*0.99/L
        
        # gradient descent
        x_kp1 = x_k - tau*(grad_k)

        # check breaking condition
        fun_val = f_func(X, y, x_kp1)

        ### residual computation ###
        res = np.linalg.norm(grad_k)**2
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
