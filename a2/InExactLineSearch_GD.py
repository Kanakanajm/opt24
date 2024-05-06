import numpy as np
from numpy import zeros
from scipy.optimize import minimize
import time as clock


# Gradient Descent Algorithm with Inexact Line Search Method
def gd(model, options, tol, maxiter, check):
    """

    This function implements the gradient descent algorithm:

        x^{k+1} = x^{k} - tau*\nabla f(x^{k}).

    To determine the step size, wer perform inexact line search

    \min_{\tau >0} f(x\iter\k - \tau \nabla f(x\iter\k))  
    

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
    if 'storeResidual'  not in options:
        options['storeResidual']  = False
    if 'storeTime'      not in options:
        options['storeTime']      = False
    if 'storePoints'    not in options:
        options['storePoints']    = False
    if 'storeObjective' not in options:
        options['storeObjective'] = False
    

    # load model parameters
    a = model['a']
    b = model['b']
    N = model['N']

    # load parameter
    tau    = options['stepsize']

    # initialization
    #   x_kp1:  initial point (use 'kp1' here, which is set to x_k
    #                         in the beginning of each iteration)
    #           - given in options['init']
    #   f_kp1:  initial objective value f(x_kp1)
    #           - needs to be computed here
    #   res0:   initial residual 
    #           - distance to the global minimizer (a,a^2).
    #   grad_k: dummy initialization of the gradient variable 
    #           - initialize as zero vector
    ### TODO: Initialize the variables as listed above. ###
    def f(x):
        return (a - x[0])**2 + b*(x[1] - x[0]**2)**2
    
    def fres(x):
        return np.linalg.norm(x - np.array([a,a**2]))
    
    x_kp1 = options['init']
    f_kp1 = f(x_kp1)
    res0 = fres(x_kp1)
    grad_k = np.zeros(N)
    # recording 
    if options['storeResidual'] == True:
        seq_res = zeros(maxiter+1)
        seq_res[0] = res0
    if options['storeTime'] == True:
        seq_time = zeros(maxiter+1)
        seq_time[0] = 0
    if options['storePoints'] == True:
        seq_x = zeros((N,maxiter+1))        
        seq_x[:,0] = x_kp1
    if options['storeObjective'] == True:
        seq_obj = zeros(maxiter+1)        
        seq_obj[0] = f_kp1
    time = 0

    # solve 
    breakvalue = 1 # this variable determines the breaking mode of the algorithm
                    # 1: Termination by exceeding the maximal number of iterations.
                    # 2: Tolerance threshold is reached.
    for iter in range(1,maxiter+1):
        
        stime = clock.time()
        
        # update variables
        x_k = x_kp1.copy()
        # f_k = f_kp1.copy()

        # compute gradient
        ### TODO: Compute the gradient 'grad_k'. 
        ### - Estimate the expession of the gradient analytically, and compute the
        ### gradient here using this result.
        ###
        grad_k = np.array([-2*(a - x_k[0]) - 4*b*x_k[0] * (x_k[1] - x_k[0] ** 2), 2*b*(x_k[1] - x_k[0]**2)])

        # line search
        ### TODO: Write the code for Inexact Line Search here.
        ### You may use "minimize" function, which is imported in the 
        ### beginning of the file, for which scipy must be installed.
        ### In the minimize function you may use appropriate initialization for tau.
        ### For method parameter, use L-BFGS-B.
        fun = lambda t: (a - (x_k[0] - grad_k[0]*t))**2 + b*(x_k[1] - grad_k[1]*t - (x_k[0] - grad_k[0]*t)**2)**2
        tau_k = minimize(fun, tau, method='L-BFGS-B').x[0]
        x_kp1 = x_k - tau_k * grad_k

        # check breaking condition
        ### TODO: compute the current residual (distance to global minimizer)

        res = fres(x_kp1)
        if res < tol:
            breakvalue = 2

        # tape residual
        time = time + (clock.time() - stime)
        if options['storeResidual'] == True:
            seq_res[iter] = res
        if options['storeTime'] == True:
            seq_time[iter] = time
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1
        if options['storeObjective'] == True:
            seq_obj[iter] = f_kp1

        # print info
        if (iter % check == 0):
            print('iter: %d, time: %5f, tau: %f, res: %f' % (iter, time, tau, res))
    
        # handle breaking condition
        if breakvalue == 2:
            print('Tolerance value reached!!!')
            break

    # return results
    output = {
        'sol': x_kp1,
        'breakvalue': breakvalue,
        'iter': iter
    }

    if options['storeResidual'] == True:
        output['seq_res'] = seq_res[0:iter+1]
    if options['storeTime'] == True:
        output['seq_time'] = seq_time[0:iter+1]
    if options['storePoints'] == True:
        output['seq_x'] = seq_x[:,0:iter+1]
    if options['storeObjective'] == True:
        output['seq_obj'] = seq_obj[0:iter+1]

    return output

