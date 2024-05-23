import numpy as np
from numpy import zeros
import time as clock

from scipy.sparse.linalg import spsolve
import timeout_decorator

# Newton Method


@timeout_decorator.timeout(20)
def newton(model, options,  maxiter, check):
    """

    This function implements the Newton methdo for strictly convex
    quadratic functions 
    
                    min_x 0.5*<x,Qx> - <b,x>                                                 
    
    with the update

        x^{k+1} = x^{k} - tau_k \nabla^2f(x^{k})^{-1}*\nabla f(x^{k}).


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
    if 'storePsnrs' not in options:
        options['storePsnrs'] = False

    # load model parameters

    A = model['A']
    K = model['K']
    b = model['b']
    mu = model['mu']

    img_size = model['img_size']
    run_exp = model['run_exp']

    # initialization
    x_kp1 = options['init']
    x_bar = options['orig']

    def fun_val_main(x):
        # todo: create a helper function to compute the function value
        #  0.5*|Ax-x_bar|_2^2 + 0.5*mu*|K*x|_{2}^2
        return  # todo

    fun_val = fun_val_main(x_kp1)

    # recording
    if options['storeResidual'] == True:
        seq_res = zeros(maxiter+1)
        seq_res[0] = fun_val
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

        # todo: compute gradient
        
        # todo: compute step size

        # todo: implement newton method 
        # IMP: check for efficient implementation rather than taking inverse.

        # todo: compute function value and assign to fun_val variable

        # track time
        time = time + (clock.time() - stime)

        if options['storeResidual'] == True:
            seq_res[iter] = fun_val
        if options['storeTime'] == True:
            seq_time[iter] = time
        if options['storePoints'] == True:
            seq_x[:, iter] = x_kp1

        # print info
        if (iter % check == 0):
            print('iter: %d, time: %5f, tau: %f, fun: %f' %
                  (iter, time, tau,  fun_val))
    if run_exp == 1:
        # saving files for experiment option 1
        np.savetxt('results/newton_time_'+str(img_size) +
                   '.txt', np.array([seq_time[-1]]))

    # return results
    output = {
        'sol': x_kp1,
        'iter': iter
    }

    if options['storeResidual'] == True:
        output['seq_res'] = seq_res[0:iter+1]
    if options['storeTime'] == True:
        output['seq_time'] = seq_time[0:iter+1]
    if options['storePoints'] == True:
        output['seq_x'] = seq_x[:, 0:iter+1]

    return output
