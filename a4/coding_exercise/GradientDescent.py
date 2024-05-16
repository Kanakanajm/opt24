import numpy as np
from numpy import zeros
import time as clock
from scipy.sparse.linalg import eigs
import timeout_decorator

# Preconditioned Gradient Descent Algorithm
@timeout_decorator.timeout(20)
def gd(model, options,  maxiter, check):
    """

    This function implements the preconditioned gradient descent 
    algorithm for strictly convex quadratic functions 
    
                    min_x 0.5*<x,Qx> - <b,x>                                                 
    
    with the update

        x^{k+1} = x^{k} - tau_k*PP^T\nabla f(x^{k}),

    where tau_k = 2/(L+m), where P is the preconditioner,  L is the maximum 
    eigenvalue of PQP^T and m is the small eigenvalue of PQP^T. For full details 
    regarding the algorithm, see Algorithm 8 in the lecture notes.

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

    # todo: load all necessary parameters.

    # todo:  compute the step-size and also make sure it convert it to real value 
    # with np.real command if required. You may use eigs command for computing 
    # the eigen values.

    # todo: calculate D= PP^T

    # initialization
    x_kp1 = options['init']
    x_bar = options['orig']

    def fun_val_main(x):
        # todo: create a helper function to compute the function value
        #  0.5*|Ax-x_bar|_2^2 + 0.5*mu*|K*x|_{2}^2
        return #todo

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

        # todo:  gradient descent step (hint: use D)

        # todo:  check function value
        
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
            print('iter: %d, time: %5f, tau: %f,  fun: %f' %
                  (iter, time, tau,  fun_val))

    if run_exp == 1:
        # saving files for experiment option 1
        np.savetxt('results/gd_time_'+str(img_size) +
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
