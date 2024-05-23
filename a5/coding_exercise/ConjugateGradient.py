import numpy as np
from numpy import zeros
import time as clock
import timeout_decorator
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv


# Preconditioned Conjugate Gradient Method
@timeout_decorator.timeout(20)
def cg(model, options, maxiter, check):
    """

    This function implements the Preconditioned conjugate gradient method 
    for strictly convex quadratic functions:

                    min_x 0.5*<x,Qx> - <b,x>                                                 

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

    options:    none.
    
    """

    # store options
    if 'storeResidual' not in options:
        options['storeResidual'] = False
    if 'storeTime' not in options:
        options['storeTime'] = False
    if 'storePoints' not in options:
        options['storePoints'] = False

    # todo: load  all the required model parameters

    # todo: compute D= PP^T
   

    # initialization
    x_kp1 = options['init']
    x_bar = options['orig']

    def fun_val_main(x):
        # computes the function value
        # todo: create a helper function to compute the function value
        #  0.5*|Ax-x_bar|_2^2 + 0.5*mu*|K*x|_{2}^2
        return  # todo

    # todo: compute gradient
    

    # initialization, do not change the values here.
    r_kp1 = (grad_k)
    d_kp1 = -D.dot(r_kp1)
    s_kp1 = D.dot(r_kp1)

    rr_kp1 = np.dot(r_kp1, s_kp1)

    fun_val = fun_val_main(x_kp1)  # function value

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
        r_k = r_kp1.copy()
        rr_k = rr_kp1.copy()
        x_k = x_kp1.copy()
        d_k = d_kp1.copy()
        s_k = s_kp1.copy()


        # todo: implement the preconditioned conjugate gradient method
        # Note: apply directly on x variables (denoted u in the exercise sheet) 

        # check function value
        fun_val = fun_val_main(x_kp1)

        # tape residual
        time = time + (clock.time() - stime)
        if options['storeResidual'] == True:
            seq_res[iter] = fun_val

        if options['storeTime'] == True:
            seq_time[iter] = time
        if options['storePoints'] == True:
            seq_x[:, iter] = x_kp1

        # # print info
        if (iter % check == 0):
            print('iter: %d, time: %5f,  fun: %f' % (iter, time, fun_val))

    if run_exp == 1:
        # saving files for experiment option 1
        np.savetxt('results/cg_time_'+str(img_size) +
                   '.txt', np.array([seq_time[-1]]))

    # return results
    output = {
        'sol': x_kp1,
        'iter': iter,
    }

    if options['storeResidual'] == True:
        output['seq_res'] = seq_res[0:iter+1]
    if options['storeTime'] == True:
        output['seq_time'] = seq_time[0:iter+1]
    if options['storePoints'] == True:
        output['seq_x'] = seq_x[:, 0:iter+1]

    return output
