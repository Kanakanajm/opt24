import numpy as np
from numpy import zeros
import time as clock
from my_functions import *

# BFGS Method
def bfgs(model, options,  maxiter, check):
    """
    This function implements L-BFGS  algorithm for solving the  
    optimization problem in exercise 4 of the given exercise sheet. 

    The optimization problem involved is 
    
                    min_w f(w)                                             
    
    with the BFGS update given by 

        w^{k+1} = w^{k} + tau_k*d_k

    where d_k = -H_k\nabla f(w^k), tau_k is chosen from Wolfe conditions and 
    H_k is an approximation of the inverse Hessian at w^k. 

    H_{k+1} relies on the information on w^{k+1}-w^k, 
    \nabla f(w^{k+1})-\nabla f(w^k), H_k, 
    as in updated using BFGS update formula (Eq 67 in the lecture notes).

    H_0 is typically an identity matrix.
    
    For full details regarding the algorithm, see Algorithm 14 in the lecture notes.


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
    if 'storeFuncs' not in options:
        options['storeFuncs'] = False
    if 'storeTime' not in options:
        options['storeTime'] = False
    if 'storePoints' not in options:
        options['storePoints'] = False


    # load data 
    X = model['X']
    y = model['y']

    #  initialization (this is the w according to the problem description)
    x_kp1 = options['init']

    # get the dimension of the problem
    n = x_kp1.shape[0]

    f_kp1  = f_func(X, y, x_kp1)
    grad_kp1 = grad_func(X,  y, x_kp1)


    # todo: hessian inverse approximation initialization 
    H_kp1 = 



    # recording
    if options['storeFuncs'] == True:
        seq_funcs = zeros(maxiter+1)
        seq_funcs[0] = f_kp1
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

    breakvalue = 1  # this variable determines the breaking mode of the algorithm
    # 1: Termination by exceeding the maximal number of iterations.
    # 2: Tolerance threshold is reached.
    # 3: Maximal number of backtracking iterations exceeded.

    # todo: set the backtracking parameters for the backtracking step in BFGS step

    for iter in range(1, maxiter+1):

        stime = clock.time()

        # todo: Implement the bfgs method here. Please feel free to modify 
        # the below given template.

        # todo: update variables


        # todo: compute step size

        
        # todo: obtain the descent direction

        # todo: loop for backtracking 
        # todo: choose appropriate backtracking parameters such
        # that residual is <1e-12.  See below regarding residual.

        for iterbt in range(0, backtrackingMaxiter):
            # todo: Complete the backtracking to check Armijo  and Wolfe conditions.

            # todo: gradient descent like step using the descent direction

            # todo: compute new value  of objective


            # todo: check Armijo  and Wolfe condition
            # break the loop if the conditions are satisfied.
            # use breakvalue 3 if you maximum iterations are reached.


        # todo: Implement the BFGS method updates.
        

        # todo: Compute the function value using f_func function in my_functions.py
        fun_val = 

        # todo: check breaking condition by computing residual which is 
        # squared gradient norm. Make sure to store the residual
        #  into the variable 'res'
        


        ################
        # Residual check

        if res < 1e-12:
            breakvalue = 2
        else:
            pass

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

        # handle breaking condition
        if breakvalue == 2:
            print('Tolerance value reached!!!');
            break;
        elif breakvalue == 3:
            print('Not enough backtracking iterations!!!');
            break;

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

