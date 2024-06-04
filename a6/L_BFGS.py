import numpy as np
from numpy import zeros
import time as clock
from my_functions import *

# L-BFGS Method
def l_bfgs(model, options,  maxiter, check):
    """
    This function implements L-BFGS  algorithm for solving the  
    optimization problem in exercise 4 of the given exercise sheet. 

    The optimization problem involved is 
    
                    min_w f(w)                                           
    
    and the L-BFGS update is

        w^{k+1} = w^{k} + tau_k*d_k

    where d_k = -H_k\nabla f(w^k), tau_k is chosen from Wolfe conditions and 
    H_k is an approximation of the inverse Hessian at w^k. 

    In L-BFGS, we directly compute d_k  using the Algorithm 15 
    in the lecture notes. Such a procedure can be helpful for large dimensions.

    H_0 is typically an identity matrix.
    
    For full details regarding the algorithm, see Algorithm 16 in the lecture notes.


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

    m = x_kp1.shape[0]

    f_kp1 = f_func(X, y, x_kp1)
    grad_kp1 = grad_func(X,  y, x_kp1)

    # hessian inverse initialization
    H_kp1 = np.eye(m)

    m_lbfgs = 4 #storing last 4 entities in L-BFGS

    temp_s = np.zeros((m_lbfgs, m))  # to store s values
    temp_y = np.zeros((m_lbfgs, m))  # to store y values
    temp_rho = np.zeros(m_lbfgs)     # to store rho values

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
        seq_x = zeros((m, maxiter+1))
        seq_x[:, 0] = x_kp1

    time = 0

    breakvalue = 1  # this variable determines the breaking mode of the algorithm
    # 1: Termination by exceeding the maximal number of iterations.
    # 2: Tolerance threshold is reached.
    # 3: Maximal number of backtracking iterations exceeded.

    # todo: set the backtracking options for the backtracking step in L-BFGS step
    tau_0 = 18 # init time step
    decay = 0.5 # time step decay factor
    backtrackingMaxiter =5 # maxiter
    gamma = 1e-4 # armijo param
    eta = 0.9 # wolfe param

    for iter in range(1, maxiter+1):

        stime = clock.time()

        # todo: update variables
        x_k = x_kp1.copy()
        f_k = f_kp1.copy()
        grad_k = grad_kp1.copy()
        H_k = H_kp1.copy()

        # todo: compute step size
        tau = tau_0

        # todo: implement the L-BFGS method
        
        # todo: loop for backtracking within L-BFGS method
        # todo: choose appropriate backtracking parameters such 
        # that residual is <1e-12.  See below regarding residual.

        for iterbt in range(0, backtrackingMaxiter):
            pass
            # todo: Complete the backtracking to check Armijo  and Wolfe conditions.

            # todo: gradient descent like step using the descent direction

            # todo: compute new value  of objective

            # todo: check Armijo  and Wolfe condition
            # break the loop if the conditions are satisfied.
            # use breakvalue 3 if you maximum iterations are reached.


        # todo: Compute the function value using f_func function in my_functions.py
        fun_val = f_func(X, y, x_kp1)

        # todo: check breaking condition by computing residual which is
        # squared gradient norm. Make sure to store the residual
        #  into the variable 'res'
        res = np.linalg.norm(grad_func(X,  y, x_kp1))**2
        ################
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
            print('Tolerance value reached!!!')
            break
        elif breakvalue == 3:
            print('Not enough backtracking iterations!!!')
            break

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
