import numpy as np
from numpy import zeros
import time as clock

# Gradient Descent Algorithm with Exact Line Search
def gd(model, options,  maxiter, check):
    """

    This function implements the gradient descent algorithm for strictly convex
    quadratic functions 
    
                    min_x 0.5*<x,Qx> - <b,x>                                                 
    
    with the update

        x^{k+1} = x^{k} - tau_k*\nabla f(x^{k}),

    where tau_k is chosen by solving the following optimization problem:

        min_{tau>0} f(x^{k} - tau*\nabla f(x^{k})).

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
        options['storeResidual']  = False;
    if 'storeTime'      not in options:
        options['storeTime']      = False;
    if 'storePoints'    not in options:
        options['storePoints']    = False;
    if 'storePsnrs'    not in options:
        options['storePsnrs']    = False;

    #TODO: load model parameters

    A = model['A'] #example
   

    # initialization
    x_kp1 = options['init'];
    x_bar = options['orig'];

    def fun_val_main(x):
        # TODO: Create a helper function which computes the value 
        # 0.5*|Ax-z|_2^2 + 0.5*mu*|K*x|_{2}^2 
        return 

    fun_val = # todo: compute the function value with the helper function
    psnr_val = # todo: compute the PSNR value 


    # recording 
    if options['storeResidual'] == True:
        seq_res = zeros(maxiter+1);
        seq_res[0] = fun_val;
    if options['storeTime'] == True:
        seq_time = zeros(maxiter+1);
        seq_time[0] = 0;
    if options['storePoints'] == True:
        seq_x = zeros((N,maxiter+1));        
        seq_x[:,0] = x_kp1;
    if options['storePsnrs'] == True:
        seq_psnr = zeros(maxiter+1);
        seq_psnr[0] = psnr_val;
    time = 0;


    for iter in range(1,maxiter+1):
        
        stime = clock.time();
        
        #TODO: update variables

        #TODO: compute gradient
       

        #todo: compute step size

        
        #todo: gradient descent step

        fun_val = #todo: compute the current function value (use helper function)

        psnr_val = #todo: compute the psnr value

        

        # tape residual
        time = time + (clock.time() - stime);
        if options['storePsnrs'] == True:
            seq_psnr[iter] = psnr_val;
        if options['storeResidual'] == True:
            seq_res[iter] = fun_val;
        if options['storeTime'] == True:
            seq_time[iter] = time;
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;

        # print info
        if (iter % check == 0):
            print('iter: %d, time: %5f, tau: %f, psnr: %f, fun: %f' % (iter, time, tau, psnr_val, fun_val));


    # return results
    output = {
        'sol': x_kp1,
        'iter': iter
    }

    if options['storePsnrs'] == True:
        output['seq_psnr'] = seq_psnr[0:iter+1];
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res[0:iter+1];
    if options['storeTime'] == True:
        output['seq_time'] = seq_time[0:iter+1];
    if options['storePoints'] == True:
        output['seq_x'] = seq_x[:,0:iter+1];

    return output;

