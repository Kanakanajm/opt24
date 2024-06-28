import numpy as np
from numpy import zeros
import time as clock

# Conjugate Gradient Method
def cg(model, options, maxiter, check):
    """

    This function implements the conjugate gradient method for strictly convex
    quadratic functions:

                    min_x 0.5*<x,Qx> - <b,x> + c                                                 

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
    if 'storeResidual'  not in options:
        options['storeResidual']  = False;
    if 'storeTime'      not in options:
        options['storeTime']      = False;
    if 'storePoints'    not in options:
        options['storePoints']    = False;
    if 'storePsnrs'    not in options:
        options['storePsnrs']    = False;

    # load all the model parameters 

    A = model['A'] # example
    K = model['K']
    b = model['b']
    mu = model['mu']
    img_size = model['img_size']
    run_exp = model['run_exp']
    P = model['P']

    Q = A.T @ A + mu * K.T @ K
    bprime = -A.T @ b
    N = img_size

    grad = lambda x: Q @ x + bprime
    # initialization
    x_kp1 = options['init'];
    x_bar = options['orig']; #x_bar is the noisy blurry image

    def fun_val_main(x):
        # Create a helper function which computes the value 
        # 0.5*|Ax-z|_2^2 + 0.5*mu*|K*x|_{2}^2 
        return 0.5 * np.linalg.norm(A @ x - x_bar)**2 + 0.5 * mu * np.linalg.norm(K @ x)**2 

    
    grad_k = grad(x_kp1) # compute gradient

    # some example computations (feel free to modify)
    r_kp1 = grad_k;
    d_kp1 = -r_kp1;

    psnr = lambda x: np.linalg.norm(x - x_bar)**2 / img_size
    psnr_val = psnr(x_kp1) # compute psnr value x_kp1 with respect to x_bar
    fun_val =  fun_val_main(x_kp1) # compute the function value (use the helper function)

    # recording 
    if options['storeResidual'] == True:
        seq_res = zeros(maxiter+1);
        seq_res[0] = fun_val;
    if options['storePsnrs'] == True:
        seq_psnr = zeros(maxiter+1);
        seq_psnr[0] = psnr_val;
    if options['storeTime'] == True:
        seq_time = zeros(maxiter+1);
        seq_time[0] = 0;
    if options['storePoints'] == True:
        seq_x = zeros((N,maxiter+1));        
        seq_x[:,0] = x_kp1;
    time = 0;


    for iter in range(1,maxiter+1):
        
        stime = clock.time();
        
        # TODO: update variables
        x_k = x_kp1.copy()
        d_k = d_kp1.copy()
        r_k = r_kp1.copy()



        #TODO: update steps of conjugate gradient method
        tau_k = np.dot(r_k, r_k) / np.dot(d_k, Q @ d_k)
        x_kp1 = x_k + tau_k * d_k

        r_kp1 = r_k + tau_k * Q @ d_k
        beta_kp1 = np.dot(r_kp1, r_kp1) / np.dot(r_k, r_k)
        d_kp1 = -r_kp1 + beta_kp1 * d_k
        


        fun_val = psnr(x_kp1) # compute function value (with helper function)
        psnr_val = fun_val =  fun_val_main(x_kp1) # compute function value (with helper function)

        # tape residual
        time = time + (clock.time() - stime);
        if options['storeResidual'] == True:
            seq_res[iter] = fun_val;
        if options['storePsnrs'] == True:
            seq_psnr[iter] = psnr_val;
        if options['storeTime'] == True:
            seq_time[iter] = time;
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;

        # print info
        if (iter % check == 0):
            print('iter: %d, time: %5f,  psnr: %f, fun: %f' % (iter, time, psnr_val, fun_val));


    # return results
    output = {
        'sol': x_kp1,
        'iter': iter,
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

