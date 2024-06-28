import numpy as np
from numpy import zeros
import time as clock

# Conditional Gradient Method
def cgm(model, options, maxiter, check):
    """
    We implement Conditional Gradient Method given in Algorithm 20.

    Parameter:
    ----------
    model:      dictionary
                (Defined in the main-file!)
    options:    dictionary
                (see below)
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

    # todo: load model parameters
    # load here A, K, b, mu, eps

 

    # load options parameters
    backtrackingMaxiter = options['backtrackingMaxiter'];
    rho   = options['backtrackingFactor'];
    gamma = options['Armijoparam'];

    # initialization
    x_kp1 = options['init'];
    x_bar = options['orig'];

    def fun_val_main(x):
        # TODO: Compute and return the function value
        # you may reuse this function later.
        return 


    psnr_val = 0 
    f_kp1 = fun_val = fun_val_main(x_kp1)  # function value

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

        # TODO: Implement the conditional gradient updates
        # i.e: Compute tilde x^k and obtain d^k
       
        tau = 0.05   # starting step-size in backtracking 

        # # backtracking to find step-size
        for iterbt in range(0, backtrackingMaxiter):

            # TODO: gradient descent like step  and store it in x_kp1

            # TODO:  compute new value of the objective and store in f_kp1

            # no backtracking
            if backtrackingMaxiter == 1:
                break

            # TODO: check Armijo condition
            # return breakvalue=3 if backtrackingMaxiter is reached
            # otherwise, break


        # check breaking condition
        fun_val = fun_val_main(x_kp1)
        psnr_val = 10 * np.log10( 255*255 / np.mean(np.square(x_kp1-x_bar)) )

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
            print('iter: %d, time: %5f, tau: %f,  psnr: %f, fun: %f' %
                  (iter, time, tau, psnr_val, fun_val))

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

