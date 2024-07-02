import numpy as np
from numpy import zeros
import time as clock

# Projected Gradient Method
def pgm(model, options,  maxiter, check):
    """

    We implement Projected Gradient Method given in Algorithm 22 with L=Lip.

    Lip is the Lipschitz constant of the gradient, which is provided.

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
    A = model['A']
    K = model['K']
    b = model['b']
    mu = model['mu']
    eps = model['eps']
    N = b.shape[0]


    # initialization
    x_kp1 = options['init'];
    x_bar = options['orig'];
    Lip = options['Lip'] # Lipschitz constant of the gradient

    alpha = 0.5/Lip # half way between 0 and 2/Lip

    def project_onto_box(x, p = 0, q = 1):
        return np.minimum(np.maximum(x, p), q)

    def fun_val_main(x):
        Kx = K @ x
        # split Kx into two parts
        Kx1 = Kx[:N]
        Kx2 = Kx[N:]

        return 0.5 * np.linalg.norm(A @ x - b) ** 2 + mu * np.sum(np.sqrt(Kx1**2 + Kx2**2 + eps**2))
    
    def grad_val(x):
        Kx = K @ x
        # split Kx into two parts
        Kx1 = Kx[:N]
        Kx2 = Kx[N:]
        denominator = np.sqrt(Kx1**2 + Kx2**2 + eps**2)

        grad_partial = (Kx1 / denominator) * K[:N] + (Kx2 / denominator) * K[N:]

        return A.T @ (A @ x - b) + mu * grad_partial


    fun_val = fun_val_main(x_kp1)
    psnr_val = 0

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
        
        # TODO: update variables
        x_k = x_kp1.copy()
        

        # TODO: compute gradient
        grad_k = grad_val(x_k)
        
        # TODO: compute step size
        tau = 1
        # TODO: projected gradient update step 

        x_kp1 = project_onto_box(x_k - alpha * grad_k)

        # compute function value
        fun_val = fun_val_main(x_kp1)

        psnr_val = 10 * np.log10( 255*255 / np.mean(np.square(x_kp1-x_bar)) )

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

