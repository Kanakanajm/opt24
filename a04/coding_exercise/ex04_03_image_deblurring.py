from ConjugateGradient import *
from GradientDescent import *
from scipy.sparse import diags
from myimgtools import *
from mymath import *
import matplotlib.pyplot as plt
import argparse
from scipy import sparse
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')


np.random.seed(0)


################################################################################
### sqL2-Tikhonov problem ######################################################
#                                                                              #
# Optimization problem:                                                        #
#                                                                              #
#     min_x  0.5*|Ax-z|_2^2 + 0.5*mu*|K*x|_{2}^2                               #
#                                                                              #
################################################################################

parser = argparse.ArgumentParser(description='Plots')
parser.add_argument('--img_size', '--img_size', default=24,
                    type=int,  dest='img_size')
parser.add_argument('--run_exp_option', '--run_exp_option', default=0,
                    type=int,  dest='run_exp_option')
args = parser.parse_args()


img_size = args.img_size        # img_size => image size


# run_exp  => experiment option 0 for consolidation of all results and
# experiment option 1 for plotting Newton vs Gradient Descent
run_exp = args.run_exp_option


### load image ###
if run_exp == 0:
    filename = "Dianasmall"
    img = mpimg.imread("data/" + filename + ".png")

    img = rgb2gray(img)

else:
    filename = "Diana240"
    img = mpimg.imread("data/" + filename + ".png")

    img = rgb2gray(img)

    img.resize((img_size, img_size))    # resizing the image

(ny, nx) = np.shape(np.asarray(img))
print("image dimensions: ", np.shape(img))
N = nx*ny
mpimg.imsave("figures/" + filename + "ground_truth.png", img, cmap=plt.cm.gray)

### construction of blurr kernel ###

# filter of size 2*k+1
k = 10
s = 2*k+1
filter = np.zeros((s, s))

filter_img = mpimg.imread("data/filter.png")
s = np.shape(filter_img)[0]
filter = filter_img/np.sum(filter_img)

mpimg.imsave("figures/" + filename + "filter.png", filter, cmap=plt.cm.gray)

# blurr operator as matrix
A = make_filter2D(ny, nx, filter)

### model blurry and noisy image ###

# reshape to fit dimension of optimization variable
b = img.reshape((N, 1))
b = A.dot(b) + np.random.normal(0.0, 0.005, (N, 1))
b = b.flatten()

# write blurry image
mpimg.imsave("figures/" + filename + "blurry.png",
             b.reshape(ny, nx), cmap=plt.cm.gray)


################################################################################
### Model ######################################################################

K = make_derivatives2D(ny, nx)

# regularization parameter
mu = 0.005

### Compute the Pre-conditioning matrix


# specify model data
model = {
    'A':     A,
    'K':     K,
    'b':     b,
    'mu':    mu,
    'img_size': img_size,
    'run_exp': run_exp,
}


################################################################################
### Run Algorithms #############################################################

# general parameter
maxiter = 100
check = 1

# initialization
x0 = b

# taping
xs = []
rs = []
ts = []
cols = []
legs = []
nams = []


if run_exp == 0:
    # Experiment 0: To compare various algorithms
    # turn algorithms to be run on or off

    run_gd_0 = 1     # Gradient Descent
    run_gd_1 = 1     # Gradient Descent - Jacobi Preconditioner
    run_gd_2 = 1     # Gradient Descent - Cholesky Preconditioner

    run_cg_0 = 1     # Conjugate Gradient Method

if run_exp == 1:
   
    # turn algorithms to be run on or off
    
    # NOTE: DO NOT CHANGE THE VALUES

    run_gd_0 = 1        # Gradient Descent without Pre-conditioner.
    run_gd_1 = 0
    run_gd_2 = 0
    run_cg_0 = 0


################################################################################

if run_gd_0:
    # Without pre-conditioning
    # TODO: Compute the preconditioner here and assign it to P (Hint: trivial)
    P  = np.eye(N)

    model['P'] = P

    print('')
    print('********************************************************************************')
    print('*** Gradient Descent without Pre-conditioning***')
    print('************************')

    options = {
        'init':           x0,
        'orig':           b,
        'storeResidual':  True,
        'storeTime':      True,
        'storePsnrs':      True
    }

    output = gd(model, options, maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    ts.append(output['seq_time'])
    cols.append((0.5, 0, 0.5, 1))
    legs.append('GD-ELS')
    nams.append('GD-ELS')

    if run_exp == 0:
        mpimg.imsave("figures/deblurred_image_gd.png",
                     output['sol'].reshape(ny, nx), cmap=plt.cm.gray)

################################################################################
Q = A.T @ A + mu * K.T @ K
if run_gd_1:
    # With Jacobi pre-conditioning
 
    #TODO: Compute the Jacobi pre-conditioner and assign it to variable P.
    # P_ii = 1/sqrt(Q_ii)
    P = np.diag(1/np.sqrt(np.diag(Q.toarray())))
    
    model['P'] = P

    print('')
    print('********************************************************************************')
    print('*** Gradient Descent with Jacobi Pre-conditioning***')
    print('************************')

    options = {
        'init':           x0,
        'orig':           b,
        'storeResidual':  True,
        'storeTime':      True,
        'storePsnrs':      True
    }

    output = gd(model, options, maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    ts.append(output['seq_time'])
    cols.append((0.1, 0.5, 0.5, 1))
    legs.append('GD-ELS-JACOBI')
    nams.append('GD-ELS-JACOBI')

    if run_exp == 0:
        mpimg.imsave("figures/deblurred_image_gd_jacobi.png",
                     output['sol'].reshape(ny, nx), cmap=plt.cm.gray)

################################################################################

if run_gd_2:

    # TODO: Compute P from Cholesky decomposition
    # You may use np.linalg.cholesky to compute cholesky decomposition, which you need to convert the involved matrices to dense format.
    # You may use sparse.csr_matrix to convert a dense matrix to sparse csr matrix.

    L = np.linalg.cholesky(sparse.csr_matrix(Q).toarray())
    P = np.linalg.inv(L).T

    model['P'] = P

    print('')
    print('********************************************************************************')
    print('*** Gradient Descent with Cholesky decomposition based Pre-conditioner***')
    print('************************')

    options = {
        'init':           x0,
        'orig':           b,
        'storeResidual':  True,
        'storeTime':      True,
        'storePsnrs':      True
    }

    output = gd(model, options, maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    ts.append(output['seq_time'])
    cols.append((0.8, 0.6, 0, 1))
    legs.append('GD-CHOLESKY')
    nams.append('GD-CHOLESKY')

    if run_exp == 0:
        mpimg.imsave("figures/deblurred_image_gd_cholesky.png",
                     output['sol'].reshape(ny, nx), cmap=plt.cm.gray)

################################################################################
if run_cg_0:
    # Without pre-conditioning
    # TODO: Compute the preconditioner here and assign it to P (Hint: trivial)

    P  = np.eye(N)
    model['P'] = P

    print('')
    print('********************************************************************************')
    print('*** Conjugate Gradient without Pre-conditioner***')
    print('**************************')

    options = {
        'init':           x0,
        'orig':           b,
        'storeResidual':  True,
        'storeTime':      True,
        'storePsnrs':      True,
    }

    output = cg(model, options,  maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    ts.append(output['seq_time'])
    cols.append((0.5, 0, 0.5, 1))
    legs.append('CG')
    nams.append('CG')
    if run_exp == 0:
        mpimg.imsave("figures/deblurred_image_cg.png",
                     output['sol'].reshape(ny, nx), cmap=plt.cm.gray)

################################################################################


if run_exp == 0:
    print('===============================================================')
    for i in range(4): # TYPO: range(3) instead of range(7)
        print(legs[i] + ':' + str(rs[i][-1]))
        print('----------------------------------')

    # Plotting Gradient Descent Methods
    fig = plt.figure()
    j = 0
    for i in range(3):
        plt.loglog(rs[i], '-', color=cols[i], linewidth=2)
    plt.legend(legs[:3])
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    plt.title('Function vs Iterations')


    # Plotting Conjugate Gradient Methods
    # plt.show(block=False)

    # fig = plt.figure()
    # j = 0
    for i in range(3, 4):
        plt.loglog(rs[i], '-', color=cols[i], linewidth=2)
    plt.legend(legs[3: 4])
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    plt.title('Function vs Iterations')

    plt.show(block=False)
    input()