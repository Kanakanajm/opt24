import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


np.random.seed(0)

from mymath import *
from myimgtools import *


################################################################################
### Image Deblurring problem (sqL2-Tikhonov problem)
################################################################################
#                                                                              #
# Optimization problem:                                                        #
#                                                                              #
#     min_x  0.5*|Ax-z|_2^2 + 0.5*mu*|K*x|_{2}^2                               #
#                                                                              #
################################################################################


### load image ###

filename = "Diana240";
img = mpimg.imread("data/" + filename + ".png");
img = rgb2gray(img);
(ny,nx) = np.shape(img);
print ("image dimensions: ", np.shape(img))
N = nx*ny;
mpimg.imsave(filename + "ground_truth.png", img, cmap=plt.cm.gray);

### construction of blurr kernel ###

# filter of size 2*k+1
k = 10;
s = 2*k+1;
filter = np.zeros((s,s));

filter_img = mpimg.imread("data/filter.png");
s = np.shape(filter_img)[0];
filter = filter_img/np.sum(filter_img);

mpimg.imsave(filename + "filter.png", filter, cmap=plt.cm.gray);

# blurr operator as matrix
A = make_filter2D(ny, nx, filter);  

### model blurry and noisy image ###

# reshape to fit dimension of optimization variable
b = img.reshape((N,1)); 
b = A.dot(b) + np.random.normal(0.0, 0.005, (N,1));
b = b.flatten();

# write blurry image
mpimg.imsave(filename + "blurry.png", b.reshape(ny,nx), cmap=plt.cm.gray);

################################################################################
### Model ######################################################################

K = make_derivatives2D(ny, nx); # forward-difference operator

mu = 0.001; #changeme: try the vary the regularization parameter.


# specify model data
model = {
         'A':     A,
         'K':     K,
         'b':     b,
         'N':    N,
         'mu':      mu,
        }


################################################################################
### Run Algorithms #############################################################
from GradientDescent_ELS import gd as gd_els


# general parameter
maxiter = 100;
check = 10;

# initialization
x0 = b

# taping
xs = [];
rs = [];
ts = [];
psnrs = [];
cols = [];
legs = [];
nams = [];

# turn algorithms to be run on or off
run_gd_els           = 1;    # Gradient Descent with ELS

################################################################################

if run_gd_els: 
    
    print('');
    print('********************************************************************************');
    print('*** Gradient Descent with Exact Line Search***');
    print('************************');

    options = {
        'init':           x0,
        'orig':           b,
        'storeResidual':  True,
        'storeTime':      True,
        'storePsnrs':      True
    }
    
    output = gd_els(model, options, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    psnrs.append(output['seq_psnr']);
    cols.append((0.5,0,0.5,1))
    cols.append((1,0,0,1))
    legs.append('GD-ELS');
    nams.append('GD-ELS');

    mpimg.imsave("deblurred_image_els.png", output['sol'].reshape(ny,nx), cmap=plt.cm.gray);



################################################################################



# fig = plt.figure();
# j = 0;
# for i in range(2):
#     plt.plot(rs[i], '-', color=cols[i], linewidth=2);

plt.plot(range(maxiter+1), rs[0])
plt.legend(legs);
plt.xlabel('Iterations');
plt.ylabel('Function value');
plt.title('Function vs Iterations');

# plt.show(block=False);
plt.savefig('function_val_plot.png')





fig3, ax = plt.subplots(nrows=2, ncols=2)

# for a in ax:
#     a.axis('off')

img_main = img.reshape(ny,nx) 
ax[0, 0].imshow(img_main, cmap=plt.cm.gray)
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

b_main = b.reshape(ny,nx)
ax[0, 1].imshow(b_main, cmap=plt.cm.gray)
ax[0, 1].set_title('Blurred Image')
ax[0, 1].axis('off')

r_1_main = xs[0].reshape(ny,nx) 
ax[1, 0].imshow(r_1_main, cmap=plt.cm.gray)
ax[1, 0].set_title('Deblurred Image - ELS')
ax[1, 0].axis('off')

ax[1, 1].axis('off')

# r_2_main = xs[1].reshape(ny,nx) 
# ax[3].imshow(r_2_main, cmap=plt.cm.gray)
# ax[3].set_title('Deblurred Image - CG')
plt.tight_layout()
# plt.show();
plt.savefig('img_plot.png')
