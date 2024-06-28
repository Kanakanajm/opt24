import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


np.random.seed(0)

from mymath import *
from myimgtools import *


################################################################################
### Image Deblurring problem                                                   #
################################################################################


### load image ###

filename = "Diana240";
img = mpimg.imread(filename + ".png");
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

filter_img = mpimg.imread("filter.png");
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

K = make_derivatives2D(ny, nx)  # forward difference operator

mu = 0.5; #TODO: Tune this parameter til the reconstruction is reasonable.
eps = 1.0;


# specify model data
model = {
         'A':       A,
         'K':       K,
         'b':       b,
         'mu':      mu,
         'eps':     eps,
        }


################################################################################
### Run Algorithms #############################################################
from ProjectedGradient import pgm
from ConditionalGradient import cgm


# general parameter
maxiter = 100;
check = 10;

# initialization
x0 = np.maximum(0.0, np.minimum(1.0, b))


# taping
xs = [];
rs = [];
ts = [];
psnrs = [];
cols = [];
legs = [];
nams = [];

# turn algorithms to be run on or off
run_pgm           = 1;    # Projected Gradient Method
run_cgm           = 1     # Conjugate Gradient Method

################################################################################

if run_pgm: 
    
    print('');
    print('********************************************************************************');
    print('*** Projected Gradient Method***');
    print('************************');

    from scipy.sparse.linalg import eigs
    vals, vecs = eigs(K.T.dot(K))
    Lip = np.real(np.max(vals)) + mu*8/(eps)
    

    options = {
        'init':           x0,
        'orig':           b,
        'Lip':            Lip,
        'storeResidual':  True,
        'storeTime':      True,
        'storePsnrs':      True
    }
    
    output = pgm(model, options, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    psnrs.append(output['seq_psnr']);
    cols.append((0.5,0,0.5,1));
    legs.append('PGM');
    nams.append('PGM');

    mpimg.imsave("deblurred_image_pgm.png", output['sol'].reshape(ny,nx), cmap=plt.cm.gray);



################################################################################
if run_cgm: 
    
    print('');
    print('********************************************************************************');
    print('*** Conditional Gradient Method ***');
    print('**************************');

    options = {
        'init':           x0,
        'orig':           b,
        'backtrackingMaxiter':  30,  # TODO: tune this parameter
        'backtrackingFactor':   0.9,  # TODO: tune this parameter
        'Armijoparam':          1e-2,  # TODO: tune this parameter
        'storeResidual':  True,
        'storeTime':      True,
        'storePsnrs':      True,
    }
    
    output = cgm(model, options,  maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    psnrs.append(output['seq_psnr']);
    cols.append((0.8,0.6,0,1));
    legs.append('CGM');
    nams.append('CGM');

    mpimg.imsave("deblurred_image_cgm.png", output['sol'].reshape(ny,nx), cmap=plt.cm.gray);


fig = plt.figure();
j = 0;
for i in range(2):
    plt.plot(rs[i], '-', color=cols[i], linewidth=2);
plt.legend(legs);
plt.xlabel('Iterations');
plt.ylabel('Function value');
plt.title('Function vs Iterations');

plt.show(block=False);


fig3, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))

for a in ax:
    a.axis('off')

img_main = img.reshape(ny,nx) 
ax[0].imshow(img_main, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

b_main = b.reshape(ny,nx)
ax[1].imshow(b_main, cmap=plt.cm.gray)
ax[1].set_title('Blurred Image')

r_1_main = xs[0].reshape(ny,nx) 
ax[2].imshow(r_1_main, cmap=plt.cm.gray)
ax[2].set_title('Deblurred Image - PGM')

r_2_main = xs[1].reshape(ny,nx) 
ax[3].imshow(r_2_main, cmap=plt.cm.gray)
ax[3].set_title('Deblurred Image - CGM')

plt.show();
