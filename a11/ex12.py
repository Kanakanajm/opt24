from scipy.sparse.linalg import norm
from scipy.optimize import linprog
import time;
import numpy as np
#import matplotlib as mpl 
#mpl.use('TkAgg')


import matplotlib.pyplot as plt
import myimgtools
from myimgtools import make_derivatives2D

### Solve c ############################################
#
# min_{u \in [-1,1]^N} <c,u> + ||K*u||_1
#
# where
# 
#    u       relaxed binary segmentation (vectorized image of dimension N)
#    c       cost matrix (vectorized image of dimension N)
#    K       discrete x- and y-derivative operator (2N x N - matrix)
#
# Note that the regularization weight can be absorbed in the cost matrix c
#
###############################################################################

### load data
filename_image = 'data/flower.png';
img = plt.imread(filename_image)
(ny,nx,nc) = np.shape(img);
print ("image dim: ", np.shape(img));

### constuct the cost matrix
fg_color = np.array([0,0,1]);
bg_color = np.array([0,0,0]);

rho = 0.1; # regularization weight
cost = np.zeros((ny,nx));
ones = np.ones((ny,nx));
for i in np.arange(0,3):
    cost = cost + (img[:,:,i] - fg_color[i]*ones)**2 
    cost = cost - (img[:,:,i] - bg_color[i]*ones)**2 
c = np.reshape(rho*cost, nx*ny); # vectorize cost image into cost vector 

K = make_derivatives2D(ny, nx);

# TODO:
# Rewrite the optimization problem as a linear program and use 'linprog' to solve it.
# This requires to define 
# obj      = ...  # holds the coefficients from the objective function.
# lhs_ineq = ...  # holds the left-side coefficients
# rhs_ineq = ...  # holds the right-side coefficients

# TODO:
# Try different solvers (simplex, interior-point)
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
...               method="") # method = simplex or method = interior-point.

# TODO:
# Extract the part of the solution vector 'opt' that corresponds to the 
# segmentation (length = nx*ny)
u_vec = ...



### evaluation (apply classifier to all pixels)
# segmentation col_img 
class_u = np.reshape(u_vec, (ny,nx)); 
img_class = np.zeros((ny,nx));
img_class[class_u>0] = 1.0;
img_class[class_u<0] = 0.0;
col_img = np.zeros((ny, nx, 3))
col_img[:, :, 0] = img_class 
col_img[:, :, 2] = 1.0-img_class


# Construct overlay image 'convex combination of image and segmentation';
overlay = img.copy();
overlay = 0.45*col_img + 0.55*img;

plt.figure();
plt.subplot(1,3,1)
img_c = np.reshape(c, (ny,nx));
plt.imshow(img_c);
plt.colorbar()

plt.subplot(1,3,2);
img_u = np.reshape(x, (ny,nx));
plt.imshow(img_u);
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow((overlay * 255).astype(np.uint8))
plt.tight_layout()
# plt.show();

plt.imsave('segmentation_l1.png', col_img)
plt.imsave('overlay_l1.png', (overlay * 255).astype(np.uint8)) 

# naive thresholding implementation
naive_img = np.zeros((ny, nx, 3))
class_naive_u = np.zeros((ny, nx))
class_naive_u[img_c > 0] = 0.0
class_naive_u[img_c < 0] = 1.0
naive_img[:, :, 0] = class_naive_u
naive_img[:, :, 2] = 1-class_naive_u

plt.imsave('naive_img_l1.png', naive_img)
