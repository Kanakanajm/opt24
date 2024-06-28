import numpy as np
from numpy import random;
import matplotlib.pyplot as plt

################################################################################
#                                                                              #
# Optimization problem                                                         #
#                                                                              #
#       min_{u} f(u);                                                          #
#               f(u) = 1/2*sum_{i=1}^N (u_i - c_i)^2                           #
#                           + mu/2*\sum_{i=1}^{N-1} ((u_{i+1} - u_i)/h)^2      #
#                                                                              #
# where                                                                        #
#       c \in \R^N                                                             #
#       N       number of samples in [0,1]                                     #
#       h       sampling rate (difference between two sample points in [0,1])  #
#       mu > 0                                                                 #
#                                                                              #
################################################################################

### Define model parameter
# 
# - A signal c is contructed in [0,1]. 
#        (A signal is a vector in \R^N, where each coordinate c_i corresponds to a 
#         value at the point x_i = (i-1)h \in [0,1]. It can be considered as a 
#         function that is defined on a discrete set of points x_i.) 
# - The interval [0,1] is sampled at N points (including the boundary points).
#   The sampled points are stored in dom.
#
N = 100
dom = np.linspace(0, 1, N)
c = random.normal(0, 0.1, N) + dom - 0.7
#
# - Define a suitable parameter value for mu. ("Suitable" means: The variable z 
#                                              that is defined later should have 
#                                              the form (-1,-1,-1,...,1,1,1).)
# - Define the sampling rate h.
#
mu = 10
h = 1


### Construct the finite difference matrix D 
# 
# D = [-h  h  0  0  0  0  0]
#     [ 0 -h  h  0  0  0  0]
#     [ 0  0 -h  h  0  0  0]
#               ...
#     [ 0  0  0  0  0 -h  h]
#     [ 0  0  0  0  0  0  0]
#
# You can use the command np.eye(...).
#
D1 = -h * np.eye(N, k=0)
D2 = h * np.eye(N, k=1)
D = D1 + D2
D[-1] = 0

### Solve \nabla f(u) = 0 
#
# - Compute the derivative of f analytically. (See exercise sheet.)
# - Write the derivative in in matrix--vector notation. (See exercise sheet.)
# - Setup the linear system of equations of the form Au = b.
#
A = np.eye(N) + mu * D.T @ D
#
# - solve Au=b using numpy's np.linalg.solve function 
#
u = np.linalg.solve(A, c)

### Threshold/cut the solution to [0,1]
# 
# - Set z_i = +1, if u_i > 0, 
#       z_i = -1, if u_i < 0,
#       z_i =  0, otherwise.     
#

z = np.sign(u)

### Visualize the signals c, u, z using matplotlib as follows
#
# - Create a figure with title 'Signal Denoising' 
# - plot the signals c, u, z in different colors
# - add a legend with the labels
#   c: 'input', u: 'denoised', z: 'thresholded'
# - Add labels 'domain' and 'signal' to the axis 
# - modify the vertical axis to show values in [-1.1, 1.1]
# - Show the figure
#
plt.figure('Signal Denoising')
plt.plot(dom, c, label='input')
plt.plot(dom, u, label='denoised')
plt.plot(dom, z, label='thresholded')
plt.xlabel('domain')
plt.ylabel('signal')
plt.legend()
plt.ylim(-1.1, 1.1)
# uncomment to show plot
# plt.show()
plt.savefig(f'out_mu{mu}.png')



