from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

##################################################################
##  We consider the problem of binary classification
##  as described in Exercise 3.
#
# We consider the problem of binary classification.
# Here, we are provided with input data (x_i,y_i), i= 1,...,m,
# where x_i in R^{30} and y_i in {0,1} for all
# i in {1,...,m}.
#
#  The goal is to construct a classifier (function),
# where for any given x in R^{30} it results in the output
# y in {0,1}. For any x in R, the sigmoid function
# sigma : R -> R is given by sigma(x) := {1}/{1 + \exp(-x)}\,.

# Based on the input data,  consider the functions
# p : R^{30} x R^{10 x30} x R^{10} x R^{1 x10} x R -> R,
# f : R^{10 x 30} x R^{10} x R^{1 x 10} xR -> R defined as following:
# p(x, W_1, b_1, W_2,  b_2 ) =   sigma(W_2*sigma(W_1x + b_1) + b_2)\,,
# f(W_1, b_1, W_2,  b_2) = -(1/m)*sum_{i=1}^m
#           (y_i\log(p(x_i, W_1, b_1, W_2,  b_2))
#           + (1 - y_i)\log(1- p(x_i, W_1, b_1, W_2,  b_2)))\,,
# where  W_1 in R^{10 x30}, b_1 in R^{10},  W_2 in R^{1 x10},
# b_2 in R, xin R^{30} and sigma is applied element wise.
#
# Based on the before mentioned functions, your task is to solve
# the following optimization problem
# \min_{W_1, b_1, W_2, b_2 in R} f(W_1, b_1, W_2,  b_2)
# using Gradient Descent algorithm with Backtracking Line Search method,
# by filling in the TODOs given in this file.
#
# The crucial part in the code is to obtain the gradient  nabla f
# using the backward mode of automatic differentiation.
#
# For fixed {bar W}_1 in R^{10 x30},\, {bar b}_1 in R^{10},\,
# {bar W}_2 in R^{1 x10},\, {bar b}_2 in R, the classifier
# C : R^{30} -> {0,1} for any x in R^{30} is given by
# C(x) = 0, if  p(x, {bar W}_1, {bar b}_1, {bar W}_2,  {bar b}_2) <= 0.5,
# and 1, otherwise
#
# The classification error is given by
#   E(x) = (1/m)sum_{i=1}^mI_i\,,
# where I_i = 1 if C(x_i) is not equal to y_i, else I_i = 0.

##################################################################


################# Loading data  ##################################
data = load_breast_cancer()
X = data['data']
Y = data['target']
(m, n) = X.shape
m = Y.shape
##################################################################


def sigmoid_derivative(x):
    # derivative of the sigmoid function
    y = 1/(1 + np.exp(-x))
    return y*(1-y)


# Dimensions for reference
# X => 569 X 30
# Y => 30
# W_1 => 10 X 30
# b_1 => 10
# W_2 => 1 X 10
# b_2 => 1


def loss_func_and_classification_error(W_1, b_1, W_2, b_2, X, Y):
    # This function returns the loss and the classification error.
    # X,Y => Input data

    error = 0  # to track classification error
    loss = 0
    (m, n) = X.shape
    for i in range(m):
        # input vector
        x = X[i, :]

        # TODO: Forward pass for input x and store the final output to a_2
        # By forward pass, we mean to evaluate the expression 
        # sigma(W_2*sigma(W_1x + b_1) + b_2)
        


        # prediction
        p = a_2

        # loss values
        loss += -(1/m)*(Y[i]*np.log(p) + (1-Y[i])*(np.log(1-p)))

        # using the classifier
        if p <= 0.5:
            y_pred = 0
        else:
            y_pred = 1

        # accumulating error
        if y_pred != Y[i]:
            error += 1

    return loss[0], error/m  # dividing by m to normalize


max_iter = 100
backtrackingMaxiter = 10

# Initialization of the variables
W_1_kp = np.random.random((10, 30))
b_1_kp = np.random.random((10))

W_2_kp = np.random.random((1, 10))
b_2_kp = np.random.random((1))

loss_vals = []
error_vals = []

# initial loss and classification error
f_kp1, error = loss_func_and_classification_error(
    W_1_kp, b_1_kp, W_2_kp, b_2_kp, X, Y)

loss_vals.append(f_kp1)
error_vals.append(error)

breakvalue = 1

for iter in range(max_iter):
    ## Update the variables
    W_1_k = W_1_kp.copy()
    b_1_k = b_1_kp.copy()
    W_2_k = W_2_kp.copy()
    b_2_k = b_2_kp.copy()

    f_k = f_kp1.copy()
    (m, n) = X.shape

    ## computing the gradient
    grad_W_1 = 0
    grad_b_1 = 0
    grad_W_2 = 0
    grad_b_2 = 0

    ## Looping over all samples to accumulate the gradient
    for i in range(m):
        x = X[i, :]

        ## TODO: Forward pass for input x and store the output to a_2
        ## you might need to keep track of intermediate variables
        

        ## TODO: Evaluate the gradient with reverse mode automatic differentiation
        ## the derivatives with respect to W_1 must be stored in d_w_1, 
        # similarly for b_1 use d_b_1, for w_2 use d_w_2, for b_2 use d_b_2.
        ## Make sure to consider (1/m) factor in f.
        
        # accumulate the gradient for each sample
        grad_W_1 = grad_W_1 + d_w_1
        grad_b_1 = grad_b_1 + d_b_1
        grad_W_2 = grad_W_2 + d_w_2
        grad_b_2 = grad_b_2 + d_b_2

    tau = 0.9
    gamma = 1e-2

    ## Backtracking
    for iterbt in range(0, backtrackingMaxiter):

        #TODO: Do the gradient descent update on W_1_k and store the
        # result in W_1_kp variable. Similarly, apply the gradient descent update
        # b_1_k, W_2_k, b_2_k and store it in b_1_kp, W_2_kp, b_2_kp respectively. 
        # For all the variables, use 'tau' as step-size.


        # squared grad norm
        squared_grad_norm = np.linalg.norm(grad_W_1)**2\
            + np.linalg.norm(grad_W_2)**2\
            + np.linalg.norm(grad_b_1)**2\
            + np.linalg.norm(grad_b_2)**2

        # compute new value of smooth part of objective
        f_kp1, _ = loss_func_and_classification_error(
            W_1_kp, b_1_kp, W_2_kp, b_2_kp, X, Y)

        # no backtracking
        if backtrackingMaxiter == 1:
            break

        # check Armijo condition
        Delta = -gamma*tau*squared_grad_norm
        if (f_kp1 < f_k + Delta + 1e-8):
            break
        else:
            tau = tau*0.9
            if (iterbt+1 == backtrackingMaxiter):
                breakvalue = 3

    f_kp1, error = loss_func_and_classification_error(
        W_1_kp, b_1_kp, W_2_kp, b_2_kp, X, Y)

    loss_vals.append(f_kp1.copy())
    error_vals.append(error)

    # print info
    print('iter: %d, tau: %f, loss val: %f, classification error val: %f' %
          (iter, tau, f_kp1, error))

    # handle breaking condition
    if breakvalue == 2:
        print('Tolerance value reached!!!')
        break
    elif breakvalue == 3:
        print('Not enough backtracking iterations!!!')
        break


### --- Convergence plot ----
fig = plt.figure()
plt.loglog(loss_vals, '-',  linewidth=2)
plt.xlabel('Iterations (log scale)')
plt.ylabel('Function value (log scale)')
plt.title('Function vs Iterations')
plt.grid(True)
plt.show(block=False)


### --- Convergence plot ----
fig2 = plt.figure()
plt.plot(error_vals, '-',  linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Classification error')
plt.title('Classification error vs Iterations')
plt.grid(True)
plt.show()
