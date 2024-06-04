import numpy as np 

def h_theta(x, theta):
    return 1/(1+np.exp(-x.dot(theta)))


def f_func(x, y, theta):
    m = x.shape[0]
    temp_val = (h_theta(x, theta))
    return -(1/m)*np.sum(y.dot(np.log(temp_val)) + (1-y).dot(np.log(1-temp_val))) 

def grad_func(x,  y, theta):
    m = x.shape[0]
    return (1/m)*x.T.dot(h_theta(x, theta)-y)

def hessian_func(x, theta):
    m = x.shape[0]
    h = h_theta(x, theta)
    hh = h*(1-h)
    return (1/m)*x.T.dot(np.diag(hh)).dot(x)