import numpy as np 

def h_theta(x, theta):
    return 1/(1+np.exp(-x.dot(theta)))


def f_func(x, y, theta):
    m = x.shape[0]
    temp_val = (h_theta(x, theta))
    return -(1/m)*np.sum(y.dot(np.log(temp_val)) + (1-y).dot(np.log(1-temp_val))) 

def grad_func(x,  y, theta):
    #todo: compute the gradient here.

    

