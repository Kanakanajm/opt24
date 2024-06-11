import time as clock
import numpy as np;
import matplotlib.pyplot as plt

################################################################################
### Regression Problem 
################################################################################
#                                                                              #
# Optimization problem:                                                        # 
#                                                                              #
#     min 0.5*|T(theta,t1,t2)*P_ - Q_|^2                                       # 
#                                                                              #
# Model:                                                                       #
#   |C|^2   is defined as sum_{i,j} C_{i,j}^2                                  #
#   T       2x3-matrix representing a Euclidean transformation in 2D:          #
#           T(theta,t1,t2)P_ = [cos(theta), -sin(theta)]*P_ + [t1, ..., t1];   #
#                              sin(theta),  cos(theta) ]      [t2, ..., t2];   #
#   P_       2xn matrix where each column represents a vector in 2D            #
#   Q_       2xn matrix where each column represents a vector in 2D            #
#           Q_ is approximately (up to noise) a transforma of the points in P_ #
#   n       number of points                                                   #
#                                                                              #
################################################################################


# load data from file
data = np.load('data.npy', allow_pickle=True).item();
P_ = data['P_'];
Q_ = data['Q_'];
n  = data['n'];

# setup problem structure:
#   Construct matrices P and Q such that 
#           min 0.5*|T(theta,t1,t2)*P_ - Q_|^2
#   is equivalent to 
#           min 0.5*|P*A(theta,t1,t2) - Q|^2
#   where A(theta,t1,t2) is the (column) vector 
#           [cos(theta),sin(theta),-sin(theta),cos(theta),t1,t2]
#
# TODO
###################
A = lambda theta, t1, t2: np.array([np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta), t1, t2])

# P is constructed with smaller matrices P_i where P_i = [[P_[1, i], 0, P_[2, i], 0, -1, 0], [0, P_[1, i], 0, P_[2, i], 0, -1]]
P = np.zeros((2*n, 6))
for i in range(n):
    P[2*i:2*i+2, :] = np.array([[P_[0, i], 0, P_[1, i], 0, -1, 0], [0, P_[0, i], 0, P_[1, i], 0, -1]])

# Q is constructed with smaller matrices Q_i where Q_i = [Q_[1, i], Q_[2, i]]
Q = Q_.flatten()

F = lambda x: P @ A(x[0], x[1], x[2]) - Q
tau = 0.01

################################################################################
### Section 1: Run Gauss-Newton Method ####################################################
print('');
print('************************************************************************');
print('*** Gauss-Newton Method ***');
print('***************************');

# general parameter
maxiter = 50;
check = 1;
tol = 1e-12;

# initialization
theta = 0.0;
t1 = 0.0;
t2 = 0.0;
x_kp1 = np.array([theta,t1,t2]);

time = 0;
breakvalue = 1;
for iter in range(1,maxiter+1):
    
    stime = clock.time();
    
    # update variables
    x_k = x_kp1.copy();

    # set up linear subproblem of the form
    #    min 0.5|P_lin*x_kp1 - Q_lin|^2
    # using the matrices Q and P from above.
    # TODO
    ###################
    F_k = F(x_k)

    # compute the Jacobian of F with forward differences
    JF_k = np.zeros((2*n, 3))
    h = 1e-6
    for i in range(3):
        x = x_k.copy()
        x[i] += h
        JF_k[:, i] = (F(x) - F_k) / h
    
    grad_F_k = JF_k.T

    # solve linear subproblem using np.linalg.solve
    # TODO
    ###################
    assert grad_F_k.shape == (3, 2*n)
    print(F_k.shape)
    assert F_k.shape == (2*n,)
    d_k = np.linalg.solve(grad_F_k @ grad_F_k.T, -np.dot(grad_F_k, F_k))

    x_kp1 = x_k + tau * d_k


    print(x_kp1)

    # check breaking condition
    res = np.sum((x_kp1 - x_k)**2);
    if res < tol:
        breakvalue = 2;
    time = time + (clock.time() - stime);

    # print info
    if (iter % check == 0):
        print('iter: %d, time: %5f, res: %f' % (iter, time, res));

    # handle breaking condition
    if breakvalue == 2:
        print('Tolerance value reached!!!');
        break;

gauss_newton_sol = x_kp1.copy()

exit()
################################################################################
### Section 2: Run Levenberg-Marquardt Method
#####################################################
print('')
print('************************************************************************')
print('*** Levenberg-Marquardt Method ***')
print('***************************')

# general parameter
maxiter = 50
check = 1
tol = 1e-12

# initialization
theta = 0.0
t1 = 0.0
t2 = 0.0
x_kp1 = np.array([theta, t1, t2])

time = 0
breakvalue = 1
for iter in range(1, maxiter+1):
    # Implement Levenberg-Marquardt Method update step here.
    
    stime = clock.time()

    # update variables
    x_k = x_kp1.copy()

    # set up linear subproblem of the form
    #    min 0.5|P_lin*x_kp1 - Q_lin|^2
    # using the matrices Q and P from above.
    # TODO
    # This part is same as that of Gauss-Newton Method.
    ###################


    # solve linear subproblem using np.linalg.solve

    # TODO
    ###################
    print(x_kp1)

    # check breaking condition
    res = np.sum((x_kp1 - x_k)**2)
    if res < tol:
        breakvalue = 2
    time = time + (clock.time() - stime)

    # print info
    if (iter % check == 0):
        print('iter: %d, time: %5f, res: %f' % (iter, time, res))

    # handle breaking condition
    if breakvalue == 2:
        print('Tolerance value reached!!!')
        break

levenberg_marquardt_sol = x_kp1.copy()


################################################################################
### Gauss-Newton evaluation ####################################################

theta = gauss_newton_sol[0]
t1 = gauss_newton_sol[1]
t2 = gauss_newton_sol[2]

t = np.array([t1, t2])
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
TP = np.dot(R, P_) + t.reshape((2, 1))

fig = plt.figure()
plt.plot(P_[0, :], P_[1, :], 'x', markersize=8, color=(1, 0, 0, 1))
plt.plot(TP[0, :], TP[1, :], 'o', markersize=8, color=(1, 1, 0, 1))
plt.plot(Q_[0, :], Q_[1, :], 'x', markersize=8, color=(0, 0, 1, 1))

# T*P are the transformed points computed using the optimal parameters
plt.legend(['P', 'T*P', 'Q'])
plt.title('Gauss-Newton Evaluation')
plt.show(block=False)


################################################################################
### Levenberg-Marquardt evaluation #############################################

theta = levenberg_marquardt_sol[0]
t1 = levenberg_marquardt_sol[1]
t2 = levenberg_marquardt_sol[2]

t = np.array([t1, t2])
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
TP = np.dot(R, P_) + t.reshape((2, 1))

fig2 = plt.figure()
plt.plot(P_[0, :], P_[1, :], 'x', markersize=8, color=(1, 0, 0, 1))
plt.plot(TP[0, :], TP[1, :], 'o', markersize=8, color=(1, 1, 0, 1))
plt.plot(Q_[0, :], Q_[1, :], 'x', markersize=8, color=(0, 0, 1, 1))

# T*P are the transformed points computed using the optimal parameters
plt.legend(['P', 'T*P', 'Q'])
plt.title('Levenberg-Marquardt Evaluation')
plt.show()

