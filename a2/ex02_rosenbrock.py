import numpy as np
from numpy import random;
import matplotlib.pyplot as plt

################################################################################
### 2D Rosenbrock problem ######################################################
#                                                                              #
# Optimization problem:                                                        #
#                                                                              #
#     min_x  f(x1,x2) = (a-x1)^2 + b(x2-x1^2)^2                                #
#                                                                              #
# Model:                                                                       #
#     a       scalar parameter                                                 #
#     b       scalar parameter                                                 #
#     N       dimension of optimization variable                               #
#                                                                              #
################################################################################


def Model(dataset):
    """
    Define the model data of the problem to be solved in this project.

    Parameters:
    -----------
    dataset : string
        Choose one the following predefined datasets:
            'standard'            Commonly used parameter setting.

    Returns:
    --------
    struct
    .'a'      scalar parameter
    .'b'      scalar parameter
    .'N'      dimension of optimization variable

    """

    if dataset == 'standard':

        a = 1;
        b = 100;

    else:

        print 'The dataset %s was not defined!' % dataset

    return {'a': a, 'b': b, 'N': 2};


# generate problem
model = Model('standard');

################################################################################
### Run Algorithms #############################################################
from GradientDescent import gd;
from InExactLineSearch_GD import gd as gd_inexact;

# general parameter
maxiter = 7500;
check = 1000;
tol = 1e-2;

# initialization
x0 = np.array([0.3,2.5]);

# taping
xs = [];
xss = [];     # sequences of iterates 
rs = [];
ts = [];
cols = [];
legs = [];
nams = [];

# HINT: Test each algorithm by setting to 1 then finally set all of them to 1
run_gd                  = 1;    # Gradient Descent
run_gd_bt               = 1;    # Gradient Descent with backtracking
run_gd_inexact          = 1;    # Gradient Descent with Inexact line search

################################################################################
if run_gd: 
    
    print('');
    print('********************************************************************************');
    print('*** Gradient Descent ***');
    print('************************');

    options = {
        'init':           x0,
        ### TODO: find a suitable step size
        'stepsize':       0.00001,
        'storeResidual':  True,
        'storePoints':    True,
        'storeTime':      True
    }
    
    output = gd(model, options, tol, maxiter, check);
    xs.append(output['sol']);
    xss.append(output['seq_x']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.7,0.85,0,1));
    legs.append('GD');
    nams.append('GD');

################################################################################
if run_gd_bt: 
    
    print('');
    print('********************************************************************************');
    print('*** Gradient Descent with Backtracking ***');
    print('******************************************');

    options = {
        'init':                 x0,
        ### TODO: find a suitable 
        ### - 'stepsize'
        ### - 'backtrackingMaxiter': maximal number of backtracking iterations
        ### - 'backtrackingFactor':  scaling of the step size during backtracking
        ### - 'Armijoparam': gamma-parameter in the Armijo condition
        ### 
        'stepsize':             1.0,
        'backtrackingMaxiter':  5,
        'backtrackingFactor':   1e-2,
        'Armijoparam':          0.99,
        'storeResidual':        True,
        'storePoints':          True,
        'storeTime':            True
    }
    
    output = gd(model, options, tol, maxiter, check);
    xs.append(output['sol']);
    xss.append(output['seq_x']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0.8,0.9,1));
    legs.append('GD-BT');
    nams.append('GD_BT');

################################################################################
if run_gd_inexact: 
    
    print('');
    print('********************************************************************************');
    print('*** Gradient Descent with  Inexact Line Search ***');
    print('******************************************');

    # TODO: Just complete gd_inexact function

    options = {
        'init':                 x0,
        'storeResidual':        True,
        'storePoints':          True,
        'storeTime':            True
    }
    
    output = gd_inexact(model, options, tol, maxiter, check);
    xs.append(output['sol']);
    xss.append(output['seq_x']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.3,0.2,0.5,1));
    legs.append('GD-LS-INEXACT');
    nams.append('GD-LS-INEXACT');


################################################################################
### evaluation #################################################################
nalgs = len(rs);

# print final residual
print('');
for i in range(0,nalgs):
    print('alg: %s, time: %f, res: %f' % (legs[i], ts[i][-1], rs[i][-1]))

# plotting
fig1 = plt.figure();
plt.title('Rosenbrock2D')


for i in range(0,nalgs):
    plt.plot(ts[i], rs[i], '-', color=cols[i], linewidth=2);


plt.legend(legs);
plt.yscale('log');
plt.xscale('log');

plt.xlabel('time')
plt.ylabel('residual');

plt.show(block=False); # This command can be helpful to test.



# plot contour lines and optimization trajectories
a = model['a'];
b = model['b'];
delta = 0.025;
x0 = np.arange(-2.0, 2.0, delta);
x1 = np.arange(-1.0, 3.0, delta);
(X0, X1) = np.meshgrid(x0, x1);
Z = (a-X0)**2 + b*(X1-X0**2)**2;

fig1=plt.figure();
plt.title('Rosenbrock2D contours');
plt.xlabel('x0');
plt.ylabel('x1');
plt.contour(X0, X1, Z, 40, label='_nolegend_');
plt.plot(a, a**2, 'x', markersize=8, color=(1,0,0,1), label='_nolegend_'); 
for i in range(0,nalgs):
    x = xss[i];
    plt.plot(x[0,:], x[1,:], '-', markersize=8, color=cols[i]); 
plt.legend(legs);

plt.show(block=True);








