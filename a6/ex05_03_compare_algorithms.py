from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import load_iris
from sklearn import preprocessing
from NewtonMethod import newton
from BFGS import bfgs 
from L_BFGS import l_bfgs
from GradientDescent import gd 
import numpy as np

import matplotlib.pyplot as plt

##############################################################################
# Problem Description

## Consider a set of points $\{(x_1,y_1),(x_2,y_2),\ldots,(x_m,y_m)\}$ where 
# $x_i \in \R^n$ and $y_i \in \{0,1\}$.

# For $w \in \R^n$, denote the following 

##    h_{w}(x) = \frac{1}{1+e^{-\scal{w}{x}}}\,,

##    f(w) = -\frac{1}{m}\sum_{i=1}^m\left(y_i\log(h_{w}(x_i)) 
#                   + (1-y_i)\log(1 - h_{w}(x_i)) \right)\,.

## We want to optimize f(w) here.

## We compare here Gradient Descent, Newton Method, BFGS, L-BFGS methods. 

##############################################################################

data = load_iris()
X = data['data'][:-50,:]
X = preprocessing.normalize(X)

Y = data['target'][:-50]

np.random.seed(0)


# specify  data
model = {
    'X':     X,
    'y':     Y,
}


################################################################################
### Run Algorithms #############################################################

# general parameter
maxiter = 100
check = 1

# initialization
x0 = np.ones(4)*0.01

# taping
xs = []
rs = []
ts = []
cols = []
legs = []
nams = []
funcs = []




# turn algorithms to be run on or off

run_bfgs = 1        # BFGS method
run_l_bfgs = 1      # L-BFGS method
run_newton = 1      # Newton Method
run_gd = 1      # Gradient Descent



################################################################################

if run_newton:

    print('')
    print('********************************************************************************')
    print('***Newton Method***')
    print('************************')

    options = {
        'init':           x0,
        'storeFuncs':     True,
        'storeResidual':  True,
        'storeTime':      True,
    }

    output = newton(model, options, maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    funcs.append(output['seq_funcs'])
    ts.append(output['seq_time'])
    cols.append((0.5, 0.1, 0.1, 1))
    legs.append('Newton')
    nams.append('Newton')

    newton_sol = output['sol']



################################################################################
if run_bfgs:
    # BFGS Method
    print('')
    print('********************************************************************')
    print('*** BFGS ***')
    print('**************************')

    options = {
        'init':           x0,
        'storeFuncs':     True,
        'storeResidual':  True,
        'storeTime':      True,
    }

    output = bfgs(model, options,  maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    funcs.append(output['seq_funcs'])
    ts.append(output['seq_time'])
    cols.append((0.1, 0.5, 0.5, 1))
    legs.append('BFGS')
    nams.append('BFGS')

    bfgs_sol = output['sol']

################################################################################
if run_l_bfgs:
    # L-BFGS Method
    print('')
    print('*******************************************************************')
    print('*** L-BFGS ***')
    print('**************************')

    options = {
        'init':           x0,
        'storeFuncs':     True,
        'storeResidual':  True,
        'storeTime':      True,
    }

    output = l_bfgs(model, options,  maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    funcs.append(output['seq_funcs'])
    ts.append(output['seq_time'])
    cols.append((0.1, 0.1, 0.5, 1))
    legs.append('L-BFGS')
    nams.append('L-BFGS')

    lbfgs_sol = output['sol']


################################################################################

if run_gd:

    print('')
    print('*********************************************************************')
    print('***Gradient Method***')
    print('************************')

    options = {
        'init':           x0,
        'storeFuncs':     True,
        'storeResidual':  True,
        'storeTime':      True,
    }

    output = gd(model, options, maxiter, check)
    xs.append(output['sol'])
    rs.append(output['seq_res'])
    funcs.append(output['seq_funcs'])
    ts.append(output['seq_time'])
    cols.append((0.5, 0.5, 0.1, 1))
    legs.append('Gradient Descent')
    nams.append('Gradient Descent')

    gd_sol = output['sol']

print('===============================================================')
for i in range(4):
    print(legs[i] + ':' + str(rs[i][-1]))
    print('----------------------------------')


# We are providing below the code for plotting the results.

# Plotting data and decision boundary.

# Using Incremental PCA to project onto lower dimensional space
ipca = IncrementalPCA(n_components=2)
X_after_ipca = ipca.fit_transform(X)


def plot_decision_boundary(w, filename='newton_decision_boundary.png'):
    fig = plt.figure()
    for i in range(len(Y)):
        if Y[i] == 0:
            plt.scatter(X_after_ipca[i, 0], X_after_ipca[i, 1], c='b')
        elif Y[i] == 1:
            plt.scatter(X_after_ipca[i, 0], X_after_ipca[i, 1], c='r')

    w_pca = ipca.transform(np.array([w]))

    x_0_min = X_after_ipca[:,0].min()
    x_0_max = X_after_ipca[:, 0].max()

    x_1_min = X_after_ipca[:, 1].min()
    x_1_max = X_after_ipca[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_0_min, x_0_max, 0.0001),
                        np.arange(x_1_min, x_1_max, 0.0001))

    def line_plot(x):
        return x.dot(w_pca[0,:])

    Z = line_plot(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, cmap="pink", levels=[0])

    plt.grid(True)
    plt.title('Data visualization and Decision boundary')
    plt.savefig(filename)
    plt.close(fig)

print("--plotting and saving newton method based decision boundary --")
plot_decision_boundary(newton_sol, filename='figures/newton_sol.png')

print("--plotting and saving bfgs method based decision boundary --")
plot_decision_boundary(bfgs_sol, filename='figures/bfgs_sol.png')

print("--plotting and saving lbfgs method based decision boundary --")
plot_decision_boundary(lbfgs_sol, filename='figures/lbfgs_sol.png')

print("--plotting and saving gradient descent based decision boundary --")
plot_decision_boundary(gd_sol, filename='figures/gd_sol.png')


fig = plt.figure()
for i in range(4):
    plt.loglog(rs[i], '-', color=cols[i], linewidth=2)
plt.legend(legs[:4])
plt.xlabel('Iterations (log scale)')
plt.ylabel('Residual value (log scale)')
plt.title('Residual vs Iterations')
plt.grid(True)
plt.show(block=False)


fig2 = plt.figure()
for i in range(4):
    plt.loglog(funcs[i], '-', color=cols[i], linewidth=2)
plt.legend(legs[:4])
plt.xlabel('Iterations (log scale)')
plt.ylabel('Function value (log scale)')
plt.title('Function vs Iterations')
plt.grid(True)
plt.show()
