import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import load_iris
from sklearn import preprocessing
import numpy as np


import matplotlib as mpl
mpl.use('TkAgg')


##############################################################################
# Problem Description
# Linearly separable data - Binary classification

# (x_i,y_i) => training data for i = 1,...,m.
# I am assuming implicitly x_i contains 1 as the last element,
# this is tilde x_i as per the exercise.

# We want to find w such that we  have:
# <w,x_i> >= 1 if y_i = 1,
# <w,x_i> <= 1 if y_i = -1,

# where above two equations can be unified as following:
# y_i(<w,x_i>) >= 1 for i = 1,...,m.

# Then set of weights which can result in linear separability is 
# W = {w : y_i(<w,x_i>) >= 1 for i = 1,...,m. }.

# The goal is to find a point in W.

# Denote C_i = {w : y_i(<w,x_i>) >= 1} for some i in {1, ..., m}.

# Then, our update step using POCS algorithm is 
# w_kp = P_{C_m}(P_{C_{m-1}}(....P_{C_1}(w_k))).

##############################################################################

data = load_iris()
X = data['data']
X = preprocessing.normalize(X)

Y = data['target']

# operations needed for binary classification
Y[Y == 1] = -1
Y[Y == 2] = -1
Y[Y == 0] = 1 

X_new = np.ones((150,5)) # adding ones column
X_new[:,:-1] = X


X = X_new   # this is already tilde X, where each row is the data sample \tilde x_i

# general parameter
maxiter = 1000
check = 1

# initialization
w = np.ones(5)*0.1

(m, n) = X.shape

def check_halfspace(w, x, y):
    # TODO: Check if y_i(<w,x_i>) >= 1 holds, return True if so else False.


def P_C(w,i):
    # TODO: Projection of w onto the hyperplane {w : y_i(<w,x_i>) >= 1}.
    # You may use check_halfspace if required.

rs = [] # residual values

for iter in range(maxiter):
    prev_w = w.copy()

    # TODO: Compute the POCS update
    

    # Misclassification error count computation
    count = 0
    for i in range(m):
        x = X[i, :]
        y = Y[i]
        if check_halfspace(w, x, y):
            pass
        else:
            count +=1
    
    res_value = np.linalg.norm(w - prev_w)**2
    print('Misclassification Error ', count, ' residual ', res_value)
    rs.append(res_value)

# Plotting data and decision boundary.

# Using Incremental PCA to project onto lower dimensional space
ipca = IncrementalPCA(n_components=2)
X_after_ipca = ipca.fit_transform(X)


def plot_decision_boundary(w, filename='decision_boundary.png'):
    fig = plt.figure()
    for i in range(len(Y)):
        if Y[i] == -1:
            plt.scatter(X_after_ipca[i, 0], X_after_ipca[i, 1], c='b')
        elif Y[i] == 1:
            plt.scatter(X_after_ipca[i, 0], X_after_ipca[i, 1], c='r')

    w_pca = ipca.transform(np.array([w]))

    x_0_min = X_after_ipca[:, 0].min()
    x_0_max = X_after_ipca[:, 0].max()

    x_1_min = X_after_ipca[:, 1].min()
    x_1_max = X_after_ipca[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_0_min, x_0_max, 0.0001),
                         np.arange(x_1_min, x_1_max, 0.0001))

    def line_plot(x):
        return x.dot(w_pca[0, :])

    Z = line_plot(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, cmap="pink", levels=[0])

    plt.grid(True)
    plt.title('Data visualization and Decision boundary')
    plt.show(block=False)
    plt.savefig(filename)
    # plt.close(fig)


print("--plotting and saving decision boundary --")
plot_decision_boundary(w, filename='decision_boundary.png')

fig = plt.figure()
plt.loglog(rs, '-', linewidth=2)
plt.xlabel('Iterations (log scale)')
plt.ylabel('Residual value (log scale)')
plt.title('Residual vs Iterations')
plt.grid(True)
plt.show()

