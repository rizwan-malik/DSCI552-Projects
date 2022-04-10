#####DSCI HW-6
# Muhammad Rizwan Malik
# Hamza Belal Kazi
##

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

with open('linsep.txt', 'r') as in_file:
	np_data = loadtxt('linsep.txt', delimiter=',')

X = np_data[:, 0:2]
y = np_data[:, 2]

# Initializing values and computing H. Note the 1. to force to float type
m, n = X.shape
y = y.reshape(-1, 1) * 1.
X_dash = y * X
H = np.dot(X_dash, X_dash.T) * 1.

# Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

# Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

# w parameter in vectorized form
w = ((y * alphas).T @ X).reshape(-1, 1)

# Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

# Computing b
b = y[S] - np.dot(X[S], w)

# Display results
print('Alphas = ', alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])

# plot graph  based on parameter

plt.scatter(X[:, 0], X[:, 1], c=y)

slope = -w[0][0] / w[1][0]
intercept = -b[0] / w[1][0]

# abline(slope, intercept)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')

plt.show()
