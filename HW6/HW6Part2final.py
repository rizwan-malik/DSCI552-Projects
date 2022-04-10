#####DSCI HW-6
# Muhammad Rizwan Malik
# Hamza Belal Kazi
##

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import normalize
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


with open('nonlinsep.txt', 'r') as in_file:
	np_data = loadtxt('nonlinsep.txt', delimiter=',')

X = np_data[:, 0:2]
y = np_data[:, 2]


def gaussian(x):
	sigma = 1.0
	denom = 2 * sigma * sigma
	numerator = -1 * x * x
	return math.exp(numerator / denom)


def getDist(p1, p2):
	return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


# Initializing values and computing H. Note the 1. to force to float type
m, n = X.shape
y = y.reshape(-1, 1) * 1.
X_dash = y * X
H = np.zeros((100, 100))

X = X.T
X = normalize(X)
X = X.T


for i in range(0, 100):
	for j in range(0, 100):
		yi = y[i]
		yj = y[j]
		temp = getDist(X[i], X[j])
		H[i][j] = yi * yj * gaussian(temp)

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

# Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

# Display results
print('Alphas = ', alphas[alphas > 1e-4])


# plot graph  based on parameter

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(X[S, 0], X[S, 1], marker='*', c='red', s=300)

plt.show()

