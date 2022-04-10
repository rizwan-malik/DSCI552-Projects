import numpy as np


# Returns the path with max probability and tracking tables, given the Observations (y), Transition Model (A),
# Observation Model (B) and Prior Probabilities (Pi)
def viterbi(y, A, B, Pi):
	K = A.shape[0]  # k=10 over here which basically represent the size of the state space.

	T = len(y)
	T1 = np.empty((K, T), 'd')
	T2 = np.empty((K, T), 'B')

	# Initialize the tracking tables from first observation
	T1[:, 0] = Pi.T * B[:, y[0]]  # y[index] can be used directly since our columns in Observation Model [0 to 11]
	# have 1-1 correspondence with the values which observations can take.
	T2[:, 0] = 0

	# Iterate through the observations updating the tracking tables
	for i in range(1, T):
		T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
		T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

	# Build the output, optimal model trajectory
	x = np.empty(T, 'B')
	x[-1] = np.argmax(T1[:, T - 1])
	for i in reversed(range(1, T)):
		x[i - 1] = T2[x[i], i]

	return x, T1, T2


K = 10  # total possible states it can take
N = 12  # total number of possible observations values   ( [8-1, 6-1, 4-1, 6-1, 5-1, 4-1, 5-1, 5-1, 7-1, 9-1] )
y = np.array([8, 6, 4, 6, 5, 4, 5, 5, 7, 9])

A = np.zeros((K, K))

A[0, 1] = 1  # set first row special 1st row
A[-1, 8] = 1  # set last row special 10th row


# Defining the Transition Model and Observation Model as per the problem specification.
# For transition model setting value = 0.5 for A[row, yi-1] and A[row, yi+ 1]
for row in range(1, K-1):
	for col in range(0, K):
		if col == row:
			A[row, col - 1] = 0.5
			A[row, col + 1] = 0.5

# For transition model setting value = 1/3 for B[row, yi-1], B[row, yi], B[row, yi+1]
B = np.zeros((K, N))
for row in range(0, K):
	for col in range(0, N):
		if col == row + 1:
			B[row, col - 1] = 1 / 3
			B[row, col] = 1 / 3
			B[row, col + 1] = 1 / 3

# For Prior Probabilities all the states are equally probable hence every element is set = 1/10
Pi = np.zeros((K, 1))
for i in range(0, K):
	Pi[i] = 1 / 10

xf, T1f, T2f = viterbi(y, A, B, Pi)

# Since the above function iterates over the models based on 0-indexing, hence to get the values corresponding to our
# problem we need to add 1 to the final values since our domain is 1-10.
for i in range(0, len(xf)):
	xf[i] = xf[i] + 1

print("Most likely sequence of states that the model went through: ", xf)
