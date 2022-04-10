import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from numpy.linalg import eig as eig
from mpl_toolkits.mplot3d import Axes3D


# Algorithm for reducing the dimensionality of given vectors.
def PCA_algorithm(my_data_points):
	# Computes mean of the given set of vectors, by summing all the columns and dividing by the number of vectors.
	def find_mean(data_array):
		sum_columns = data_array.sum(axis=0)
		mean = sum_columns / len(data_array)
		# print(sum_columns)
		return mean

	# Subtracts mean from all the N vectors and returns the new Xi matrix.
	def get_new_x(data_array, mean):
		first_column = data_array[:, 0] - mean[0]
		second_column = data_array[:, 1] - mean[1]
		third_column = data_array[:, 2] - mean[2]
		data_array[:, 0] = first_column[:, ]
		data_array[:, 1] = second_column[:, ]
		data_array[:, 2] = third_column[:, ]
		return data_array

	# Computes the covariance matrix by simply multiplying each vector with its transpose (outer product) and then
	# summing those over N and in the end dividing by N.
	def get_covariance_matrix(data_array):
		matrix_sum = 0
		for row in data_array:
			row = np.reshape(row, (3, 1))
			matrix_sum = matrix_sum + np.matmul(row, np.transpose(row))

		covariance = matrix_sum / len(data_array)
		return covariance

	# Returns the sorted eigen values and eigen vectors of the covariance matrix.
	def get_eig_values_and_vectors(covariance):
		e, v = eig(covariance)

		idx = e.argsort()[::-1]  # argsort returns an array of indices which correspond to a sorted version of
		# eigenvalues array in ascending order. [::-1] is simply to reverse the order to descending order.
		e = e[idx]  # Actually sorts the eigenvalues vector.
		v = v[:, idx]  # Sorts the eigenvectors matrix using the same indices.
		return e, v

	# Returns the truncated matrix of eigenvectors.
	def get_u_truncate(matrix, k):
		trunc = matrix[:, 0:k]
		print("First principle component vector: " + str(round(trunc[0,0],2)) +" i ,"+ str(round(trunc[1,0],2)) +" j ,"+ str(round(trunc[2,0],2))+" k" )
		print("Second principle component vector: " + str(round(trunc[0,1],2)) +" i ,"+str(round(trunc[1,1],2)) +" j ,"+ str(round(trunc[2,1],2))  +" k" )

		return trunc

	# Computes the matrix of z vectors by simply multiplying transpose of u_trunc with the corresponding x vector.
	def find_z(data_array, trunc_matrix, k):
		z = np.empty((len(data_array), k))

		for i, row in enumerate(data_array):
			z[i] = np.matmul(np.transpose(trunc_matrix), np.transpose(row))

		return z

	# K is a constant, which I passed in different functions. All the function calls were made from here.
	K = 2
	mu = find_mean(my_data_points)
	new_data_points = get_new_x(my_data_points, mu)
	covariance_matrix = get_covariance_matrix(new_data_points)
	eig_values, eig_vectors = get_eig_values_and_vectors(covariance_matrix)
	u_trunc = get_u_truncate(eig_vectors, K)
	z_matrix = find_z(data_points, u_trunc, K)
	return z_matrix


# Plots the 3d data.
def plot_results1(data_array):
	# fig = plt.figure()
	ax = fig.add_subplot(2,1,1, projection='3d')
	xs = data_array[:, 0]
	ys = data_array[:, 1]
	zs = data_array[:, 2]

	ax.scatter(xs, ys, zs, c='r', marker='o')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	return


# Plots 2d data.
def plot_results2(data_array):
	# fig = plt.figure()
	ax = fig.add_subplot(212)
	xs = data_array[:, 0]
	ys = data_array[:, 1]

	ax.scatter(xs, ys)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')

	return


# Reads tab delimited file and returns a matrix of vectors.
def readFile():
	with open('pca-data.txt', 'r') as in_file:
		np_data_points = loadtxt('pca-data.txt', delimiter='\t')
	return np_data_points


data_points = readFile()
reduced_matrix = PCA_algorithm(data_points)

fig = plt.figure()

plot_results1(data_points)
# plt.show()

plot_results2(reduced_matrix)
plt.show()


#print(readFile())
