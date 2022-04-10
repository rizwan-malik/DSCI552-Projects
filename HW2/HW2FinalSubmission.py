### Team Members #####
# Muhammad Rizwan Malik
# Hamza Belal Kazi

# DSCI 552  -  HW2

import csv
import random
import math
import numpy as np
from numpy import loadtxt
from numpy.linalg import inv, det, multi_dot
from matplotlib import pyplot as plt


def KMeansAlgorithm(my_data_points, mu):
	# Returns a list whose length is equal to data list. Each list element corresponds to the centroid to which that
	# data
	# point belongs. For each data point it calculates distance to all the centroids and then chooses the one which
	# gave
	# the minimum distance.

	def assignCentroids(data_points, centroids):
		index_list = [-1] * len(
			data_points)  # Initialize the list of None of length equal to that of data points list.
		for i in range(len(data_points)):
			distance = float('inf')
			for j in range(len(centroids)):
				x_square = (centroids[j][0] - data_points[i][0]) ** 2
				y_square = (centroids[j][1] - data_points[i][1]) ** 2
				temp_distance = math.sqrt(x_square + y_square)
				if temp_distance < distance:  # Makes the comparison between currently calculated distance and
					# previously
					# stored one.
					distance = temp_distance
					index_list[i] = j  # Puts in the index_list the index number of the centroid which gave the least
		# distance.
		return index_list

	# Returns a list of centroids. It iterates through the whole of index_list (which holds the centroid to which
	# corresponding data points belong) and looks for 0, 1, 2 values (centroid numbers) and then stores the index of
	# in a
	# separate sub-list. It then goes through each sublist and picks out the index number and using that index
	# number it
	# picks out the data point from data_points list and keeps a running sum of x and y coordinates and eventually
	# computes the average / mean.
	def getNewCentroids(data_points, indices):
		temp_index = [[], [], []]
		mu = []
		for i, j in enumerate(indices):
			if 0 in indices and j == 0:
				temp_index[0].append(i)
			elif 1 in indices and j == 1:
				temp_index[1].append(i)
			elif 2 in indices and j == 2:
				temp_index[2].append(i)
		count = 0
		for centroid in temp_index:
			if len(centroid) > 0:
				x_sum = 0
				y_sum = 0
				for index in centroid:
					x_sum += data_points[index][0]
					y_sum += data_points[index][1]
				x_avg = x_sum / len(centroid)
				y_avg = y_sum / len(centroid)
				mu.append([x_avg, y_avg])
				count += 1
		return mu

	# This function carries out the iterations, calling both the above functions repetetively and the comparing the
	# current index list with the previous one, which basically means that it is trying to look if there is any
	# difference in the assigned clusters from the previous one. As soon as it sees that two consecutive assignments
	# were
	# identical, it breaks out of the loop.
	def kMeans(data_points, mu):
		prev_indices = [0]
		current_indices = [2]
		count = 0
		while prev_indices != current_indices:
			prev_indices = current_indices
			current_indices = assignCentroids(data_points, mu)
			mu = getNewCentroids(data_points, current_indices)
			print('Iteration = ', count, ' and Centroid are = ', mu)
			count += 1
		return mu

	return kMeans(my_data_points, mu)


# data_points = A Numpy array of 150x2 shape
# data_points_t = Transpose of data_points array and has a 2x150 shape
# r_ic = Weights matrix of shape 150x3
def GaussianMixtureModel(gmm_data_points, k):
	gmm_data_points_t = np.transpose(gmm_data_points)

	# I have written a lot of helper functions inside the main function. Each function has its own separate
	# description.
	# This function simply initializes the weights matrix to random values according to the given dimensions.
	def initializeWeights(n, k):
		weight_matrix = np.random.rand(n, k)
		for i in range(len(weight_matrix)):
			sum = np.sum(weight_matrix[i])
			for j in range(len(weight_matrix[i])):
				weight_matrix[i][j] = weight_matrix[i][j] / sum
		return weight_matrix

	# This function computes the mean of the normal distribution according the formula taught by professor.
	def compute_mu(data_points, weights_matrix, k):
		mu = np.zeros((k, 2))
		for c in range(k):
			x_sum = 0
			y_sum = 0
			r_ic_sum = 0
			for i in range(len(weights_matrix)):
				x_sum = x_sum + data_points[i, 0] * weights_matrix[i, c]
				y_sum = y_sum + data_points[i, 1] * weights_matrix[i, c]
				r_ic_sum = r_ic_sum + weights_matrix[i, c]
			mu_x = x_sum / r_ic_sum
			mu_y = y_sum / r_ic_sum
			mu[c, 0] = mu_x
			mu[c, 1] = mu_y
		return mu

	# This function computes the dxd covariance matrices and then stores them in an array named 'covariance' and
	# returns it in the end. The interim steps include:
	# d_sum = ((Xi - MUi) x (Xi - MUi).transpose) * r_ic
	# sigma = d_sum / sum of R_ic for a given c
	def compute_covariance(data_points_t, mu_t, weights_matrix):
		covariance = np.array([np.zeros((len(data_points_t), len(data_points_t)))] * len(mu_t[0]))

		for c in range(len(mu_t[0])):
			d_sum = np.zeros((len(data_points_t), len(data_points_t)))
			r_ic_sum = 0
			for i in range(len(weights_matrix)):
				x_mu_difference = data_points_t[:, i] - mu_t[:, c]
				x_mu_difference = x_mu_difference.reshape(len(x_mu_difference), 1)
				outer_product = np.matmul(x_mu_difference, np.transpose(x_mu_difference))
				d_sum = d_sum + weights_matrix[i, c] * outer_product
				r_ic_sum = r_ic_sum + weights_matrix[i, c]

			sigma = d_sum / r_ic_sum
			covariance[c] = np.copy(sigma)

		return covariance

	# This function is quite simple, it simply computes sum of R_ic for a given c.
	def compute_pi_c(weights):
		pi_matrix = np.zeros((1, len(weights[0])))
		for c in range(len(pi_matrix[0])):
			pi_matrix[0, c] = np.sum(weights[:, c])
		return pi_matrix

	# This function is basically the Expectation calculation step. Here we are computing the weights matrix. For
	# weights matrix we need:
	# {Pi_c * N(Xi; MU_c, Sigma_c)} for given c / Numerator for all C's
	def compute_r_ic(pi, data_points_t, mu_t, covariance, weights):

		# This function computes the gaussian function for a given Xi. It is computed using that long equation.
		def compute_gaussian_distribution(my_mu, my_X, my_sigma):
			my_mu = my_mu.reshape((2, 1))
			my_X = my_X.reshape((2, 1))
			first_matrix = np.transpose(my_X - my_mu)
			second_matrix = inv(my_sigma)
			third_matrix = my_X - my_mu
			exponential = math.exp(-0.5 * multi_dot([first_matrix, second_matrix, third_matrix]))
			coefficient = 1 / (math.sqrt(det(my_sigma)) * (2 * math.pi) ** (len(my_X) / 2))
			return coefficient * exponential

		for c in range(len(pi[0])):
			for i in range(len(data_points_t[0])):
				gaussian = compute_gaussian_distribution(mu_t[:, c], data_points_t[:, i], covariance[c])
				numerator = pi[:, c] * gaussian
				denominator = 0

				# The prime values indicate that we are computing these functions for all the values of C.
				for c_prime in range(len(pi[0])):
					gaussian_prime = compute_gaussian_distribution(mu_t[:, c_prime], data_points_t[:, i], covariance[
						c_prime])
					pi_prime = pi[:, c_prime]
					denominator = denominator + gaussian_prime * pi_prime
				weights[i, c] = numerator / denominator
		return weights

	def percent_change(prev, current):
		difference_matrix = prev - current
		change_matrix = np.divide(difference_matrix, prev)
		max_change = np.amax(change_matrix) * 100
		return max_change

	# We have yet to figure out the convergence criterion for our program. For now I am simply hardcoding the number
	# of iterations I want to run the program for. There is something called log-likelihood function which people use
	# as convergence criterion, we will need to look that up.
	r_ic = initializeWeights(len(gmm_data_points), k)
	mu = np.zeros((len(gmm_data_points[0]), k))
	r_ic_prev = np.copy(r_ic)
	change = 100
	while change > 1.0:#1.0:
		mu = compute_mu(gmm_data_points, r_ic, k)
		mu_t_main = np.transpose(mu)
		covariance_matrix = compute_covariance(gmm_data_points_t, mu_t_main, r_ic)
		pi_c = compute_pi_c(r_ic)
		r_ic = compute_r_ic(pi_c, gmm_data_points_t, mu_t_main, covariance_matrix, r_ic)
		change = percent_change(r_ic_prev, r_ic)
		r_ic_prev = np.copy(r_ic)
		print('Current Max Percentage Change = ', change)
	return mu, covariance_matrix, pi_c



# File Reading. Found this amazing short method to read CSV files.
def readFile():
	with open('clusters.txt', 'r') as in_file:
		my_data_points = [list(map(float, rec)) for rec in csv.reader(in_file, delimiter=',')]
		np_data_points = loadtxt('clusters.txt', delimiter=',')
	# print(np_data_points)
	return my_data_points, np_data_points


# Creating a list of Random centroids.
def initializeMu(k):
	mu = []
	for i in range(k):
		temp=random.uniform(-2.0, 6.0)
		temp_mu = [temp, temp]
		mu.append(temp_mu)
	return mu


def plotResults(data_points, centroids, title,subplt):
	data1 = np.array(data_points)
	data2 = np.array(centroids)
	x1, y1 = data1.T
	x2, y2 = data2.T
	plt.title(title)
	plt.scatter(x1, y1)
	plt.scatter(x2, y2, color='red')
	#plt.show()
	return


if __name__ == '__main__':
	K = 3
	list_dataPoints, np_data_points = readFile()
	main_mu = initializeMu(K)
	k_means_centroids = KMeansAlgorithm(list_dataPoints, main_mu)
	print("Centroid from K-means:  " , k_means_centroids)

	gaussian_mu, gmm_covariance, amplitude = GaussianMixtureModel(np_data_points, K)


	print('Results of K-Means Algorithm:')
	print('The mean of 1st Centroid = (', k_means_centroids[0][0], ', ', k_means_centroids[0][1], ')')
	print('The mean of 2nd Centroid = (', k_means_centroids[1][0], ', ', k_means_centroids[1][1], ')')
	print('The mean of 3rd Centroid = (', k_means_centroids[2][0], ', ', k_means_centroids[2][1], ')')
	print('*****************************************************************************************')
	print('Gaussian Mixture Model Results: ')
	print('    a) 1st Gaussian Distribution Parameters:')
	print('         Mean:      ', gaussian_mu[0])
	print('         Covariance:', gmm_covariance[0, 0])
	print('                    ', gmm_covariance[0, 1])
	print('         Amplitude: ', amplitude[0, 0])
	print(' ')
	print('    b) 2nd Gaussian Distribution Parameters:')
	print('         Mean:      ', gaussian_mu[1])
	print('         Covariance:', gmm_covariance[1, 0])
	print('                    ', gmm_covariance[1, 1])
	print('         Amplitude: ', amplitude[0, 1])
	print(' ')
	print('    c) 3rd Gaussian Distribution Parameters:')
	print('         Mean:      ', gaussian_mu[2])
	print('         Covariance:', gmm_covariance[2, 0])
	print('                    ', gmm_covariance[2, 1])
	print('         Amplitude: ', amplitude[0, 2])

	#fig, axs = plt.subplots(2)
	plt.subplot(2, 1, 1)
	plotResults(list_dataPoints, k_means_centroids, 'K-Means Algorithm',0)
	plt.subplot(2, 1, 2)
	plotResults(list_dataPoints, gaussian_mu, 'Gaussian Mixture Model',1)
	plt.show()
	

# K-Means
# [[-0.9606529070232559, -0.6522184128604652],
# [5.738495346032257, 5.164838081193549],
# [3.2888485605151514, 1.9326883657575762]]

# GMM 500 iterations
# [[ 3.83452406  4.34898852]
#  [-0.99598414 -0.64023537]
#  [ 4.63136256  2.70620569]]

# [[ 3.8353404   4.34550992]
#  [ 4.63247798  2.70562814]
#  [-0.99599427 -0.64015777]]

# GMM 1000 iterations
# [[-0.84519835 -1.12264847]
#  [-1.34481956  1.38023483]
#  [ 4.45404462  3.35430896]]

# [[-0.84519835 -1.12264847]
#  [ 4.45404462  3.35430896]
#  [-1.34481956  1.38023483]]
