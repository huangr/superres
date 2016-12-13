import os
import numpy as np 
import scipy 
from scipy import ndimage 
from scipy import misc 
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import scipy.misc
from skimage.measure import block_reduce
from PIL import Image
import skimage.io
from pylab import *
import itertools
import array
import math
from calc_params import *

file_path = "/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/grayscale/alpaca"

# Get the initial center square of an image
def get_original_center(file_path, m, n, b):
	original_location = os.getcwd()

	# Generate the constants
	y = []

	# Generate the y values
	if(os.path.isdir(file_path)):
		os.chdir(file_path)
		print file_path
		files = os.listdir(file_path)
		index = 0

		for filename in files:
			if filename.endswith('.png'):
				img = misc.imread(filename, True)

				dimensions = img.shape
				center_coord = (dimensions[0]/2, dimensions[1]/2)
				for i in range(-m/2, m/2):
					for j in range(-m/2, m/2):
						y.append((img[center_coord[0] + i][(center_coord[1] + j)])/255.0-0.5)
				break
	else:
		print file_path, 'is not a directory.'
		return None
	return y

def img_print(img, row, col):
	# 2D-fy
	twod_img = [[None for i in range(row)] for j in range(col)]
	for i in range(len(img)):
		row_num = i / col
		col_num = i % col
		twod_img[row_num][col_num] = img[i]

	plt.imshow(twod_img, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
	plt.savefig('test.png')
	plt.show()

def normalize(img_array):
	total = max(img_array) - min(img_array)
	new_array = [None for i in range(len(img_array))]
	for i in range(len(img_array)):
		new_array[i] = (img_array[i] - min(img_array)) / total * 255 
	return new_array

def normalize_2d(img_array):
	mx = max([max(img_array[i]) for i in range(len(img_array))])
	mi = min([min(img_array[i]) for i in range(len(img_array))])
	new_array = [[None for j in range(len(img_array[i]))] for i in range(len(img_array))]
	for i in range(len(img_array)):
		for j in range(len(img_array[i])):
			new_array[i][j] = (img_array[i][j] - mi) / total * 255
	return new_array

def get_start_indices(row, col, m):
	start_indices = []
	for j in range(col / m):
		for i in range(row / m):
			start_indices.append([i, j])
	return start_indices

def construct_flat(vec):
	overall = []
	for i in range(len(vec)):
		overall.extend(vec[i])
	return overall

def construct_2d(flat_vec, dim):
	overall = [[None for i in range(dim)] for j in range(dim)]
	for i in range(len(flat_vec)):
		overall[i / dim][i % dim] = dim[i]
	return overall

def get_y(file_path, start_ind, m, k):
	y = []
	print start_ind
	if(os.path.isdir(file_path)):
		os.chdir(file_path)
		print file_path
		files = os.listdir(file_path)
		index = 0

		for filename in files:
			if filename.endswith('.png'):
				y.append([])
				img = misc.imread(filename, True)
				dimensions = img.shape
				for i in range(start_ind[0] * m, start_ind[0] * m + m):
					for j in range(start_ind[1] * m, start_ind[1] * m + m):
						y[index].append((img[i][j])/255.0-0.5)

				index = index + 1
				if index == k:
					break
	return y

def build_high_res(file_path, m, n, K, gamma, theta, shift, beta):
	# Get the dimensions
	if(os.path.isdir(file_path)):
		os.chdir(file_path)
		print file_path
		files = os.listdir(file_path)
		index = 0

		for filename in files:
			if filename.endswith('.png'):
				img = misc.imread(filename, True)

				dimensions = img.shape
				break

	# # Get start indices
	# start_indices = get_start_indices(dimensions[0], dimensions[1], m)
	# # Get the high resolution 
	# high_res_img = [[0 for i in range(dimensions[1]/ m * n)] for j in range(dimensions[0] / m * n)]

	# for ind in start_indices: 
	# 	# M x M from top left of start_indices
	# 	y_input = get_y(file_path, ind, m, K)
	# 	# N x N generate!
	# 	mu = compute_mu(m, n, K, gamma, theta, shift, beta, y_input)

	# 	for i in range(n):
	# 		for j in range(n):
	# 			high_res_img[ind[0]*n + i][ind[1]*n + j] = mu[i*n+j]

	high_res_img = normalize_2d(high_res_img)

	plt.imshow(high_res_img, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
	plt.savefig('test.png')
	plt.show()

# img_print(get_original_center(file_path, 3, 5, 400), 3, 3)
gamma = 1.1
theta = [1.53, 1.61, 1.58, 1.55, 1.53, 1.53, 1.59, 1.55, 1.5, 1.65, 1.63, 1.61, 1.59, 1.46, 1.58]
theta = [t - math.pi / 2 for t in theta]
s = [[12.2, -1.13], [15.3, -0.54], [14.55, -0.32], [15.48, -0.81], [10.61, -1.3], [9.81, -1.18], [13.83, -0.7], [13.77, -0.62], [10.04, -1.32], [13.02, -1.03], [13.96, -0.86], [12.22, -0.82], [13.53, -0.75], [6.48, -0.72], [-2.23, 0.73]]

build_high_res(file_path, 4, 8, 15, gamma, theta, s, 400)

