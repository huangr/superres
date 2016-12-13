import os
import numpy as np 
import scipy 
from scipy import ndimage 
from scipy import misc 
import matplotlib.pyplot as plt
import scipy.misc
from skimage.measure import block_reduce
from PIL import Image
import skimage.io
from pylab import *
import itertools
import math

"""
Here, we look at the following parameters and initialize them based off of images.

data {
  int <lower=0> K; # number of images
  int <lower=0> M; # height/width of each image
  int <lower=0> N; # number of pixels in superimage
  vector[M] y[K]; # the input images
  matrix[N, N] Zx; # covariance matrix (in terms of A and r)
  vector[2] v_bar;
  vector[2] vi[N];
  vector[2] vj[M];
  real beta; # noise 
}

We will consider an inner 15x15 center square, and we will extrapolate this by 2
to form a 35x35 final image, and perform the mu transformations over all y.

"""

file_path = "/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/grayscale/alpaca"

# Defines the relevant variables given a file_path to a directory of
# low resolution images 
# Requirement: Images in file_path must be in '.png' form
def define_variables(file_path, m, n, b):
	(K, M, N, y, Zx, v_bar, vi, vj, beta) = (None, None, None, None, None, None, None, None, None)
	img_to_index = {}
	original_location = os.getcwd()

	# Generate the constants
	M = m**2
	N = n**2
	beta = b
	y = []
	Zx = [[None for i in range(N)] for j in range(N)]
	vi = [None for i in range(N)]
	vj = [None for i in range(M)]

	# Generate the y values
	if(os.path.isdir(file_path)):
		os.chdir(file_path)
		print file_path
		files = os.listdir(file_path)
		index = 0

		for filename in files:
			if filename.endswith('.png'):
				y.append([])
				img_to_index[filename] = index
				img = misc.imread(filename, True)

				dimensions = img.shape
				center_coord = (dimensions[0]/2, dimensions[1]/2)
				for i in range(-m/2, m/2):
					for j in range(-m/2, m/2):
						y[index].append((img[center_coord[0] + i][(center_coord[1] + j)])/255.0-0.5)

				index = index + 1
				if index == 15:
					break
	else:
		print file_path, 'is not a directory.'
		return

	print "y-values have been generated."

	# Generate the value for the number of images that we use.
	K = len(y)
	print "We have used", K, "images."

	upscale = float(n)/float(m)
	begin = float(1+upscale)/2.
	for i in range(m):
		for j in range(m):
			vj[i*m + j] = [i * upscale + begin, j * upscale + begin]
	for i in range(N):
		vi[i] = [(i/int(math.sqrt(N)))+1, (i % int(math.sqrt(N)))+1]
	v_bar = (math.sqrt(N)/2, math.sqrt(N)/2)
	for i in range(N):
		for j in range(N):
			Zx[i][j] = 0.04*math.exp((-np.linalg.norm(np.array(vi[i]) - np.array(vi[j]))**2))
	print "All variables have been generated! :)"


	print np.linalg.det(np.array(Zx))

	data = {
		'K': K,
		'M': M,
		'N': N,
		'y': y,
		'Zx': Zx, 
		'v_bar': v_bar,
		'vi': vi, 
		'vj': vj,
		'beta': beta
	}

	inputs = {
	 	'gamma': 4.0,
	 	'theta': [1.0]*K,
	 	's': [[1.0, 1.0] for i in range(K)]
	}

	os.chdir(original_location)

	return data, inputs
