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
	print len(img)
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


# img_print(get_original_center(file_path, 3, 5, 400), 3, 3)
mu = [1.36, -0.51, -2.84, -4.94, -6.18, 2.95, 1.02, -1.58, -4.08, -5.72, 4.6,
	2.82, 0.2, -2.49, -4.44, 5.82, 4.37, 1.97, -0.66, -2.74, 6.24, 5.2, 3.21, 0.88, -1.1]
img_print(normalize(mu), 5, 5)
