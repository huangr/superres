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
Compute the vi and vj defined in Bishop for m and n.
Conditions: m < n.
Returns: vi and vj as m**2 and n**2 coordinate values.
"""
def compute_vi_vj(m, n):
	M = m**2
	N = n**2
	vi = [None for i in range(N)]
	vj = [None for i in range(M)]

	upscale = float(n)/float(m)
	begin = float(1+upscale)/2.

	for i in range(m):
		for j in range(m):
			vj[i*m + j] = [i * upscale + begin, j * upscale + begin]

	for i in range(N):
		vi[i] = [(i/int(math.sqrt(N)))+1, (i % int(math.sqrt(N)))+1]

	return (vi, vj)

"""
Compute Zx, covariance matrix, of vi, with dimension N x N.
"""
def compute_zx(N, vi, a = 0.04, r = 1):

	Zx = [[None for i in range(N)] for j in range(N)]

	for i in range(N):
		for j in range(N):
			Zx[i][j] = a*math.exp((-np.linalg.norm(np.array(vi[i]) - np.array(vi[j]))**2) / r**2)

	return Zx

"""
u is defined as R(vj-v) + v + s.
Note that the Bishop parameters include the rotational parameter theta and shift s.
k is the number of test images to use
"""
def compute_u(theta, shift, K, vi, n):
	vbar = [float(1+n)/2., float(1+n)/2.]

	# Generate the rotation matrix
	
	u = [[None for i in range(len(vi))] for j in range(K)]

	for k in range(K):
		for j in range(len(vi)):
			c, s = np.cos(theta[k]), np.sin(theta[k])
			R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
			u[k][j] = np.add(np.add(np.transpose(np.dot(R, np.subtract(np.transpose(np.matrix(vi[j])), np.matrix(vbar)))), np.matrix(vbar)), np.matrix(shift[k]))

	return u

"""
Compute W, the transformation matrix
"""
def compute_w(m, n, K, gamma, theta, shift):
	M = m**2
	N = n**2
	(vi, vj) = compute_vi_vj(m, n)
	u = compute_u(theta, shift, K, vj, n)
	W_tilde = [[[None for i in range(N)] for j in range(M)] for k in range(K)]
	
	for k in range(K):
		for j in range(M):
			for i in range(N):
				W_tilde[k][j][i] = math.exp(-np.linalg.norm(vi[i] - u[k][j])**2/gamma**2)

	W_sum = [[0 for j in range(M)] for k in range(K)]

	for k in range(K):
		for j in range(M):
			for i in range(N):
				W_sum[k][j] += W_tilde[k][j][i]

	W = [[[None for i in range(N)] for j in range(M)] for k in range(K)]

	for k in range(K):
		for j in range(M):
			for i in range(N):
				W[k][j][i] = float(W_tilde[k][j][i])/float(W_sum[k][j])

	return W

"""
Compute posterior covariance sigma
"""
def compute_sigma(m, n, K, gamma, theta, shift, beta):
	W = compute_w(m, n, K, gamma, theta, shift)
	(vi, vj) = compute_vi_vj(m, n)
	Zx = compute_zx(n*n, vi)

	# Turn into numpy matrices
	Zx = np.matrix(Zx)
	W = np.matrix(Zx)

	total_matrix = np.matrix([[0 for i in range(n*n)] for j in range(n*n)])

	for k in range(K):
		total_matrix = np.add(total_matrix, np.dot(np.transpose(np.matrix(W[k])), np.matrix(W[k])))

	sigma_inverse = np.add(np.linalg.inv(Zx), beta * total_matrix)

	return np.linalg.inv(sigma_inverse)

"""
Compute posterior mu
"""
def compute_mu(m, n, K, gamma, theta, shift, beta, y_sheet):
	sigma = compute_sigma(m, n, K, gamma, theta, shift, beta)
	W = compute_w(m, n, K, gamma, theta, shift)

	total_matrix = np.matrix([[0 for i in range(K)] for j in range(n*n)])
	for k in range(K):
		total_matrix = np.add(total_matrix, np.dot(np.transpose(np.matrix(W[k])), np.transpose(y_sheet)))
	mu = beta * np.dot(sigma, total_matrix)

	return mu



