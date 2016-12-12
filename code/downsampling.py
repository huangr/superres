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

color_directory_list = os.listdir("/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/color/low_results/original")

appendices = [
	"_bicubic",
	"_glasner",
	"_LR",
	"_nearest",
	"_ScSR",
	"_SRCNN"
]

folders = [
	"bicubic",
	"glasner",
	"LR",
	"nearest",
	"scsr",
	"srcnn"
]

# Removes an appendix in file name for all files in directory.
def rename_files(directory, rm_appendix):
	original_location = os.getcwd()
	if(os.path.isdir(directory)):
		files = os.listdir(directory)
		os.chdir(directory)
		for filename in files:
			if(filename != filename.replace(rm_appendix, '')):
				print 'Renamed', filename, 'to', filename.replace(rm_appendix, '')
				os.rename(filename, filename.replace(rm_appendix, ''))
	os.chdir(original_location)
	print "Finished renaming files."

# Downsamples all files inside directory of a folder ('original') to
# dimensions of images in a given directory.
def downsample_metric(folder_dir, metric_dir):
	original_location = os.getcwd()
	target_im_dim = {}

	# Get images from the metric directory
	if(os.path.isdir(metric_dir)):
		# print 'cd', metric_dir
		os.chdir(metric_dir)
		files = os.listdir(metric_dir)
		for file in files:
			if(file.endswith(".png")):
				im = Image.open(file)
				width, height = im.size 
				target_im_dim[file] = (width, height)
				print file, 'width:', width, 'height:', height
		print '##### Successfully completed image dimension extraction! #####'
		os.chdir(original_location)
	else:
		print 'The second argument is not a valid directory.'
		return

	if(os.path.isdir(folder_dir)):
		folders = os.listdir(folder_dir)
		# print 'cd', folder_dir
		os.chdir(folder_dir)
		for folder in folders:
			# print 'cd', folder
			if(os.path.isdir(folder)):
				cur_dir = os.getcwd()
				files = os.listdir(folder)
				os.chdir(folder)
				for filename in files:
					if(filename.endswith(".png")):
						with open(filename, 'r+b') as f:
							with Image.open(f) as image:
								cover = image.resize(target_im_dim[filename])
								cover.save(filename)
				os.chdir(cur_dir)
				print '##### Successfully completed downsampling for', folder, '#####'
		os.chdir(original_location)
	else:
		print 'The first argument is not a valid directory.'
		return
	os.chdir(original_location)

# Downsamples all files inside directory of a folder to a target grayscale directory right outside
def grayscale(folder_dir, target_dir):
	original_location = os.getcwd()

	if(os.path.isdir(folder_dir)):
		files = os.listdir(folder_dir)
		os.chdir(folder_dir)
		for filename in files:
			if(filename.endswith(".png")):
				image_file = Image.open(filename)
				image_file = image_file.convert('LA')
				image_file.save(target_dir + "/" + filename)

# downsample_metric("/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/color/low_results/correct_dim", "/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/color/low_results/correct_dim/LR")

filepath = "/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/color/"
for f in folders:
	grayscale(filepath + "low_results/correct_dim/" + f, filepath + "grayscale/" + f)

