import os
import numpy as np 
import scipy 
from scipy import ndimage 
from scipy import misc 
import matplotlib.pyplot as plt
import scipy.misc

color_directory_list = os.listdir("/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/color/low_results/original")

appendices = [
	"_bicubic",
	"_glasner",
	"_LR",
	"_nearest",
	"_ScSR",
	"_SRCNN"
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

# Downsamples an image to the given dimensions of that image.
# Returns the downsampled image as a numpy image.
def downsample(image_path, row_dim, col_dim):
	img = misc.face()
	misc.imsave(image_path, img)
	print img.shape

# Downsample all files given the directory to scaled factor, 
# and saves it to a given directory 
def downsample_directory(parent_dir, target_dir, scaled_factor):
	print "NotImplemented"

# Downsamples all files given the curre
