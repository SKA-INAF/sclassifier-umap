#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## MODULES
from sclassifier_umap import __version__, __date__
from sclassifier_umap import logger
from sclassifier_umap.data_provider import DataProvider


#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-filelists','--filelists', dest='filelists', required=True, nargs='+', type=str, default=[], help='List of image filelists') 
	parser.add_argument('-catalog_file','--catalog_file', dest='catalog_file', required=False, type=str, default='', help='Caesar source catalog ascii file') 

	# - Data process options
	parser.add_argument('--crop_img', dest='crop_img', action='store_true',help='Crop input images')	
	parser.set_defaults(crop_img=False)	
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=51, action='store',help='Image crop width in pixels (default=51)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=51, action='store',help='Image crop height in pixels (default=51)')	

	parser.add_argument('--normalize_img', dest='normalize_img', action='store_true',help='Normalize input images in range [0,1]')	
	parser.set_defaults(normalize_inputs=False)
	parser.add_argument('-normdatamin', '--normdatamin', dest='normdatamin', required=False, type=float, default=-0.0100, action='store',help='Normalization min used to scale data in [0,1] range (default=-100 mJy/beam)')	
	parser.add_argument('-normdatamax', '--normdatamax', dest='normdatamax', required=False, type=float, default=10, action='store',help='Normalization max used to scale data in [0,1] range (default=10 Jy/beam)')
	
	parser.add_argument('--normalize_img_to_first_chan', dest='normalize_img_to_first_chan', action='store_true',help='Normalize input images to first channel')	
	parser.set_defaults(normalize_img_to_first_chan=False)
	parser.add_argument('--normalize_img_to_chanmax', dest='normalize_img_to_chanmax', action='store_true',help='Normalize input images to channel maximum')	
	parser.set_defaults(normalize_img_to_chanmax=False)

	parser.add_argument('--apply_weights', dest='apply_weights', action='store_true',help='Apply weights to input image channels')	
	parser.set_defaults(apply_weights=False)	
	parser.add_argument('-img_weights','--img_weights', dest='img_weights', required=False, nargs='+', type=float, default=[], help='List of image weights (must have same size of input filelists)') 


	args = parser.parse_args()	

	return args



##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Input filelist
	filelists= args.filelists
	catalog_file= args.catalog_file
	print(filelists)

	# - Data process options	
	crop_img= args.crop_img
	nx= args.nx
	ny= args.ny

	normalize_img= args.normalize_img
	normdatamin= args.normdatamin
	normdatamax= args.normdatamax
	normalize_img_to_chanmax= args.normalize_img_to_chanmax
	normalize_img_to_first_chan= args.normalize_img_to_first_chan
	
	apply_weights= args.apply_weights
	img_weights= args.img_weights

	#===========================
	#==   CHECK ARGS
	#===========================
	if apply_weights and len(img_weights)!=len(filelists):
		logger.error("Input image weights has size different from input filelists!")
		return 1
	
	#===========================
	#==   READ DATA
	#===========================
	# - Create data provider
	dp= DataProvider(filelists=filelists)

	# - Set options
	dp.set_catalog_filename(catalog_file)
	dp.enable_inputs_normalization(normalize_img)
	dp.set_input_data_norm_range(normdatamin,normdatamax)
	dp.enable_inputs_normalization_to_first_channel(normalize_img_to_first_chan)
	dp.enable_inputs_normalization_to_chanmax(normalize_img_to_chanmax)
	dp.enable_img_crop(crop_img)
	dp.set_img_crop_size(nx,ny)
	dp.enable_img_weights(apply_weights)
	dp.set_img_weights(img_weights)


	# - Read data	
	logger.info("Running data provider to read image data ...")
	status= dp.read_data()
	if status<0:
		logger.error("Failed to read input image data!")
		return 1
	
	# - Compute image similarity data
	logger.info("Compute image similarity data ...")
	status= dp.compute_similarity_data()
	if status<0:
		logger.error("Failed to compute image similarity data!")
		return 1

	# - Draw data (first image)
	#logger.info("Drawing the first image as test...")
	#dp.save_data(0,save_to_file=False,outfile='source_plot.png')

	# - Draw ssim data (first image)
	logger.info("Drawing the first image ssim as test...")
	dp.save_ssim_data(save_to_file=False)
	
	# - Draw ssim grad data (first image)
	#logger.info("Drawing the first image ssimgrad as test...")
	#dp.save_ssimgrad_data(0,save_to_file=False,outfile='source_ssimgrad_plot.png')
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

