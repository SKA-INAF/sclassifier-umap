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
from sclassifier_umap.feature_extractor import FeatureExtractor

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

	# - Feature extractor options
	parser.add_argument('-seedThr', '--seedThr', dest='seedThr', required=False, type=float, default=5, action='store',help='Seed significance threshold used to create source mask (default=5)')	
	parser.add_argument('-mergeThr', '--mergeThr', dest='mergeThr', required=False, type=float, default=2.5, action='store',help='Merge significance threshold used to create source mask (default=2.5)')	
	parser.add_argument('-sourceMaxDistFromCenter', '--sourceMaxDistFromCenter', dest='sourceMaxDistFromCenter', required=False, type=float, default=5, action='store',help='Maximum distance in pixels of detected soure from center to be considered in the mask (default=5)')	
	parser.add_argument('-outerLayerSize', '--outerLayerSize', dest='outerLayerSize', required=False, type=float, default=21, action='store',help='Outer layer size in pixels used to create the outer mask (default=21)')	
	parser.add_argument('-ssimWindowSize', '--ssimWindowSize', dest='ssimWindowSize', required=False, type=int, default=9, action='store',help='Window size in pixels used for ssim index computation (default=9)')	
	
	

	# - Output options
	# ...

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
	normalize_img_to_first_chan= args.normalize_img_to_first_chan
	normalize_img_to_chanmax= args.normalize_img_to_chanmax
	apply_weights= args.apply_weights
	img_weights= args.img_weights

	# - Feature extractor options
	seedThr= args.seedThr
	mergeThr= args.mergeThr
	sourceMaxDistFromCenter= args.sourceMaxDistFromCenter
	outerLayerSize= args.outerLayerSize
	ssim_window_size= args.ssimWindowSize

	# - Output file
	# ...	

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
	#dp= DataProvider(filelists=filelists)
	dp= DataProvider()

	# - Set options
	dp.set_filelists(filelists)
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

	#===========================
	#==   EXTRACT FEATURES
	#===========================
	fextractor= FeatureExtractor(dp)
	fextractor.set_seed_thr(seedThr)
	fextractor.set_merge_thr(mergeThr)
	fextractor.set_max_source_dist_from_center(sourceMaxDistFromCenter)
	fextractor.set_outer_layer_size(outerLayerSize)
	fextractor.set_ssim_window_size(ssim_window_size)

	logger.info("Computing similarity data ...")
	status= fextractor.compute_similarity_data()
	if status<0:
		logger.error("Failed to compute similarity images!")
		return 1

	logger.info("Computing source masks ...")
	status= fextractor.compute_source_masks()
	if status<0:
		logger.error("Failed to compute source masks!")
		return 1

	logger.info("Running feature extractor ...")
	status= fextractor.extract_features()
	if status<0:
		logger.error("Failed to extract features!")
		return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

