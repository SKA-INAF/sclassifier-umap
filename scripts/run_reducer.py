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
from sclassifier_umap.classifier import UMAPClassifier

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
	parser.add_argument('-filelists','--filelists', dest='filelists', required=True, nargs='+', type=str, default=[], help='List of image filelists or feature files') 
	parser.add_argument('-catalog_file','--catalog_file', dest='catalog_file', required=False, type=str, default='', help='Caesar source catalog ascii file') 
	parser.add_argument('--isfeaturedata', dest='isfeaturedata', action='store_true',help='Regard filelists as feature files (default=false)')	
	parser.set_defaults(isfeaturedata=False)	

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

	parser.add_argument('--apply_logtransform', dest='apply_logtransform', action='store_true',help='Apply log transform to feature data')	
	parser.set_defaults(apply_logtransform=False)

	parser.add_argument('--apply_weights', dest='apply_weights', action='store_true',help='Apply weights to input image channels')	
	parser.set_defaults(apply_weights=False)	
	parser.add_argument('-img_weights','--img_weights', dest='img_weights', required=False, nargs='+', type=float, default=[], help='List of image weights (must have same size of input filelists)') 

	# - Classifier options
	parser.add_argument('-ndim', '--ndim', dest='ndim', required=False, type=int, default=2, action='store',help='Encoded data dim (default=2)')
	parser.add_argument('-mindist', '--mindist', dest='mindist', required=False, type=float, default=0.1, action='store',help='Min dist UMAP par (default=0.1)')
	parser.add_argument('-nneighbors', '--nneighbors', dest='nneighbors', required=False, type=int, default=15, action='store',help='N neighbors UMAP par (default=15)')
	
	
	# - Output options
	parser.add_argument('-outfile_unsupervised', '--outfile_unsupervised', dest='outfile_unsupervised', required=False, type=str, default='encoded_data_unsupervised.dat', action='store',help='Name of encoded data output file (default=encoded_data_unsupervised.dat)')
	parser.add_argument('-outfile_supervised', '--outfile_supervised', dest='outfile_supervised', required=False, type=str, default='encoded_data_supervised.dat', action='store',help='Name of output file with encoded data produced using supervised method (default=encoded_data_supervised.dat)')
	parser.add_argument('-outfile_preclassified', '--outfile_preclassified', dest='outfile_preclassified', required=False, type=str, default='encoded_data_preclassified.dat', action='store',help='Name of output file with encoded data produced from pre-classified data (default=encoded_data_preclassified.dat)')

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
	isfeaturedata= args.isfeaturedata
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
	apply_logtransform= args.apply_logtransform
	apply_weights= args.apply_weights
	img_weights= args.img_weights

	# - Classifier options
	ndim= args.ndim
	mindist= args.mindist
	nneighbors= args.nneighbors
	
	# - Output file
	outfile_unsupervised= args.outfile_unsupervised
	outfile_supervised= args.outfile_supervised
	outfile_preclassified= args.outfile_preclassified
		

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
	#dp.set_filelists(filelists)
	#dp.set_catalog_filename(catalog_file)
	dp.enable_inputs_normalization(normalize_img)
	dp.set_input_data_norm_range(normdatamin,normdatamax)
	dp.enable_inputs_normalization_to_first_channel(normalize_img_to_first_chan)
	dp.enable_inputs_normalization_to_chanmax(normalize_img_to_chanmax)
	dp.apply_log_transform(apply_logtransform)
	dp.enable_img_crop(crop_img)
	dp.set_img_crop_size(nx,ny)
	dp.enable_img_weights(apply_weights)
	dp.set_img_weights(img_weights)


	# - Read feature data
	if isfeaturedata:
		logger.info("Running data provider to read feature data ...")
		status= dp.read_feature_data(filelists[0])
		if status<0:
			logger.error("Failed to read input image data!")
			return 1
	
	else:
		logger.info("Running data provider to read image data ...")
		status= dp.read_data(filelists,catalog_file)
		if status<0:
			logger.error("Failed to read input image data!")
			return 1
	

	#===========================
	#==   UMAP REDUCER
	#===========================
	logger.info("Running UMAP classifier reducer ...")
	classifier= UMAPClassifier(dp)

	classifier.set_encoded_data_unsupervised_outfile(outfile_unsupervised)
	classifier.set_encoded_data_supervised_outfile(outfile_supervised)
	classifier.set_encoded_data_preclassified_outfile(outfile_preclassified)
	classifier.set_encoded_data_dim(ndim)
	classifier.set_min_dist(mindist)
	classifier.set_n_neighbors(nneighbors)

	status= classifier.train()
	if status<0:
		logger.error("UMAP training failed!")
		return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

