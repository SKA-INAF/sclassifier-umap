#!/usr/bin/env python

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
##from ctypes import *

## ASTRO
from scipy import ndimage
##import pyfits
from astropy.io import fits
from astropy.units import Quantity
from astropy.modeling.parameters import Parameter
from astropy.modeling.core import Fittable2DModel
from astropy.modeling.models import Box2D, Gaussian2D, Ring2D, Ellipse2D, TrapezoidDisk2D, Disk2D, AiryDisk2D, Sersic2D
#from photutils.datasets import make_noise_image
from astropy import wcs

try:
	## ROOT
	import ROOT
	from ROOT import gSystem, TFile, TTree, gROOT, AddressOf

	## CAESAR
	gSystem.Load('libCaesar')
	from ROOT import Caesar
except:
	print("WARN: Cannot load ROOT & Caesar modules (not a problem if you are not going to use them)...")

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## Graphics modules
import matplotlib.pyplot as plt

## PACKAGE MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)
##################################################


##############################
##     CLASS DEFINITIONS
##############################
class FeatureExtractor(object):
	""" Class to extract features from stack of images

			Arguments:
				-
	"""

	def __init__(self,data_provider):
		""" Return a FeatureExtractor object """

		# - Check input data
		if data_provider is None:
			logger.error('Null data given as input!')
			raise IOError(errmsg)

		self.dp= data_provider

		self.input_data= self.dp.get_data()
		self.data_labels= self.dp.get_data_labels()
		self.source_names= self.dp.get_source_names()
		self.imgshape= self.input_data.shape
		self.nsamples= self.imgshape[0]
		self.nx= self.imgshape[1]
		self.ny= self.imgshape[0]
		self.nchannels= self.imgshape[3]

		# - Source masks
		self.chanId= 0
		self.seedThr= 5
		self.mergeThr= 2.5
		self.minNPix= 5
		self.outerLayerSize= 21 # pixels
		self.applyDistThr= True
		self.maxDist= 5
		self.inner_masks= []
		self.outer_masks= []
		self.use_mask= True
		self.mask_computed= False

		# - SSIM images
		self.input_data_ssim= None
		self.input_data_ssimgrad= None

		# - Features
		self.Sratio_inner_data= None
		self.Sratio_outer_data= None
		self.SSIM_inner_data= None
		self.SSIM_outer_data= None
		
	#################################
	##     SETTERS
	#################################
	def set_seed_thr(self,thr):
		""" Set seed threshold for source extraction """
		self.seedThr= thr

	def set_merge_thr(self,thr):
		""" Set merge threshold for source extraction """
		self.mergeThr= thr

	def set_max_source_dist_from_center(self,d):
		""" Set max source distance from center """
		self.maxDist= d

	def set_outer_layer_size(self,n):
		""" Set outer layer size in pixel """
		self.outerLayerSize= n


	#################################
	##     COMPUTE SRATIO FEATURES
	#################################
	def __extract_flux_ratio_features(self):
		""" Extract flux ratio features from data using masks """

		Sratio_inner_list= []
		Sratio_outer_list= []	

		# - Loop over data cube and extract features
		for i in range(self.nsamples):
			
			# - Compute integrated fluxes over inner/outer masks (if available)
			Sratio_inner_stack= []
			Sratio_outer_stack= []
			S_inner_list= []
			S_outer_list= []
			useMask= False
			innerMaskPixels= None
			outerMaskPixels= None
			if self.use_mask and self.mask_computed:
				n_tot= self.inner_masks[i].size
				n_inner= np.count_nonzero(self.inner_masks[i])
				n_outer= np.count_nonzero(self.outer_masks[i])
				logger.info("Source %s: n_tot=%d, n_inner=%d, n_outer=%d" % (self.source_names[i],n_tot,n_inner,n_outer))

				if n_inner<=0 or n_outer<=0:
					logger.warn("Inner and/or outer mask was not properly computed for source %s, not using mask ..." % self.source_names[i])
				else:
					useMask= True
					innerMaskPixels= self.inner_masks[i]==1
					outerMaskPixels= self.outer_masks[i]==1

			for j in range(0,self.nchannels):				
				imgdata= self.input_data[i,:,:,j]
				S_min= np.min(imgdata) 
				S_max= np.max(imgdata) 
				S_inner= 0
				S_outer= 0
				if useMask:
					S_inner= imgdata[innerMaskPixels].sum()
					S_outer= imgdata[outerMaskPixels].sum()
					#print("== S_inner pixels ==")
					#print imgdata[innerMaskPixels]
					#print("== S_outer pixels ==")
					#print imgdata[outerMaskPixels]
				else:
					S_inner= np.sum(imgdata)
					S_outer= S_inner
	
				S_inner_list.append(S_inner)
				S_outer_list.append(S_outer)

				logger.info("Source %s (CHAN %d): S min/max=%f/%f, S_inner=%f, S_outer=%f" % (self.source_names[i],j+1,S_min,S_max,S_inner,S_outer))
			

			# - Compute flux ratios
			for j in range(0,self.nchannels):
				S_inner_j= S_inner_list[j]
				S_outer_j= S_outer_list[j]
				for k in range(j+1,self.nchannels):
					S_inner_k= S_inner_list[k]
					S_outer_k= S_outer_list[k]
					Sratio_inner= S_inner_j/S_inner_k
					Sratio_outer= S_outer_j/S_outer_k
					Sratio_inner_stack.append(Sratio_inner)
					Sratio_outer_stack.append(Sratio_outer)
					logger.info("Source %s (CHAN %d-%d): S_inner=(%f,%f), S_outer(%f,%f)" % (self.source_names[i],j+1,k+1,S_inner_j,S_inner_k,S_outer_j,S_outer_k))
			
			Sratio_inner_cube= np.dstack(Sratio_inner_stack)
			Sratio_outer_cube= np.dstack(Sratio_outer_stack)
			Sratio_inner_list.append(Sratio_inner_cube)	
			Sratio_outer_list.append(Sratio_outer_cube)	

		# - Create array
		self.Sratio_inner_data= np.array(Sratio_inner_list)
		self.Sratio_inner_data= self.Sratio_inner_data.astype('float32')

		self.Sratio_outer_data= np.array(Sratio_outer_list)
		self.Sratio_outer_data= self.Sratio_outer_data.astype('float32')

		logger.info("Shape of Sratio data")
		print(np.shape(self.Sratio_inner_data))

		return 0

	#################################
	##     COMPUTE SSIM FEATURES
	#################################
	def __extract_ssim_features(self):
		""" Extract ssim features from data using masks """

		
		# - Check if SIMG were computed
		# ...
		# ...
		print("== SSIM SHAPE ==")
		print self.input_data_ssim.shape
		nimgs= self.input_data_ssim.shape[3]

		# - Loop over data cube and extract features
		SSIM_inner_list= []
		SSIM_outer_list= []	

		for i in range(self.nsamples):
			SSIM_inner_stack= []
			SSIM_outer_stack= []

			useMask= False
			innerMaskPixels= None
			outerMaskPixels= None
			if self.use_mask and self.mask_computed:
				n_tot= self.inner_masks[i].size
				n_inner= np.count_nonzero(self.inner_masks[i])
				n_outer= np.count_nonzero(self.outer_masks[i])
				logger.info("Source %s: n_tot=%d, n_inner=%d, n_outer=%d" % (self.source_names[i],n_tot,n_inner,n_outer))

				if n_inner<=0 or n_outer<=0:
					logger.warn("Inner and/or outer mask was not properly computed for source %s, not using mask ..." % self.source_names[i])
				else:
					useMask= True
					innerMaskPixels= self.inner_masks[i]==1
					outerMaskPixels= self.outer_masks[i]==1

			# - Compute SSIMG averages
			for j in range(0,nimgs):
				ssimimgdata= self.input_data_ssim[i,:,:,j]
			
				SSIM_mean_inner= 0
				SSIM_mean_outer= 0		
				if useMask:
					SSIM_mean_inner= ssimimgdata[innerMaskPixels].mean()
					SSIM_mean_outer= ssimimgdata[outerMaskPixels].mean()
				else:
					SSIM_mean_inner= np.mean(ssimimgdata)
					SSIM_mean_outer= SSIM_mean_inner
	
				SSIM_inner_stack.append(SSIM_mean_inner)
				SSIM_outer_stack.append(SSIM_mean_outer)
	
				logger.info("Source %s (CHAN %d): SSIM_mean_inner=%f, SSIM_mean_outer=%f" % (self.source_names[i],j+1,SSIM_mean_inner,SSIM_mean_outer))
			
		
			SSIM_inner_cube= np.dstack(SSIM_inner_stack)
			SSIM_outer_cube= np.dstack(SSIM_outer_stack)
			SSIM_inner_list.append(SSIM_inner_cube)	
			SSIM_outer_list.append(SSIM_outer_cube)	

		# - Create array
		self.SSIM_inner_data= np.array(SSIM_inner_list)
		self.SSIM_inner_data= self.SSIM_inner_data.astype('float32')

		self.SSIM_outer_data= np.array(SSIM_outer_list)
		self.SSIM_outer_data= self.SSIM_outer_data.astype('float32')

		logger.info("Shape of SSIM average data")
		print(np.shape(self.SSIM_inner_data))

		return 0


	#################################
	##     COMPUTE FEATURES
	#################################
	def extract_features(self):
		""" Extract features from data using masks """

		# - Check if data entry exists
		if self.input_data is None:
			logger.error("No input data present (hint: read data first!")
			return -1


		#============================
		#==   FLUX RATIOS
		#============================
		logger.info('Computing flux ratio features ...')
		status= self.__extract_flux_ratio_features()
		if status<0:
			logger.error('Failed to compute flux ratio features ...')
			return -1

		#===============================
		#==   IMG SIMILARITY FEATURES
		#===============================
		logger.info('Computing average similarity features ...')
		status= self.__extract_ssim_features()
		if status<0:
			logger.error('Failed to compute average similarity features ...')
			return -1

		return 0

		

	#################################
	##     COMPUTE SOURCE MASKS
	#################################
	def compute_source_masks(self,chan_id=0):
		""" Compute source masks over selected channel """
		
		# - Check if data entry exists
		if self.input_data is None:
			logger.error("No input data present (hint: read data first!")
			return -1

		imgshape= self.input_data.shape
		nsamples= imgshape[0]
		nchannels= imgshape[3]
		if self.chanId<0 or self.chanId>=nchannels:
			logger.error("Invalid channel id given!")
			return -1

		# - Loop over images
		for i in range(nsamples):
			imgdata= self.input_data[i,:,:,self.chanId]

			# - Compute inner/outer source mask images
			logger.info("Computing inner/output mask for source %s ..." % (self.source_names[i]))
			res= Utils.compute_source_inner_outer_masks(
				imgdata,
				self.seedThr,self.mergeThr,self.minNPix,
				self.outerLayerSize,
				self.applyDistThr,self.maxDist
			)
			if not res:
				logger.error('Failed to extract source inner/outer masks!')
				return -1

			smask_inner= res[0]
			smask_outer= res[1]
			nx= smask_inner.GetNx()
			ny= smask_inner.GetNy()
		
			# - Convert Caesar img to numpy array data
			smask_inner_pixdata= smask_inner.GetPixels()
			smask_outer_pixdata= smask_outer.GetPixels()
			smask_inner_data= np.asarray(smask_inner_pixdata)
			smask_outer_data= np.asarray(smask_outer_pixdata)

			smask_inner_data_reshaped= smask_inner_data.reshape(ny,nx)
			smask_outer_data_reshaped= smask_outer_data.reshape(ny,nx)

			# - Append masks to global data
			self.inner_masks.append(smask_inner_data_reshaped)
			self.outer_masks.append(smask_outer_data_reshaped)


		self.mask_computed= True

		return 0

	#################################
	##     COMPUTE SSIM IMAGES
	#################################
	def compute_similarity_data(self):
		""" Compute similarity between maps """

		# - Check if data entry exists
		if self.input_data is None:
			logger.error("No input data present (hint: read data first!")
			return -1

		# - Loop over images
		imgshape= self.input_data.shape
		nsamples= imgshape[0]
		nchannels= imgshape[3] 
		simg_list= []
		sgradimg_list= []

		for i in range(nsamples):
			img_stack= []
			gradimg_stack= []
			for j in range(0,nchannels):
				for k in range(j+1,nchannels):
					(mssim, grad_img, sim_img)= Utils.compute_img_similarity(self.input_data[i,:,:,j],self.input_data[i,:,:,k])
					img_stack.append(sim_img)
					gradimg_stack.append(grad_img)
			simg_cube= np.dstack(img_stack)
			sgradimg_cube= np.dstack(gradimg_stack)
			simg_list.append(simg_cube)
			sgradimg_list.append(sgradimg_cube)

		self.input_data_ssim= np.array(simg_list)
		self.input_data_ssim= self.input_data_ssim.astype('float32')
		self.input_data_ssimgrad= np.array(sgradimg_list)
		self.input_data_ssimgrad= self.input_data_ssimgrad.astype('float32')

		logger.info("Shape of input ssim data")
		print(np.shape(self.input_data_ssim))

		return 0




