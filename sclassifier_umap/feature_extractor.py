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
		self.ssim_window_size= 9
		self.input_data_ssim= None
		self.input_data_ssimgrad= None

		# - Features
		self.Sratio_inner_data= None
		self.Sratio_outer_data= None
		self.SSIM_inner_data= None
		self.SSIM_outer_data= None
	
		# - Output file
		self.outfile= 'features.dat'
		
	#################################
	##     SETTERS
	#################################
	def set_ssim_window_size(self,w):
		""" Set window size for ssim index computation"""
		self.ssim_window_size= w

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

		#===============================
		#==   SAVE DATA
		#===============================
		logger.info('Saving features to file %s ...' % self.outfile)
		self.__save_feature_data(self.outfile)

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
			#res= Utils.compute_source_inner_outer_masks(
			res= self.__compute_source_inner_outer_masks(
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

				# Normalize image to max 
				imgdata_j= self.input_data[i,:,:,j]
				max_j= np.max(imgdata_j)
				imgdata_j= np.divide(imgdata_j,max_j)

				for k in range(j+1,nchannels):
					# Normalize image to max 
					imgdata_k= self.input_data[i,:,:,k]
					max_k= np.max(imgdata_k)
					imgdata_k= np.divide(imgdata_k,max_k)

					# Compute similarity index
					#(mssim, grad_img, sim_img)= Utils.compute_img_similarity(imgdata_j,imgdata_k,self.ssim_window_size)
					(mssim, grad_img, sim_img)= self.__compute_img_similarity(imgdata_j,imgdata_k,self.ssim_window_size)
					
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


	#################################
	##     SAVE FEATURE DATA
	#################################
	def __save_feature_data(self,outfile):
		""" Save feature data to file """

		logger.info("Saving feature data to file ...")
		N= self.Sratio_inner_data.shape[0]
		print("Sratio shape=",self.Sratio_inner_data.shape)
		print("Data N=",N)
		print("SSIIM shape=",self.SSIM_inner_data.shape)

		obj_ids= []
		obj_subids= []
		obj_confirmeds= []

		for i in range(N):
			source_name= self.source_names[i]
			has_labels= source_name in self.data_labels	
			obj_id= 0
			obj_subid= 0
			obj_confirmed= 0

			if has_labels:
				obj_info= self.data_labels[source_name]
				obj_id= obj_info['id']
				obj_subid= obj_info['subid']
				obj_confirmed= obj_info['confirmed']
				
			obj_ids.append(obj_id)
			obj_subids.append(obj_subid)	
			obj_confirmeds.append(obj_confirmed)

			#print('Source name=%s, has_labels=%d, id=%d, subid=%s' % (source_name,has_labels,obj_id,obj_subid))

		# - Merge encoded data
		snames= np.array(self.source_names).reshape(N,1)
		objids= np.array(obj_ids).reshape(N,1)
		objsubids= np.array(obj_subids).reshape(N,1)
		objconfirmeds= np.array(obj_confirmeds).reshape(N,1)
		sratio_inner= self.Sratio_inner_data.reshape(self.Sratio_inner_data.shape[0],self.Sratio_inner_data.shape[3])	
		sratio_outer= self.Sratio_outer_data.reshape(self.Sratio_outer_data.shape[0],self.Sratio_outer_data.shape[3])	
		ssim_inner= self.SSIM_inner_data.reshape(self.SSIM_inner_data.shape[0],self.SSIM_inner_data.shape[3])	
		ssim_outer= self.SSIM_outer_data.reshape(self.SSIM_outer_data.shape[0],self.SSIM_outer_data.shape[3])	
	
		out_data= np.concatenate(
			(snames,sratio_inner,sratio_outer,ssim_inner,ssim_outer,objids,objsubids,objconfirmeds),
			axis=1
		)

		#head= '# sname Sratio_inner Sratio_outer SSIM_inner SSIM_outer id subid confirmed'
		head= '# ' 
		head+= 'sname '	
		for k in range(sratio_inner.shape[1]):
			head+= 'Sratio_inner_' + str(k) + ' '
		for k in range(sratio_outer.shape[1]):
			head+= 'Sratio_outer_' + str(k) + ' '
		for k in range(ssim_inner.shape[1]):
			head+= 'SSIM_inner_' + str(k) + ' '
		for k in range(ssim_outer.shape[1]):
			head+= 'SSIM_outer_' + str(k) + ' '
		head+= 'id '
		head+= 'subid '
		head+= 'confirmed'
		
		Utils.write_ascii(out_data,outfile,head)	

	#################################
	##     COMPUTE FEATURE IMAGE
	#################################
	def __compute_caesar_img(self,data,compute_stats=False):
		""" Return a Caesar image from 2D numpy array data """

		# - Check input data
		if data is None:
			logger.error("Null data given!")
			return None
		imgsize= data.shape
		nx= imgsize[1]
		ny= imgsize[0]
		if nx<=0 or ny<=0:
			logger.error("Invalid image size detected in numpy data given!")	
			return None

		# - Create and fill image
		img= Caesar.Image(nx,ny)

		for ix in range(nx):
			for iy in range(ny):
				w= data[iy,ix]
				img.FillPixel(ix,iy,w)		

		# - Compute stats
		if compute_stats:
			robustStats=True
			img.ComputeStats(robustStats)

		return img

	#################################
	##     GET SIGNIFICANCE MAP
	#################################
	def __get_significance_map(self,imgdata,bkgEstimator=Caesar.eMedianBkg):
		""" Compute significance map """

		# - Convert numpy input data to Caesar image
		computeStats= True
		img= self.__compute_caesar_img(imgdata,computeStats)
		if img is None:
			logger.error("Failed to convert input data to Caesar image!")
			return ()
		
		# - Compute global background
		localBkg= False
		bkgData= img.ComputeBkg(bkgEstimator,localBkg)
		if bkgData is None:
			logger.error("Failed to compute image background!")
			return ()

		# - Compute significance map
		zmap= img.GetSignificanceMap(bkgData,localBkg)
		if zmap is None:
			logger.error("Failed to compute image significance map!")
			return ()

		return (img,zmap)


	#################################
	##     FIND SOURCES
	#################################
	def __find_sources(self,imgdata,seedThr=5,mergeThr=2.5,minPixels=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image """

		# - Convert numpy input data to Caesar image
		computeStats= True
		img= self.__compute_caesar_img(imgdata,computeStats)
		if img is None:
			logger.error("Failed to convert input data to Caesar image!")
			return ()
		
		# - Compute global background
		localBkg= False
		bkgEstimator= Caesar.eMedianBkg
		bkgData= img.ComputeBkg(bkgEstimator,localBkg)
		if bkgData is None:
			logger.error("Failed to compute image background!")
			return ()

		# - Compute significance map
		zmap= img.GetSignificanceMap(bkgData,localBkg)
		if zmap is None:
			logger.error("Failed to compute image significance map!")
			return ()

		# - Extract sources
		findNested= False
		sources= ROOT.std.vector("Caesar::Source*")()
		status= img.FindCompactSource(sources,zmap,bkgData,seedThr,mergeThr,minPixels,findNested)
		if status<0:
			logger.error("Failed to extract sources!")
			return ()

		logger.info('#%d sources found in map ...' % sources.size())

		# - Selecting sources by distance from center
		nx= img.GetNx()
		ny= img.GetNy()
		xc= (int)(nx/2.)
		yc= (int)(ny/2.)
		sources_sel= ROOT.std.vector("Caesar::Source*")()
		if applyDistThr:
			logger.info('Selecting sources closer to image center by dist %f ...' % maxDist)
			for i in range(sources.size()):
				x= sources[i].X0
				y= sources[i].Y0
				d= np.sqrt( (x-xc)*(x-xc) + (y-yc)*(y-yc ) )
				if d<=maxDist:
					sources_sel.push_back(sources[i])
				else:
					logger.info('Skipping source found at (%f,%f) with distance d=%f from image center (%f,%f) ...' % (x,y,d,xc,yc))
		else:
			sources_sel= sources
			
		logger.info('#%d selected sources in map ...' % sources_sel.size())

		return (img, zmap, sources_sel)

	#################################
	##     FIND SOURCE MASK
	#################################
	def __compute_source_mask(self,imgdata,seedThr=5,mergeThr=2.5,minPixels=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image and get a binary mask """

		# - Extracting sources
		logger.info('Extracting sources from map ...')
		res= self.__find_sources(imgdata,seedThr,mergeThr,minPixels,applyDistThr,maxDist)	
		if not res:
			logger.error('Failed to extract sources from map (return tuple is empty!)')
			return None

		img= res[0]
		zmap= res[1]
		sources= res[2]
		if img is None:
			logger.error('Failed to get Caesar map from data!')
			return None

		logger.info('#%d sources found in map ...' % sources.size())

		# - Return empty mask if no sources found
		if sources.empty():
			logger.info('Returning empty source mask ...')
			smask= img.GetCloned("")
			smask.Reset()
			return smask

		# - Compute source mask
		logger.info('Computing source mask ...')
		isBinary= True
		invert= False
		smask= img.GetSourceMask(sources,isBinary,invert)
		
		return smask

	######################################
	##     FIND SOURCE MASK INNER/OUTER
	######################################
	def __compute_source_inner_outer_masks(self,imgdata,seedThr=5,mergeThr=2.5,minPixels=5,outer_npix=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image and get a inner and outer binary mask """

		# - Compute first source mask
		smask= self.__compute_source_mask(imgdata,seedThr,mergeThr,minPixels,applyDistThr,maxDist)
	
		# - Get dilated mask
		niters= 1
		smask_dil= smask.GetMorphDilatedImage(outer_npix,niters,False)

		# - Substract smask from smask_dil to get crown around source	
		smask_outer= smask_dil.GetCloned("")
		smask_outer.Add(smask,-1)

		return (smask,smask_outer)

	######################################
	##     FIND SOURCE Z MASK SHELLS
	######################################
	def compute_source_z_masks(self,imgdata,seedThr=5,mergeThrs=[4,3,2.5,2,1.5],minPixels=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image and get a binary masks corresponding to significance ranges """

		# - Compute significance map
		logger.info('Compute significance map ...')
		res= self.__get_significance_map(imgdata)
		if not res:
			logger.error('Failed to compute significance map (return tuple is empty!)')
			return None

		img= res[0]
		zmap= res[1]
		if zmap is None:
			logger.error('Failed to get Caesar significance map from data!')
			return None

		# - Loop over merge thr and find masks
		findNested= False
		isBinary= True
		invert= False
		smasks= []
		for mergeThr in mergeThrs:
			logger.info('Finding sources using seedThr=%f, mergeThr=%f ...' % (seedThr,mergeThr))
			sources= ROOT.std.vector("Caesar::Source*")()
			status= img.FindCompactSource(sources,zmap,None,seedThr,mergeThr,minPixels,findNested)
			if status<0:
				logger.error("Failed to extract sources!")
				return None

			smask= img.GetSourceMask(sources,isBinary,invert)
			smasks.append(smask)

		# - Substract smask from previous to get z shells
		nthrs= len(smasks)
		smasks_shell= []
		for i in range(nthrs):
			if i==0:
				smasks_shell.append(smasks[i])
				continue
			smask_shell= smasks[i].GetCloned("")
			smask_shell.Add(smasks[i-1],-1)
			smasks_shell.append(smask_shell)

		return smasks_shell
	
