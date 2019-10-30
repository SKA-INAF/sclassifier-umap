#!/usr/bin/env python

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import string
import logging
import numpy as np

## ASTRO MODULES
from astropy.io import fits
from astropy.io import ascii 

## SCIKIT LEARN
from skimage.measure import compare_ssim

## GRAPHICS MODULES
import matplotlib.pyplot as plt


try:
	## ROOT
	import ROOT
	from ROOT import gSystem, TFile, TTree, gROOT, AddressOf

	## CAESAR
	gSystem.Load('libCaesar')
	from ROOT import Caesar
	from ROOT.Caesar import Image

except:
	print("WARN: Cannot load ROOT & Caesar modules (not a problem if you are not going to use them)...")


logger = logging.getLogger(__name__)


###########################
##     CLASS DEFINITIONS
###########################
class Utils(object):
	""" Class collecting utility methods

			Attributes:
				None
	"""

	def __init__(self):
		""" Return a Utils object """
		#self.logger = logging.getLogger(__name__)
		#_logger = logging.getLogger(__name__)

	@classmethod
	def has_patterns_in_string(cls,s,patterns):
		""" Return true if patterns are found in string """
		if not patterns:		
			return False

		found= False
		for pattern in patterns:
			found= pattern in s
			if found:
				break

		return found

	@classmethod
	def write_ascii(cls,data,filename,header=''):
		""" Write data to ascii file """

		# - Skip if data is empty
		if data.size<=0:
			#cls._logger.warn("Empty data given, no file will be written!")
			logger.warn("Empty data given, no file will be written!")
			return

		# - Open file and write header
		fout = open(filename, 'wt')
		if header:
			fout.write(header)
			fout.write('\n')	
			fout.flush()	
		
		# - Write data to file
		nrows= data.shape[0]
		ncols= data.shape[1]
		for i in range(nrows):
			fields= '  '.join(map(str, data[i,:]))
			fout.write(fields)
			fout.write('\n')	
			fout.flush()	

		fout.close();

	@classmethod
	def read_ascii(cls,filename,skip_patterns=[]):
		""" Read an ascii file line by line """
	
		try:
			f = open(filename, 'r')
		except IOError:
			errmsg= 'Could not read file: ' + filename
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			raise IOError(errmsg)

		fields= []
		for line in f:
			line = line.strip()
			line_fields = line.split()
			if not line_fields:
				continue

			# Skip pattern
			skipline= cls.has_patterns_in_string(line_fields[0],skip_patterns)
			if skipline:
				continue 		

			fields.append(line_fields)

		f.close()	

		return fields

	@classmethod
	def read_ascii_table(cls,filename,row_start=0,delimiter='|'):
		""" Read an ascii table file line by line """

		table= ascii.read(filename,data_start=row_start, delimiter=delimiter)
		return table

	@classmethod
	def write_fits(cls,data,filename):
		""" Read data to FITS image """

		hdu= fits.PrimaryHDU(data)
		hdul= fits.HDUList([hdu])
		hdul.writeto(filename,overwrite=True)

	@classmethod
	def read_fits(cls,filename):
		""" Read FITS image and return data """

		# - Open file
		try:
			hdu= fits.open(filename,memmap=False)
		except Exception as ex:
			errmsg= 'Cannot read image file: ' + filename
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			raise IOError(errmsg)

		# - Read data
		data= hdu[0].data
		data_size= np.shape(data)
		nchan= len(data.shape)
		if nchan==4:
			output_data= data[0,0,:,:]
		elif nchan==2:
			output_data= data	
		else:
			errmsg= 'Invalid/unsupported number of channels found in file ' + filename + ' (nchan=' + str(nchan) + ')!'
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			hdu.close()
			raise IOError(errmsg)

		# - Read metadata
		header= hdu[0].header

		# - Close file
		hdu.close()

		return output_data, header

	
	@classmethod
	def crop_img(cls,data,x0,y0,dx,dy):
		""" Extract sub image of size (dx,dy) around pixel (x0,y0) """

		#- Extract crop data
		xmin= int(x0-dx/2)
		xmax= int(x0+dx/2)
		ymin= int(y0-dy/2)
		ymax= int(y0+dy/2)		
		crop_data= data[ymin:ymax+1,xmin:xmax+1]
	
		#- Replace NAN with zeros and inf with large numbers
		np.nan_to_num(crop_data,False)

		return crop_data

	@classmethod
	def draw_histo(cls,data,nbins=100,logscale=False):
		""" Draw input array histogram """

		# - Do nothing if data is empty
		if data.ndim<=0:
			return

		# - Flatten array 
		x= data.flatten()

		# - Set histogram from data
		hist, bins = np.histogram(x, bins=nbins)
		width = 0.7 * (bins[1] - bins[0])
		center = (bins[:-1] + bins[1:]) / 2

		# - Draw plots
		plt.bar(center, hist, align='center', width=width)
		if logscale:
			plt.yscale('log')

		plt.show()

	@classmethod
	def compute_img_similarity(cls,img1,img2,window_size=3):
		""" Compute similarity index between two images """
	
		mssim, grad_img, sim_img= compare_ssim(
			img1,img2,
			win_size=window_size,
			gradient=True,
			multichannel=False,
			full=True
		)

		return mssim, grad_img, sim_img

	@classmethod
	def compute_caesar_img(cls,data,compute_stats=False):
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
	@classmethod 
	def get_significance_map(cls,imgdata,bkgEstimator=Caesar.eMedianBkg):
		""" Compute significance map """

		# - Convert numpy input data to Caesar image
		computeStats= True
		img= cls.compute_caesar_img(imgdata,computeStats)
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
	@classmethod
	def find_sources(cls,imgdata,seedThr=5,mergeThr=2.5,minPixels=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image """

		# - Convert numpy input data to Caesar image
		computeStats= True
		img= cls.compute_caesar_img(imgdata,computeStats)
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
	@classmethod
	def compute_source_mask(cls,imgdata,seedThr=5,mergeThr=2.5,minPixels=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image and get a binary mask """

		# - Extracting sources
		logger.info('Extracting sources from map ...')
		res= cls.find_sources(imgdata,seedThr,mergeThr,minPixels,applyDistThr,maxDist)	
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
	@classmethod
	def compute_source_inner_outer_masks(cls,imgdata,seedThr=5,mergeThr=2.5,minPixels=5,outer_npix=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image and get a inner and outer binary mask """

		# - Compute first source mask
		smask= cls.compute_source_mask(imgdata,seedThr,mergeThr,minPixels,applyDistThr,maxDist)
	
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
	@classmethod
	def compute_source_z_masks(cls,imgdata,seedThr=5,mergeThrs=[4,3,2.5,2,1.5],minPixels=5,applyDistThr=False,maxDist=10):
		""" Extract sources from image and get a binary masks corresponding to significance ranges """

		# - Compute significance map
		logger.info('Compute significance map ...')
		res= cls.get_significance_map(imgdata)
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
	
