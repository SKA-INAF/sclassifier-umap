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

## ASTROPY MODULES 
from astropy.io import ascii 

## ADDON ML MODULES
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

## PACKAGE MODULES
from .utils import Utils

## MATPLOTLIB
import matplotlib.pyplot as plt

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)


##############################
##     CLASS DEFINITIONS
##############################
class DataProvider(object):
	""" Class to read train data from disk and provide to network

			Arguments:
				- datadir: Data root directory where to search for images
				- fileext: Image file extension to be searched
	"""
	
	
	#def __init__(self,filelists=[]):
	def __init__(self):
		""" Return a DataProvider object """

		# - Input data
		#self.filelists= filelists
		self.filelists= []	
		self.catalog_file= ''
		self.nx= 0
		self.ny= 0	
		self.nimgs= 0
		self.crop_img= False
		self.nx_crop= 0
		self.ny_crop= 0
		
		# - Input data normalization
		self.input_data= None
		self.input_data_labels= {}
		self.source_names= []
		self.component_ids= []
		self.normalize_to_first_chan= False	
		self.normalize_to_chanmax= False
		self.apply_weights= False
		self.img_weights= []
		self.normalize_inputs= True
		self.minmax_scaler= None
		self.normmin= 0
		self.normmax= 1
		#self.normmin= 0.001
		#self.normmax= 10
		self.nBadPixThr= 0.6
		self.badPixReplaceVal= 0 #-999

		# - SSIM images
		self.input_data_ssim= None
		self.input_data_ssimgrad= None

		

	#################################
	##     SETTERS/GETTERS
	#################################
	def set_filelists(self,filelist):
		""" Set filelists """
		self.filelists= filelist

	def set_catalog_filename(self,filename):
		""" Set name of source catalog file """
		self.catalog_file= filename	

	def enable_inputs_normalization_to_first_channel(self,choice):
		""" Turn on/off inputs normalization to first channel"""
		self.normalize_to_first_chan= choice

	def enable_inputs_normalization_to_chanmax(self,choice):
		""" Turn on/off inputs normalization to channel maximum"""
		self.normalize_to_chanmax= choice

	def enable_inputs_normalization(self,choice):
		""" Turn on/off inputs normalization """
		self.normalize_inputs= choice

	def set_input_data_norm_range(self,datamin,datamax):
		""" Set input data normalization range """
		self.normmin= datamin
		self.normmax= datamax

	def get_img_size(self):
		""" Return the train image size """
		return self.nx, self.ny

	def enable_img_crop(self,choice):
		""" Turn on/off input image crop """
		self.crop_img= choice

	def set_img_crop_size(self,nx,ny):
		""" Set input data normalization range """
		self.nx_crop= nx
		self.ny_crop= ny

	def enable_img_weights(self,choice):
		""" Turn on/off apply of input image weights """
		self.apply_weights= choice

	def set_img_weights(self,w):
		""" Set input image weights """
		self.img_weights= w

	def get_data(self):
		""" Return read data """
		return self.input_data	

	def get_source_names(self):
		""" Return read source names """
		return self.source_names

	def get_component_ids(self):
		""" Return read source component ids """
		return self.component_ids

	def get_data_labels(self):
		""" Return source labels """
		return self.input_data_labels

	#################################
	##     READ FEATURE DATA
	#################################
	def read_feature_data(self,filename,row_start=0,delimiter=' '):	
		""" Read feature file and create dataset """
		
		# - Check input file
		if not filename:
			logger.error("Empty feature data filename given!")
			return -1			

		# - Read ascii table
		try:
			table= Utils.read_ascii_table(filename,row_start,delimiter)
		except IOError:
			errmsg= 'Cannot read file: ' + filename
			logger.error(errmsg)
			return -1

		nsamples= len(table)
		ncols= len(table.columns)
		colnames= table.colnames
		colnames_excluded= ['sname','id','subid','confirmed']
		colnames_feat= [i for i in colnames if not any([e for e in colnames_excluded if e in i])]

		# - Get subtable (only feature columns)
		table_feat= table[colnames_feat]
		ncols_feat= len(table.columns)

		print(table_feat)
		print(colnames_feat)

		# - Fill source dictionary
		rowIndex= 0
		self.source_names= []

		for data in table:
			source_full_name= data['sname']
			source_name= source_full_name.split('_fitcomp')[0]
			componentId= source_full_name.split('_fitcomp')[1]
			obj_id= data['id']
			obj_subid= data['subid']
			obj_confirmed= data['confirmed']
			obj_name= '' # Missing
			obj_info= {}
			obj_info['id']= obj_id
			obj_info['subid']= obj_subid
			obj_info['confirmed']= obj_confirmed
			obj_info['name']= obj_name
			self.source_names.append(source_full_name)
			self.input_data_labels[source_full_name]= obj_info

		# - Fill numpy array with feature data
		x= table_feat.as_array() # this returns a structured numpy array
		self.input_data= x.view((x.dtype[0], len(x.dtype.names))) # convert to standard numpy array
		self.input_data= self.input_data.astype('float32')

		logger.info("Shape of input data")
		print(np.shape(self.input_data))		

		# - Normalize feature data
		if self.normalize_inputs:
			logger.info("Input data (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
			
			if not self.minmax_scaler:
				self.minmax_scaler= preprocessing.MinMaxScaler(feature_range=(self.normmin,self.normmax))
				self.input_data= self.minmax_scaler.fit_transform(self.input_data) # store the scaler function for later usage

			#self.input_data= (self.input_data - self.normmin)/(self.normmax-self.normmin)
			logger.info("Input data (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
			

		return 0


	#############################
	##     READ INPUT DATA
	#############################
	def read_data(self,img_filelist='',catalog_filename='',):	
		""" Read image & source data and create dataset """

		#===========================
		#==   SET INPUT FILENAMES
		#===========================
		if catalog_filename:
			self.catalog_file= catalog_filename
		if img_filelist:
			self.filelists= img_filelist
			
		if not self.catalog_file:
			logger.error("Empty catalog filename given or set!")
			return -1
		
		if not self.filelists:
			logger.error("Empty source image filename list given or set!")
			return -1

		#===========================
		#==   READ SOURCE CATALOG
		#===========================	
		logger.info("Reading source catalog data ...")
		status= self.__read_source_catalog_data(self.catalog_file)
		if status<0:
			logger.warn("Failed to read source catalog data, no labels will be available")
			
		#===========================
		#==   READ IMAGE DATA
		#===========================	
		logger.info("Reading source image data ...")
		status= self.__read_source_image_data(self.filelists)
		if status<0:
			logger.error("Failed to read source image data!")
			return -1

		return 0


	#############################
	##     READ INPUT IMAGES
	#############################
	def __read_source_image_data(self,filelists=''):	
		""" Read data from disk """
			
		# - Check data filelists
		if not filelists:
			logger.error("Empty filelists given!")
			return -1

		nfilelists= len(filelists)
		
		# - Check weights size
		if self.apply_weights and len(self.img_weights)!=nfilelists:
			logger.error("Image weights size is different from number of channels given!")
			return -1
		
		# - Read filelists
		filelist_counter= 0
		imgfilenames= []

		for filelist in filelists:
			filelist_counter+= 1
			imgfilenames.append([])

			try:
				filenames= Utils.read_ascii(filelist,['#'])
			except IOError:
				errmsg= 'Cannot read file: ' + filelist
				logger.error(errmsg)
				return -1

			# - Check lists have the same number of files
			if filelist_counter==1:
				self.nimgs= len(filenames)
			else:
				if len(filenames)!=self.nimgs:
					logger.error("Given filelists have a different number of file entries (%s!=%s)!" % (len(filenames),self.nimgs))
					return -1

			
			# - Read image files in list
			for item in filenames:
				
				filename= item[0]
				#logger.info("Reading file %s ..." % filename) 

				imgfilenames[filelist_counter-1].append(filename)

		# - Reorder list of files by channel
		imgfilenames= map(list, zip(*imgfilenames))

		# - Loop over image files and read them
		imgcounter= 0
		imgcubecounter= 0
		input_data_list= []
		

		for i in range(len(imgfilenames)):
		
			imgdata_stack= []
			isGoodImage= True
			source_name_full= ''
			source_name= ''
			componentId= ''

			for j in range(len(imgfilenames[i])):
				imgcounter+= 1
				filename= imgfilenames[i][j]
				logger.info("Reading file %s ..." % filename) 
				data= None
				try:
					data, header= Utils.read_fits(filename)
				except Exception as ex:
					errmsg= 'Failed to read image data (err=' + str(ex) + ')'
					logger.warn(errmsg)
					isGoodImage= False
					break

				imgsize= np.shape(data)
				nx= imgsize[1]
				ny= imgsize[0]
				nchannels= len(imgfilenames[i])
				
				# Retrieve image name
				dir_names= os.path.dirname(filename).split('/')
				source_name_full= dir_names[len(dir_names)-1]
				tmp= source_name_full.split('_')
				source_name= tmp[0]
				fitcomp_name= tmp[1]
				p= fitcomp_name.find("fitcomp")
				componentId= fitcomp_name[p+7:len(fitcomp_name)]
				logger.info("Image no. %d (full_name=%s, name=%s, compid=%s, chan=%d) has size (%d,%d)" % (i+1,source_name_full,source_name,componentId,j+1,nx,ny) )	
		

				# - Extract crop img data
				data_crop= data
				if self.crop_img:
					if self.nx_crop<=0 or self.nx_crop>nx or self.ny_crop<=0 or self.ny_crop>ny:
						errmsg= 'Requested crop size is zero or exceeding image size!'
						logger.warn(errmsg)
						isGoodImage= False
						break

					if self.nx_crop!=nx and self.ny_crop!=ny:
						x0= np.ceil(nx/2.)
						y0= np.ceil(ny/2.)
						data_crop= Utils.crop_img(data,x0,y0,self.nx_crop,self.ny_crop)
						imgsize= np.shape(data_crop)
						nx= imgsize[1]
						ny= imgsize[0]

					logger.info("Cropped image no. %d (chan=%d) has size (%d,%d)" % (i+1,j+1,nx,ny) )	

				# - Check image size is equal for all files
				if imgcounter==1:
					self.nx= nx
					self.ny= ny	
				else:
					if (nx!=self.nx or ny!=self.ny):
						errmsg= 'Image no. ' + str(imgcounter) + ' has different size (nx=' + str(self.nx) + ',ny=' + str(self.ny) + ') wrt previous images!'
						logger.error(errmsg)
						return -1

				#	- Set image data as a tensor of size [Nsamples,Nx,Ny,Nchan] Nchan=1 and create stack
				data_crop.reshape(imgsize[0],imgsize[1],1)

				# - Check image value integrity
				npixels= data_crop.size
				npixels_nan= np.count_nonzero(np.isnan(data_crop)) 
				npixels_inf= np.count_nonzero(np.isinf(data_crop))
				badPixFraction= (npixels_nan+npixels_inf)/float(npixels)
				if badPixFraction>self.nBadPixThr:
					logger.warn("Cropped image no. %d (chan=%d) has too many bad pixels (badPixFract=%f), skip it" % (i+1,j+1,badPixFraction) )	
					isGoodImage= False
					break

				# - Append image channel data to stack
				imgdata_stack.append(data_crop)
				logger.info("Cropped image no. %d (chan=%d) : min/max=%f/%f" % (i+1,j+1,np.min(data_crop),np.max(data_crop)))
			

			# - Skip image if marked as bad
			if not isGoodImage:
				logger.warn("Skipping image no. %d as marked as bad..." % (i+1) )
				continue	

			# - Set source & component names
			self.source_names.append(source_name_full)
			#self.source_names.append(source_name)
			self.component_ids.append(componentId)
			imgcubecounter+= 1

			# - Apply weights to images
			if self.apply_weights:
				for index in range(0,len(imgdata_stack)):
					print('DEBUG: Chan %d weight=%f' % (index,self.img_weights[index]))
					imgdata_stack[index]*= self.img_weights[index]
					logger.info("Cropped image no. %d (name=%s, compid=%s, chan=%d) (AFTER WEIGHTS): min/max=%f/%f" % (i+1,self.source_names[imgcubecounter-1],self.component_ids[imgcubecounter-1],index+1,np.min(imgdata_stack[index]),np.max(imgdata_stack[index])))
			

			# - Normalize data to first channel?	
			if self.normalize_to_first_chan and len(imgdata_stack)>1:
				for index in range(0,len(imgdata_stack)):
					if index>0:	
						imgdata_stack[index]= np.divide(imgdata_stack[index],imgdata_stack[0])
					logger.info("Cropped image no. %d (chan=%d) (AFTER NORM): min/max=%f/%f" % (i+1,index+1,np.min(imgdata_stack[index]),np.max(imgdata_stack[index])))
			
				
			# - Replace NaNs & inf with safe value	
			badPixSafeVal= self.badPixReplaceVal
			if self.normalize_inputs:
				badPixSafeVal= self.normmin
				
			for index in range(0,len(imgdata_stack)):
				#np.nan_to_num(imgdata_stack[index])
				(imgdata_stack[index])[~np.isfinite( (imgdata_stack[index]) )]= badPixSafeVal
				logger.info("Cropped image no. %d (chan=%d) (AFTER NORM & WEIGHTS & SANITIZE): min/max=%f/%f" % (i+1,index+1,np.min(imgdata_stack[index]),np.max(imgdata_stack[index])))
			
			# - Normalize data to maximum among all channels
			if self.normalize_to_chanmax:
				chanmax_list= []
				for index in range(0,len(imgdata_stack)):
					chanmax= np.max(imgdata_stack[index])					
					chanmax_list.append(chanmax)		
				chanmax= max(chanmax_list)

				for index in range(0,len(imgdata_stack)):
					imgdata_stack[index]/= chanmax	
					logger.info("Cropped image no. %d (chan=%d) (AFTER WEIGHTS & SANITIZE & CHAN NORM): min/max=%f/%f" % (i+1,index+1,np.min(imgdata_stack[index]),np.max(imgdata_stack[index])))
			

			#	- Set image data as a tensor of size [Nsamples,Nx,Ny,Nchan]
			imgdata_cube= np.dstack(imgdata_stack)
			input_data_list.append(imgdata_cube)
			logger.info("Input data (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(imgdata_cube)),str(np.max(imgdata_cube))))
			
			
		#- Convert list to array
		self.input_data= np.array(input_data_list)
		self.input_data= self.input_data.astype('float32')

		logger.info("Shape of input data")
		nsamples= self.input_data.shape[0]
		print(np.shape(self.input_data))

		# - Normalize to [0,1]
		if self.normalize_inputs:
			logger.info("Input data (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
			self.input_data= (self.input_data - self.normmin)/(self.normmax-self.normmin)
			logger.info("Input data (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
			
		# - Flatten input data to 2D (Nsamples x (nx*ny*nchan))
		#self.input_data= self.input_data.reshape(nsamples,-1)

		# - Normalize to range
		#if self.normalize_inputs:
		#	logger.info("Input data (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
		#	if not self.minmax_scaler:
		#		self.minmax_scaler= preprocessing.MinMaxScaler(feature_range=(self.normmin,self.normmax))
		#		self.input_data= self.minmax_scaler.fit_transform(self.input_data) # store the scaler function for later usage
		#	logger.info("Input data (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
				
		return 0

	#################################
	##     READ SOURCE CATALOG DATA
	#################################
	def __read_source_catalog_data(self,filename,row_start=0,delimiter='|'):	
		""" Read source catalog data table """
		
		# - Read ascii table
		try:
			table= Utils.read_ascii_table(filename,row_start,delimiter)
		except IOError:
			errmsg= 'Cannot read file: ' + filename
			logger.error(errmsg)
			return -1

		print(table)

		# - Create source label dictionary
		rowIndex= 0
		for data in table:
			source_name= data['col1']
			componentId= data['col3']	
			source_full_name= str(source_name) + '_fitcomp' + str(componentId)
			obj_id= data['col57']
			obj_subid= data['col58']
			obj_confirmed= data['col59']
			obj_name= data['col61']
			obj_info= {}
			obj_info['id']= obj_id
			obj_info['subid']= obj_subid
			obj_info['confirmed']= obj_confirmed
			obj_info['name']= obj_name
			
			self.input_data_labels[source_full_name]= obj_info

		print('Source catalog dict')			
		print(self.input_data_labels)

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

	#################################
	##     SAVE IMAGE DATA
	#################################
	def save_data(self,data_index,save_to_file=False,outfile='source_plot.png'):
		""" Save input data read to images """
		
		# - Check if data entry exists
		if self.input_data is None:
			logger.error("No input data present (hint: read data first!")
			return -1

		imgshape= self.input_data.shape
		nsamples= imgshape[0]
		nchannels= imgshape[3] 

		if data_index>=nsamples:
			logger.error("No input data present with index %d " % data_index)
			return -1
	
		# - Find maximum & minimum data value
		data_min= np.min(self.input_data[data_index])
		data_max= np.max(self.input_data[data_index])

		
		# - Initialize plot		
		#plt.style.use("ggplot")
		
		fig = plt.figure(figsize=(20,20))
		#(fig, ax) = plt.subplots(nchannels,1,figsize=(20,20))

		for i in range(nchannels):
			logger.info("Plot the source image #%d (chan=%d) ..." % (data_index,i+1))
		
			title= 'Source ' + self.source_names[data_index] + ' - CHAN ' + str(i+1) 
			
			a = fig.add_subplot(1,nchannels,i+1)
			imgplot = plt.imshow(self.input_data[data_index,:,:,i],vmin=data_min,vmax=data_max)
			a.set_title(title)
			plt.colorbar(orientation='horizontal')

			#(fig, ax) = plt.subplots(nchannels,1,figsize=(20,20))
			#ax[0].set_title(title)
			#ax[0].set_xlabel("x")
			#ax[0].set_ylabel("y")
			#imgplot = plt.imshow(self.input_data[data_index,:,:,i])
			#plt.colorbar()

		plt.tight_layout()

		if save_to_file:
			plt.savefig(outfile)
			plt.close()
		else:	
			plt.show()
		

		return 0

	######################################
	##     SAVE IMAGE SIMILARITY DATA
	######################################
	def save_ssim_data(self,save_to_file=False):
		""" Save all similarity images to file """
		
		# - Check if data entry exists
		if self.input_data_ssim is None:
			logger.error("No input data present (hint: read data first!")
			return -1

		imgshape= self.input_data_ssim.shape
		nsamples= imgshape[0]
		nchannels= imgshape[3] 

		# - Loop over image data and draw/save
		for i in range(nsamples):
			logger.info("Saving similarity map no. %d ..." % (i+1))
			outfile= 'ssim_' + self.source_names[i] + '.png'
			self.__save_ssim_img(i,save_to_file,outfile)

		return 0
		

	def __save_ssim_img(self,data_index,save_to_file=False,outfile='ssim_plot.png'):
		""" Save similarity images to file """

		# - Check if data entry exists
		if self.input_data_ssim is None:
			logger.error("No input data present (hint: read data first!")
			return -1

		imgshape= self.input_data_ssim.shape
		nsamples= imgshape[0]
		nchannels= imgshape[3] 

		if data_index>=nsamples:
			logger.error("No input data present with index %d " % data_index)
			return -1
	
		# - Find maximum & minimum data value
		data_min= np.min(self.input_data_ssim[data_index])
		data_max= np.max(self.input_data_ssim[data_index])
		
		# - Initialize plot		
		fig = plt.figure(figsize=(20,20))
		
		for i in range(nchannels):
			logger.info("Plot the source ssim image #%d (chan=%d) ..." % (data_index,i+1))
		
			title= 'Source SSIM ' + self.source_names[data_index] + ' - CHAN ' + str(i+1) 
			
			a = fig.add_subplot(1,nchannels,i+1)
			imgplot = plt.imshow(self.input_data_ssim[data_index,:,:,i],vmin=data_min,vmax=data_max)
			a.set_title(title)
			plt.colorbar(orientation='horizontal')

		plt.tight_layout()

		if save_to_file:
			plt.savefig(outfile)
			plt.close()
		else:	
			plt.show()


		return 0


	################################################
	##     SAVE IMAGE SIMILARITY GRADIENT DATA
	################################################
	def save_ssimgrad_data(self,data_index,save_to_file=False,outfile='ssim_plot.png'):
		""" Save input data read to images """

		# - Check if data entry exists
		if self.input_data is None:
			logger.error("No input data present (hint: read data first!")
			return -1

		imgshape= self.input_data_ssimgrad.shape
		nsamples= imgshape[0]
		nchannels= imgshape[3] 

		if data_index>=nsamples:
			logger.error("No input data present with index %d " % data_index)
			return -1
	
		# - Find maximum & minimum data value
		data_min= np.min(self.input_data_ssimgrad[data_index])
		data_max= np.max(self.input_data_ssimgrad[data_index])
		
		# - Initialize plot		
		fig = plt.figure(figsize=(20,20))
		
		for i in range(nchannels):
			logger.info("Plot the source ssim gradient image #%d (chan=%d) ..." % (data_index,i+1))
		
			title= 'Source SSIM GRAD ' + self.source_names[data_index] + ' - CHAN ' + str(i+1) 
			
			a = fig.add_subplot(1,nchannels,i+1)
			imgplot = plt.imshow(self.input_data_ssimgrad[data_index,:,:,i],vmin=data_min,vmax=data_max)
			a.set_title(title)
			plt.colorbar(orientation='horizontal')

		plt.tight_layout()

		if save_to_file:
			plt.savefig(outfile)
			plt.close()
		else:	
			plt.show()


		return 0

