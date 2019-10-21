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

## UMAP MODULES
import umap

## GRAPHICS MODULES
import matplotlib.pyplot as plt

## PACKAGE MODULES
from .utils import Utils
from .data_provider import DataProvider

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)



##############################
##     Classifier CLASS
##############################
class UMAPClassifier(object):
	""" Class to create and train a UMAP classifier

			Arguments:
				- DataProvider class
	"""
	
	def __init__(self,data_provider):
		""" Return a Classifer object """

		# - Input data
		self.dp= data_provider
		self.nsamples= 0
		self.nx= 0
		self.ny= 0
		self.nchannels= 1	
		self.data= None
		self.data_labels= {}
		self.source_names= []
		
		# - Reducer & parameters
		self.reducer= None
		self.encoded_data_dim= 2
		self.encoded_data= None
		self.metric= 'euclidean' # {'manhattan','chebyshev','minkowski','mahalanobis','seuclidean',...}
		self.metric_args= {}
		self.target_metric= 'categorical'
		self.target_metric_args= {}
		self.min_dist= 0.1 # 0.1 is default, larger values (close to 1) --> broad structures, small values (close to 0) --> cluster objects
		self.n_neighbors= 15 # 15 is default
		self.embedding_init= 'spectral' # {'spectral','random'}
		self.embedding_spread= 1.0 # default=1.0
		self.embedding_apar= None # 1.576943460405378 in digit example
		self.embedding_bpar= None # 0.8950608781227859 in digit example
		self.op_mix_ratio= 1.0 # default=1.0, in range [0,1]
		self.negative_sample_rate= 5 # default=5
		self.transform_queue_size= 4.0 # default=4
		self.angular_rp_forest= False # default=false
		self.local_connectivity= 1.0 # default=1
		self.nepochs= None # default=None
		self.random_seed= 42

		# - Draw options
		self.marker_mapping= {
			0: 'o', # unknown
			-1 : 'X', # mixed type
			1: 'x', # star
			2: 'D', # galaxy
			3: '+', # PN
			6: 's', # HII
		}
		self.marker_color_mapping= {
			0: 'k', # unknown
			-1: 'k', # mixed type
			1: 'r', # star
			2: 'y', # galaxy
			3: 'b', # PN
			6: 'g', # HII
		}
		

		# - Output data
		self.outfile_encoded_data= 'encoded_data.dat'

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_encoded_data_outfile(self,outfile):
		""" Set name of encoded output file """
		self.outfile_encoded_data= outfile	

	def set_encoded_data_dim(self,dim):
		""" Set encoded data dim """
		self.self.encoded_data_dim= dim

	

	#####################################
	##     SET TRAIN DATA
	#####################################
	def __set_data(self):
		""" Set train data from provider """

		# - Retrieve input data info from provider
		self.data= self.dp.get_data()
		imgshape= self.data.shape

		self.data_labels= self.dp.get_data_labels()
		self.source_names= self.dp.get_source_names()
			
		# - Check if data provider has data filled
		if self.data.ndim!=4:
			logger.error("Invalid number of dimensions in train data (4 expected) (hint: check if data was read in provider!)")
			return -1

		# - Set data
		self.nsamples= imgshape[0]
		self.nx= imgshape[2]
		self.ny= imgshape[1]
		self.nchannels= imgshape[3] 

		# - Flatten input data to 2D (Nsamples x (nx*ny*nchan))
		self.data= self.data.reshape(self.nsamples,-1)

		logger.info("Train data size (N,nx,ny,nchan)=(%d,%d,%d,%d)" % (self.nsamples,self.nx,self.ny,self.nchannels))
		logger.info("Flattened data size=",self.data.shape)

		return 0

	
	#####################################
	##     TRAIN REDUCER
	#####################################
	def __train_reducer(self):
		""" Build and train reducer """

		#================================
		#==   BUILD REDUCER
		#================================
		logger.info("Creating the reducer ...")
		self.reducer= umap.UMAP(
			random_state=self.random_seed,
			n_components=self.encoded_data_dim,
			metric=self.metric,
			n_neighbors=self.n_neighbors,
			min_dist=self.min_dist, 
			a=self.embedding_apar, 
			angular_rp_forest=self.angular_rp_forest,
			b=self.embedding_bpar, 
			init=self.embedding_init,
			local_connectivity=self.local_connectivity, 
			metric_kwds=self.metric_args,
			n_epochs=self.nepochs,
   		negative_sample_rate=self.negative_sample_rate, 
			set_op_mix_ratio=self.op_mix_ratio,
   		spread=self.embedding_spread, 
			target_metric=self.target_metric, 
			target_metric_kwds=self.target_metric_args,
   		transform_queue_size=self.transform_queue_size, 
			transform_seed=self.random_seed, 
			verbose=False
		)

		#================================
		#==   FIT DATA
		#================================
		logger.info("Fitting input data ...")
		self.encoded_data= self.reducer.fit_transform(self.data)
		
		#================================
		#==   SAVE ENCODED DATA
		#================================
		logger.info("Saving encoded data to file ...")
		N= self.encoded_data.shape[0]
		print("Encoded data shape=",self.encoded_data)
		print("encoded_data N=",N)

		obj_ids= []
		obj_subids= []

		for i in range(N):
			source_name= self.source_names[i]
			has_labels= source_name in self.input_labels		
			obj_id= 0
			obj_subid= 0

			if has_labels:
				obj_info= self.input_labels[source_name]
				obj_id= obj_info['id']
				obj_subid= obj_info['subid']
				
			obj_ids.append(obj_id)
			obj_subids.append(obj_subid)

			#print('Source name=%s, has_labels=%d, id=%d, subid=%s' % (source_name,has_labels,obj_id,obj_subid))

		# - Merge encoded data
		snames= np.array(self.source_names).reshape(N,1)
		objids= np.array(obj_ids).reshape(N,1)
		objsubids= np.array(obj_subids).reshape(N,1)
		#print("snames size=",snames.shape)
		#print("objids size=",objids.shape)
		#print("objsubids size=",objsubids.shape)
		#print("self.encoded_data size=",self.encoded_data.shape)

		enc_data= np.concatenate(
			(snames,self.encoded_data,objids,objsubids),
			axis=1
		)

		head= '#sname z1 z2 id subid'
		Utils.write_ascii(enc_data,self.outfile_encoded_data,head)	



	#####################################
	##     RUN CLASSIFIER
	#####################################
	def train(self):
	
		#===========================
		#==   SET INPUT DATA
		#===========================	
		logger.info("Setting data from provider ...")
		status= self.__set_data()
		if status<0:
			logger.error("Set data failed!")
			return -1

		#===========================
		#==   TRAIN REDUCER
		#===========================
		logger.info("Training reducer ...")
		status= self.__train_reducer()
		if status<0:
			logger.error("Reducer training failed!")
			return -1

