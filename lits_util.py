# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:36:08 2021

@author: klin0
"""
import os
import numpy as np 
import nibabel as nib
from scipy import ndimage
import tensorflow as tf
from unet_custom import unet2d


class Param():
    '''
    parameter class to store all the parameters
    '''
    def __init__(self):
        # problem/data parameters
        self.num_channels = 4
        self.num_classes = 2  # ignore background class
        self.n_samples = 131
        
        # preprocessing 
        self.window_min = -100
        self.window_max = 400
        self.patch_shape = (128, 128, 16)
        self.equalize_histogram = False  
        self.normalize = True
        self.zdist = 5  # set z spacing to 2mm
        self.output_type = 'npy'  
        
        # NN - Data generator
        self.data_dir = 'kaggle/input'
        self.batch_size = 1
        self.suffle = True
        self.data_split()
        
        # others
        self.verbose = 2
        
    def data_split(self, ratio = [0.6, 0.8]):
        indices = [i for i in range(self.n_samples)]
        np.random.shuffle(indices)
        splits = np.split(indices, [int(0.60*self.n_samples),int(0.80*self.n_samples)])
        self.training_list, self.validation_list, self.test_list = splits
        

class 2classDataGenerator(tf.keras.utils.Sequence):
    """ generate samples for liver segmentation and lesion segmentation 
    i.e. mask is of shape (self.batch_size, 2, x,y,z)
    """
    def __init__(self, param, sample_list):
        """
        sample_list are ID numbers of the training_list
        """
        self.batch_size = param.batch_size
        self.shuffle = param.shuffle
        self.base_dir = param.base_dir
        self.dim = param.patch_shape
        self.num_channels = param.num_channels
        self.num_classes = param.num_classes
        self.verbose = param.verbose
        self.sample_list = sample_list  
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sample_list) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
        
        'Generates data containing batch_size samples'

        # Initialization
        X = np.zeros((self.batch_size, *self.dim),
                     dtype=np.float64)
        y = np.zeros((self.batch_size, self.num_classes, *self.dim),
                     dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            print("Training on: %s" % self.base_dir + ID)
            X[i], tempy = self.generate_patch(ID)
            liver_mask = tempy>0
            lesion_mask = tempy == 2
            y[i, 0] = liver_mask[np.newaxis,...]
            y[i, 1] = lesion_mask[np.newaxis,...]
            assert X[i].shape == self.dim, "dim not matching"    

        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        sample_list_temp = [self.sample_list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(sample_list_temp)

        return X, y
    
    def generate_patch(self, ID):
        vol = np.load(os.path.join(self.base_dir, 'vol' + str(ID) + '.npy'))
        mask = np.load(os.path.join(self.base_dir, 'mask' + str(ID) + '.npy'))
        if vol.shape[-1] < self.dim[-1]:
            raise ValueError("volume depth less than patch depth")
        # calculate available index to generate patch and pick a random one
        max_index = vol.shape[-1]-self.dim[-1]
        start_index = np.random.choice([i for i in range(max_index)])
        end_index = start_index + self.dim[-1]
        return vol[:,:,start_index:end_index], mask[:,:,start_index:end_index]
    
    
        
        
        
    
    


# train-validation-test split

# model definition

# 