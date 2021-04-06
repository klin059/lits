# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:36:08 2021

@author: klin0
"""
import os
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt


class Param():
    '''
    parameter class to store all the parameters
    '''
    def __init__(self, data_dir='kaggle/input', partial_data = False):
        # problem/data parameters
        self.num_channels = 1
        self.num_classes = 2  # ignore background class
        self.n_samples = 131
        
        # preprocessing 
        self.window_min = -100
        self.window_max = 400
        self.patch_shape = (128, 128, 16)
        self.equalize_histogram = False  
        self.normalize = True
        self.zdist = 2  # set z spacing to zdist mm
        self.output_type = 'npy'  
        
        # others
        self.verbose = 2
        self.partial_data = partial_data  # using partial data for testing
        
        # NN - Data generator
        self.data_dir = data_dir
        self.batch_size = 1
        self.data_split()
        self.patch_per_ID = 3
        
        # NN - training
        
        
    def data_split(self, ratio = [0.6, 0.8]):
        indices = [i for i in range(self.n_samples)]
        if self.partial_data:
            self.training_list = [0,1]
            self.validation_list = [10]
            self.test_list = [100]
            indices = [i for i in [0, 1, 10, 100]]
        else:
            indices = [i for i in range(self.n_samples)]       
            np.random.shuffle(indices)
            splits = np.split(indices, [int(0.60*self.n_samples),int(0.80*self.n_samples)])
            self.training_list, self.validation_list, self.test_list = splits
        

class DataGenerator2class(tf.keras.utils.Sequence):
    """ generate samples for liver segmentation and lesion segmentation 
    i.e. mask is of shape (self.batch_size, 2, x,y,z)
    """
    def __init__(self, param, sample_list, shuffle = True):
        """
        sample_list are ID numbers of the training_list
        """
        self.batch_size = param.batch_size
        self.shuffle = shuffle
        self.base_dir = param.data_dir
        self.dim = param.patch_shape
        self.num_channels = param.num_channels
        self.num_classes = param.num_classes
        self.verbose = param.verbose
        self.sample_list = sample_list  
        self.on_epoch_end()
        self.patch_per_ID = param.patch_per_ID

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
        # patch_per_ID means generate patch_per_ID patches for one volume loaded
        X = np.zeros((self.batch_size*self.patch_per_ID, *self.dim, self.num_channels),
                     dtype=np.float64)
        y = np.zeros((self.batch_size*self.patch_per_ID, *self.dim, self.num_classes),
                     dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            vol = np.load(os.path.join(self.base_dir, 'vol' + str(ID) + '.npy'))
            mask = np.load(os.path.join(self.base_dir, 'mask' + str(ID) + '.npy'))
            if vol.shape[-1] < self.dim[-1]:
                raise ValueError("volume depth less than patch depth")
            # generate 5 patches per ID
            start, end = i*self.patch_per_ID, (i+1)*self.patch_per_ID
            X[start:end], tempy = self.generate_patch(vol, mask)
            liver_mask = tempy>0
            lesion_mask = tempy == 2
            y[start:end,:,:,:,0] = liver_mask
            y[start:end,:,:,:,1] = lesion_mask

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
    
    def generate_patch(self, vol, mask):
        # generate patch_per_ID patches
        max_index = vol.shape[-1]-self.dim[-1]
        vol_patch = np.zeros((self.patch_per_ID, *self.dim))
        mask_patch = np.zeros((self.patch_per_ID, *self.dim))
        
        for i in range(self.patch_per_ID):
            start_index = np.random.choice([i for i in range(max_index)])
            end_index = start_index + self.dim[-1]
            vol_patch[i] = vol[:,:,start_index:end_index]
            mask_patch[i] = mask[:,:,start_index:end_index]
        return vol_patch[..., np.newaxis], mask_patch

def plot_history(history, train_metric, val_metric, start_ind = 0):  
    # plot training and validation matrices  over epochs
    n_epochs = len(history.history[train_metric])
    fig, ax = plt.subplots(figsize=(8, 8 * 3 / 4))
    ax.plot(list(range(n_epochs))[start_ind:], history.history[train_metric][start_ind:], label=train_metric)
    ax.plot(list(range(n_epochs))[start_ind:], history.history[val_metric][start_ind:], label=val_metric)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(train_metric)
    ax.legend(loc='upper right')
    fig.tight_layout()

#def load_and_predict(base_dir, ID, model):
#    vol = np.load(os.path.join(base_dir, 'vol' + str(ID) + '.npy'))
#    mask = np.load(os.path.join(base_dir, 'mask' + str(ID) + '.npy'))
#    max_index = vol.shape[-1]-self.dim[-1]
#    start_index = np.random.choice([i for i in range(max_index)])
#    end_index = start_index + self.dim[-1]
#    return vol[:,:,start_index:end_index, np.newaxis], mask[:,:,start_index:end_index]
    
    
        
        
        
    
