# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 20:17:22 2021

functions used for downsampling 

@author: klin0
"""
import os
import numpy as np 
import nibabel as nib
from scipy import ndimage


class Param():
    '''
    parameter class to store all the parameters
    '''
    def __init__(self):
        self.window_min = -100
        self.window_max = 400
        self.patch_shape = (128, 128, 16)
        self.equalize_histogram = False  
        self.normalize = True
        self.zdist = 5  # set z spacing to 2mm
        self.output_type = 'npy'  



# preprocessing functions
def read_nii(f):
    img_obj = nib.load(f)
    zdist = img_obj.header['srow_z'][-2]
    img_data = img_obj.get_fdata()
    return img_data, zdist

def hist_eq(image, number_bins=20):
    # histogram equalization
    # adopt from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def windowing(nparray_2d, _min, _max):
    # Setting hounsfield unit values to [−100, 400] to discard irrelevant structures
    np.clip(nparray_2d, _min, _max)

def norm(nparray):
    # normalize scans to [0,1]
    _min = nparray.min()
    _max = nparray.max()
    nparray = nparray - _min
    nparray = nparray / (_max - _min)
    return nparray

def norm_zscore(nparray):
    # normalize 2d scands by mean and standard deviation
    mean = nparray.mean()
    std = nparray.std()    
    nparray = nparray - mean
    nparray /= std
    return nparray

def resize_volume(orig_volume, zdist, params, order):
    """
    resize scans to desired dimension defined in parameters
    """
    resize_factor = [0]*3
    for i in range(2):
        resize_factor[i] = params.patch_shape[i]/orig_volume.shape[i]
    # rescale scan spacing to 2mm
    resize_factor[2] = zdist/params.zdist
    resized_vol = ndimage.zoom(orig_volume, resize_factor, order = order)
    return resized_vol

def preprocessing_vol(f_vol, param):
    
    vol, zdist = read_nii(f_vol)
    # windowing
    vol = np.clip(vol, param.window_min, param.window_max)  
    # resizing vol
    vol = resize_volume(vol, zdist, params, order = 3)
    
    # histogram equalization
    if param.equalize_histogram:
        vol = hist_eq(vol)
    
    # normalizing
    if param.normalize:
        vol = norm(vol)
#        vol = norm_zscore(vol)
        
    # output zdist for preprocessing_mask (spacing between scan and mask is different for some cases)
    return vol, zdist  

def preprocessing_mask(f_mask, zdist, param):
    mask, _ = read_nii(f_mask)
    mask = resize_volume(mask, zdist, params, order = 0)
    mask = np.rint(mask)
    return mask.astype(int)
    

def load_filepaths_to_dictionaries(path):
    """
    output dictionaries that record the input file paths
    """
    volumes = {}
    segments = {}
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename[:3] == 'vol':
                num = filename.split('-')[1]
                num = num.split('.')[0]
                volumes[int(num)] = os.path.join(dirname, filename)
            elif filename[:3] == 'seg':
                num = filename.split('-')[1]
                num = num.split('.')[0]
                segments[int(num)] = os.path.join(dirname, filename)
                
    assert(len(volumes.keys()) == len(segments.keys()))
    for k in volumes.keys():
        assert(k in segments)
    return volumes, segments

if __name__ == "__main__":
    path = r'kaggle/input'
    params = Param()

    volume_dict, segment_dict = load_filepaths_to_dictionaries(path = path)
    
    for key in volume_dict.keys():
        vol, zdist = preprocessing_vol(volume_dict[key], params) 
        mask = preprocessing_mask(segment_dict[key], zdist, params)
        if vol.shape != mask.shape:
            print("key, vol, mask: " + key + vol.shape + mask.shape)
        
        np.save('vol'+ str(key)+'.npy', vol)
        np.save('mask'+ str(key)+'.npy', mask)
        break
