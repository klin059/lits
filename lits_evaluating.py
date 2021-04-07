# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:50:44 2021

@author: klin0
"""
import os
import numpy as np
from unet_custom import loss, metric
import lits_util
from tensorflow.keras.models import load_model

param = lits_util.Param(partial_data = True)

model_dir = "final_model"

## load from the saved model
model = load_model(model_dir, 
                   custom_objects={'jaccard_distance_loss': loss.jaccard_distance_loss,
                                   'dice_coef' : loss.dice_coef})
model.load_weights("model.hdf5")
model.save(model_dir)
## eval on the entire volume
dice_liver_ls = []
dice_lesion_ls = []
for ID in param.test_list:
    pred_liver_mask, pred_lesion_mask = lits_util.model_prediction(model, ID, param, threshold = 0.5)
    liver_mask, lesion_mask = lits_util.get_masks(ID, param)
        
    # get dice coef metric for each class 
    dice_liver = metric.dice_coef_np(liver_mask, pred_liver_mask)
    dice_liver_ls.append(dice_liver)
    dice_lesion = metric.dice_coef_np(lesion_mask, pred_lesion_mask)
    dice_lesion_ls.append(dice_lesion)

print("Average dice scores for liver and lesion are {dice_liver_ls} and {dice_lesion_ls} consecutively")
    
ID = 0
vol = np.load(os.path.join(param.data_dir, 'vol' + str(ID) + '.npy'))
mask = np.load(os.path.join(param.data_dir, 'mask' + str(ID) + '.npy'))
pred_liver_mask, pred_lesion_mask = lits_util.model_prediction(model, 0, param)
pred_mask = pred_liver_mask.copy()
pred_mask[pred_lesion_mask==1] = 2


# plot individual scan and masks
scan_index = 5
lits_util.plot_scan_and_masks(scan_index, vol, mask, pred_mask)

# plot scan, masks over the indices evenly spaced samples
for i in range(vol.shape[-1]):
    if np.any(mask[:,:,i]):
        break
lits_util.plot_mask_comparison_over_vol(vol, mask, pred_mask, index_start=i, n_samples = 10)


