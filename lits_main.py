# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:29:15 2021

@author: klin0
"""
import numpy as np
from unet_custom import unet3d, loss, metric
import lits_util
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import os

param = lits_util.Param(data_dir = 'kaggle/input', partial_data = True)

input_size = [i for i in param.patch_shape]
input_size.append(param.num_channels)
model = unet3d.unet3d(input_size = input_size, 
               n_classes=2, 
               dropout=0.1, 
               out_activation='sigmoid', 
               padding = 'same')
#dot_img_file = 'model.png'
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


model.compile(optimizer=Adam(lr=0.01), loss=loss.jaccard_distance_loss,
                  metrics=[loss.dice_coef])

train_generator = lits_util.DataGenerator2class(param, param.training_list)
val_generator = lits_util.DataGenerator2class(param, param.validation_list)

checkpoint_filepath = 'model.hdf5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_'+ model.metrics_names[-1],
    mode='max',
    save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                              verbose=0, mode='auto', min_delta=0.000001,
                              cooldown=0, min_lr=0.000001)

history = model.fit_generator(
            generator = train_generator,
            validation_data = val_generator,
            epochs = 2,
            verbose = 3,        
            callbacks=[model_checkpoint_callback, reduce_lr])



lits_util.plot_history(history, 'loss', 'val_loss', start_ind=0)
lits_util.plot_history(history, model.metrics_names[-1], 'val_'+model.metrics_names[-1], start_ind=0)

test_generator = lits_util.DataGenerator2class(
            param, param.test_list, shuffle = False
        )
loss_val, metric_val = model.evaluate_generator(test_generator)
print("loss_value, metric_value = ", loss_val, metric_val)

## eval on the entire scan

patch_depth = param.patch_shape[-1]
threshold = 0.5
dice_liver_ls = []
dice_lesion_ls = []
for ID in param.test_list:
    # load data
    vol = np.load(os.path.join(param.data_dir, 'vol' + str(ID) + '.npy'))
    vol_depth = vol.shape[-1]
    vol = vol[..., np.newaxis]
    # handling dimensions/formatting
    mask = np.load(os.path.join(param.data_dir, 'mask' + str(ID) + '.npy'))
    mask_liver = mask > 0
    mask_lesion = mask == 2
    y_true = np.zeros((1, *mask_liver.shape, 2))
    y_true[0,:,:,:,0] = mask_liver
    y_true[0,:,:,:,1] = mask_lesion
    
    y_pred = np.zeros((1, *mask_liver.shape, 2))
    
    # iterate by patch shape and predict y_patch
    starting_indexes = [ind for ind in range(0, vol_depth-patch_depth, patch_depth)]
    
    if starting_indexes[-1] < vol_depth-patch_depth:
        # add the patch ending at the scan depth index
        starting_indexes.append(vol_depth-patch_depth)
    
    for depth in range(0, vol_depth-patch_depth, patch_depth):
        y_patch_pred = model.predict(vol[np.newaxis, :,:,depth:depth+patch_depth, :])
#        y_patch_pred = y_pred.astype("float64")
        y_patch_pred = y_patch_pred > threshold
        # use logical_or to handle potential overlaps predicted by the previous iteration
        y_pred[:,:,:,depth:depth+patch_depth,:] = np.logical_or(y_patch_pred, y_pred[:,:,:,depth:depth+patch_depth,:])
        
    # get dice coef metric for each class 
    dice_liver = metric.dice_coef_np(mask_liver, y_pred[0,:,:,:,0])
    dice_liver_ls.append(dice_liver)
    dice_lesion = metric.dice_coef_np(mask_lesion, y_pred[0,:,:,:,1])
    dice_lesion_ls.append(dice_lesion)

dice_liver_ls
dice_lesion_ls    

    
