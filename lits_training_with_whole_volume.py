# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:29:15 2021
Training with whole volume (168, 168, 168) as the input, mask as output
@author: klin0
"""
import numpy as np
from unet_custom import unet3d, loss
import lits_util
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau


#------------ controls --------------
partial_data = True  # True for training with only subsets of data (for testing purpose)
if partial_data:
    n_epochs = 2
else:
    n_epochs = 100

data_dir = 'kaggle/input/whole_vol'  
checkpoint_filepath = 'model.hdf5'
model_output_directory = "final_model"
# -----------------------------------

param = lits_util.Param(data_dir = data_dir, partial_data = partial_data, resize_option="by_vol")  
param.num_classes = 1  # work on liver seg only for now
input_size = [i for i in param.resized_vol_shape]
input_size.append(param.num_channels)
model = unet3d.unet3d(input_size = input_size, 
               n_classes=1, 
               dropout=0.1, 
               out_activation='sigmoid', 
               padding = 'same',
               res_connect = True)

metric_name = "val_dice_coef"
model.compile(optimizer=Adam(lr=0.01), loss=loss.jaccard_distance_loss,
                  metrics=[loss.dice_coef])

train_generator = lits_util.DataGenerator_liverMask_wholeVolume(param, param.training_list)
val_generator = lits_util.DataGenerator_liverMask_wholeVolume(param, param.validation_list)

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_dice_coef',
    mode='max',
    save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=5,
                              verbose=0, mode='auto', min_delta=0.000001,
                              cooldown=0, min_lr=0.000001)
history = model.fit(
            x = train_generator,
            validation_data = val_generator,
            epochs = n_epochs,
            verbose = 3, 
            callbacks=[model_checkpoint_callback, reduce_lr]) 

lits_util.plot_history(history, 'loss', 'val_loss', start_ind=0)
lits_util.plot_history(history, model.metrics_names[-1], 'val_'+model.metrics_names[-1], start_ind=0)

best_epoch = np.argmax(history.history[metric_name])
print(f'Best epoch at {best_epoch+1} out of {n_epochs} epochs')
for key in history.history:
    print(f'{key}: {history.history[key][best_epoch]}')


model.load_weights(checkpoint_filepath)
<<<<<<< HEAD
test_generator = lits_util.DataGenerator_liverMask_wholeVolume(
=======
test_generator = lits_util.DataGenerator2class(
>>>>>>> 33ae1b760cc4e4c6e3c0a46a18dd9e4ffb7599c5
            param, param.test_list, shuffle = False
        )
loss_val, metric_val = model.evaluate(x = test_generator)

print("Test set loss_value, metric_value = ", loss_val, metric_val)

model.save(model_output_directory)

<<<<<<< HEAD
import pickle
with open("history.p", "wb") as f:
    pickle.dump(history.history, f)
=======
>>>>>>> 33ae1b760cc4e4c6e3c0a46a18dd9e4ffb7599c5



