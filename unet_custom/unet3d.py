# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:11:46 2021

@author: klin0
"""
# unet 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        BatchNormalization,
        Conv3D,
        Conv3DTranspose,
        MaxPooling3D,
        Dropout,
        SpatialDropout3D,
        UpSampling3D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
    )

def conv_3dblock(inputs, n_filters, kernel_size = (3,3,3), padding = 'same', dropout = 0.1, revert = False, res_connect = False):
    if not revert:
        n1 = n_filters//2
        n2 = n_filters
    else:
        n1 = n_filters
        n2 = n_filters//2
    x = Conv3D(n1, kernel_size, activation = 'relu', padding = padding)(inputs)
    x = BatchNormalization()(x)
    if dropout > 0:
        x = SpatialDropout3D(dropout)(x)
    x = Conv3D(n2, kernel_size, activation = 'relu', padding = padding)(inputs)
    x = BatchNormalization()(x)
    if res_connect:
        res = Conv3D(n2, kernel_size = (1,1,1), activation = 'relu', padding = padding)(inputs)
        x = add([x, res])
    return x

def encoder_3dblock(inputs, n_filters, pool_size = (2,2,2), res_connect = False,
                  strides = (2,2,2), dropout = 0.1, padding = 'same'):
    
    convblock = conv_3dblock(inputs, n_filters, dropout = dropout, res_connect = res_connect)
    
    pooled = MaxPooling3D(pool_size)(convblock)
    return convblock, pooled

def decoder_3dblock(inputs,  concat_block, n_filters, kernel_size = (2,2,2), res_connect = False,
                  strides = (2,2,2), padding = 'same', dropout = 0):
    
    x = Conv3DTranspose(n_filters, kernel_size, strides = strides, padding = padding)(inputs)
    
    c = concatenate([concat_block, x])
    
    c = conv_3dblock(c, n_filters, padding = padding, revert = True, res_connect = res_connect)
    
    return c


def unet3d(input_size, n_classes=1, out_activation='sigmoid', res_connect = False,
           padding = 'same', filter_sizes = [64, 128, 256, 256], dropout=0.2):
    
    levels = len(filter_sizes)-1
    if padding == 'same':
        for i in range(levels):
            for j in range(3):
                if input_size[j]%(2**i) != 0:
                    raise ValueError("model output shape won't be the same as input shape due to rounding dimension during pooling")
    # encoder
    inputs = Input(input_size)
    pooled = inputs
    encoders = []
    
    for i in range(levels):
        encoder, pooled = encoder_3dblock(pooled, n_filters = filter_sizes[i], dropout = dropout, padding = padding, res_connect = res_connect)
        encoders.append(encoder)
    
    bottle_neck = conv_3dblock(pooled, n_filters = filter_sizes[-1], dropout = dropout, padding = padding)
    
    # decoder
    decoder = bottle_neck
    for i in range(levels-1, -1, -1):
        decoder = decoder_3dblock(decoder, encoders[i], n_filters = filter_sizes[i], dropout = dropout, padding = padding, res_connect = res_connect)
    
    outputs = Conv3D(n_classes, (1,1,1), activation = out_activation)(decoder)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model



if __name__ == "__main__":
    
    unet_orig = unet3d(input_size = (512, 512, 200, 1), n_classes=1, dropout=0, out_activation='sigmoid', padding = 'same')
    unet_orig.summary()
    dot_img_file = '3dunet.png'
    tf.keras.utils.plot_model(unet_orig, to_file=dot_img_file, show_shapes=True)
    
    unet_residual = unet3d(input_size = (512, 512, 200, 1), res_connect = True, n_classes=1, dropout=0, out_activation='sigmoid', padding = 'same')
    unet_residual.summary()
    dot_img_file = 'res3dunet.png'
    tf.keras.utils.plot_model(unet_residual, to_file=dot_img_file, show_shapes=True)