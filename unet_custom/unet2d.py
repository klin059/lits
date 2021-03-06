# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:11:46 2021

@author: klin0
"""
# unet 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, Cropping2D, add, Activation, BatchNormalization

def conv2d_block(inputs, n_filters, kernel_size = 3, padding = 'valid', res_connect = False):
    
    x = inputs
    for i in range(2):
        x = Conv2D(n_filters, kernel_size, padding = padding, activation = 'relu')(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
    
    if res_connect:
        res = Conv2D(n_filters, kernel_size = 1, padding = padding, activation = 'relu')(inputs)
        x = add([x, res])
    return x

def encoder_block(inputs, n_filters, pool_size = (2,2), strides = (2,2), dropout = 0, padding = 'valid', res_connect = False):
    convblock = conv2d_block(inputs, n_filters, 3, padding, res_connect = res_connect)
    pooled = MaxPooling2D(pool_size)(convblock)
    if dropout > 0:
        pooled = Dropout(dropout)(pooled)
    return convblock, pooled


def shapes_to_crop(target, reference):
    # decide the shape to crop the feature map for unet concatenation
    # height
    ch = target[1] - reference[1]
    if ch % 2 != 0:
        ch1, ch2 = ch//2, ch//2 + 1
    else:
        ch1, ch2 = ch//2, ch//2
    # width
    cw = target[2] - reference[2]
    if cw % 2 != 0:
        cw1, cw2 = cw//2, cw//2 + 1
    else:
        cw1, cw2 = cw//2, cw//2
    
    return (ch1, ch2), (cw1, cw2)
    
def decoder_block(inputs,  concat_block, n_filters, kernel_size = (2,2), res_connect = False,
                  strides = (2,2), padding = 'valid', dropout = 0):
    
    x = Conv2DTranspose(n_filters, kernel_size, strides = strides, padding = padding)(inputs)
    
    # cropping
    if concat_block.shape != x.shape:
        ch, cw = shapes_to_crop(concat_block.shape, x.shape)
        concat_block = Cropping2D(cropping=(ch, cw))(concat_block)
    
    # concat
    c = concatenate([concat_block, x])
    if dropout > 0:
        c = Dropout(dropout)(c)
    
    c = conv2d_block(c, n_filters, padding = padding, res_connect = res_connect)
    
    return c

def unet2d_block(inputs, n_classes=1, dropout=0, out_activation='sigmoid', res_connect = False,
           padding = 'valid', filter_sizes = [64, 128, 256, 512, 1024]):
    levels = len(filter_sizes)-1    
    # encoder
    pooled = inputs
    encoders = []
    
    for i in range(levels):
        encoder, pooled = encoder_block(pooled, n_filters = filter_sizes[i], dropout = dropout, padding = padding, res_connect = res_connect)
        encoders.append(encoder)
    
    bottle_neck = conv2d_block(pooled, n_filters = filter_sizes[-1], padding = padding, res_connect = res_connect)
    
    # decoder
    decoder = bottle_neck
    for i in range(levels-1, -1, -1):
        decoder = decoder_block(decoder, encoders[i], n_filters = filter_sizes[i], dropout = dropout, padding = padding, res_connect = res_connect)
    
    outputs = Conv2D(n_classes, (1,1), activation = out_activation)(decoder)
    return inputs, outputs

def unet2d(input_size, n_classes=1, dropout=0, out_activation='sigmoid', res_connect = False,
           padding = 'valid', filter_sizes = [64, 128, 256, 512, 1024]):
    
    levels = len(filter_sizes)-1
    if padding == 'same':
        if input_size[0]%(2**levels) != 0 or input_size[1]%(2**levels) != 0:
                raise ValueError("model output shape won't be the same as input shape due to rounding dimension during pooling")
    
    inputs, outputs = unet2d_block(Input(input_size), n_classes, dropout, out_activation, res_connect,
           padding, filter_sizes)
    
    return Model(inputs = inputs, outputs = outputs)

def cascaded_unet2d(input_size, n_classes=1, dropout=0, out_activation='sigmoid', res_connect = False,
           padding = 'valid', filter_size1 = [64, 128, 256, 512, 1024], filter_size2 = [32, 64, 128, 128], 
           add_input = True):
    input1, output1 = unet2d_block(Input(input_size), n_classes, dropout, out_activation, res_connect,
           padding, filter_size1)
    if add_input:
        input2 = add([input1, output1])
    else:
        input2 = output1
    
    input2, output2 = unet2d_block(input2, n_classes, dropout, out_activation, res_connect,
           padding, filter_size2)
    return Model(inputs = input1, outputs = [output1, output2])

if __name__ == "__main__":
    import tensorflow as tf
    
    # unet with "same" padding
    same_padding_model = unet2d(input_size = (512, 512, 1), n_classes=1, dropout=0, out_activation='sigmoid', padding = 'same')
    same_padding_model.summary()
    dot_img_file = '2dunet.png'
    tf.keras.utils.plot_model(same_padding_model, to_file=dot_img_file, show_shapes=True)
    
    # res unet
    same_padding_model = unet2d(input_size = (512, 512, 1), n_classes=1, dropout=0, out_activation='sigmoid', padding = 'same', res_connect = True)
    same_padding_model.summary()
    dot_img_file = '2dunet_res_connect.png'
    tf.keras.utils.plot_model(same_padding_model, to_file=dot_img_file, show_shapes=True)
    
    # cascade unet
    unet_cascade = cascaded_unet2d(input_size = (512, 512, 1), n_classes=1, dropout=0, out_activation='sigmoid', padding = 'same')
    unet_cascade.summary()
    dot_img_file = '2dunet_cascaded.png'
    tf.keras.utils.plot_model(unet_cascade, to_file=dot_img_file, show_shapes=True)