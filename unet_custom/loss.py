# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:20:50 2021

default mask shape (n_batch, x, y, z, n_classes)

@author: klin0
"""

from tensorflow.keras import backend as K 
#from medpy import metric
#from surface import Surface

#def soft_dice_loss(y_true, y_pred, axis=(0, 1, 2, 3), 
#                   epsilon=1):
#    """
#    compute soft dice loss
#    """
#    dice_numerator =2* K.sum(y_pred * y_true,axis=axis) + epsilon
#    dice_denominator = (K.sum(y_pred**2,axis=axis) + K.sum(y_true**2,axis=axis)) + epsilon
#    dice_loss = 1 - K.mean(dice_numerator / dice_denominator)
#
#    return dice_loss
#
#def dice_coefficient(y_true, y_pred, axis=(0, 1, 2, 3), 
#                     epsilon=0.00001):
#    """
#    Compute mean dice coefficient over all classes.     
#    """
#    
#    dice_numerator =2* K.sum(y_pred * y_true,axis=axis) + epsilon
#    dice_denominator = (K.sum(y_pred,axis=axis) + K.sum(y_true,axis=axis)) + epsilon
#    dice_coefficient = K.mean(dice_numerator / dice_denominator)
#
#    return dice_coefficient

def jaccard_distance_loss(y_true, y_pred, axis=(0,1,2,3), smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=axis)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice_coef(y_true, y_pred, axis=(0,1,2,3), smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),axis) + K.sum(K.square(y_pred),axis) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-K.mean(dice_coef(y_true, y_pred))

