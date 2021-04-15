# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:34:36 2021

@author: klin0
"""
import lits_util
"""

"""
subset_ids = [0,1,10,100]
def test_DataGenerator2class():
    param = lits_util.Param(partial_data = True, resize_option="by_zdist")
    gen = lits_util.DataGenerator2class(param, subset_ids)
    gen.batch_size = 2
    gen.patch_per_ID = 3
    X, y = gen.__getitem__(0)
    n_patches = gen.batch_size * gen.patch_per_ID
    assert X.shape == (n_patches, *param.patch_shape, 1)
    assert y.shape == (n_patches, *param.patch_shape, 2)
    assert X.dtype == 'float32'
    assert y.dtype == 'float16'
    
def test_DataGenerator_2class_wholeVolume():
    param = lits_util.Param(data_dir='kaggle\\input\\whole_vol', partial_data = True, resize_option="by_vol")
    gen = lits_util.DataGenerator_2class_wholeVolume(param, subset_ids)
    gen.batch_size = 2
    X, y = gen.__getitem__(0)
    assert X.shape == (2, *param.resized_vol_shape, 1)
    assert y.shape == (2, *param.resized_vol_shape, 2)
    assert X.dtype == 'float32'
    assert y.dtype == 'float16'
    
    
# test DataGenerator_2class_cascade
def test_DataGenerator_2class_cascade():
    param = lits_util.Param(partial_data = True, resize_option="by_vol")
    gen  = lits_util.DataGenerator_2class_cascade(param, subset_ids)
    gen.batch_size = 2
    gen.patch_per_ID = 3
    X, y = gen.__getitem__(0)
    n_patches = gen.batch_size * gen.patch_per_ID
    assert X.shape == (n_patches, *param.patch_shape, 1)
    assert y[0].shape == (n_patches, *param.patch_shape, 1)
    assert y[1].shape == (n_patches, *param.patch_shape, 1)
    assert X.dtype == 'float32'
    assert y[0].dtype == 'float16'
    assert y[1].dtype == 'float16'

# test DataGenerator_2class_wholeVolume_cascade
def test_DataGenerator_2class_wholeVolume_cascade():
    param = lits_util.Param(data_dir='kaggle\\input\\whole_vol', partial_data = True, resize_option="by_vol")
    
    gen2 = lits_util.DataGenerator_2class_wholeVolume_cascade(param, subset_ids)
    # gen2.dim = param.resized_vol_shape
    gen2.batch_size = 2
    X, y = gen2.__getitem__(0)
    assert X.shape == (2, *param.resized_vol_shape, 1)
    assert y[0].shape == (2, *param.resized_vol_shape, 1)
    assert y[1].shape == (2, *param.resized_vol_shape, 1)
    assert X.dtype == 'float32'
    assert y[0].dtype == 'float16'
    assert y[1].dtype == 'float16'
    
def test_DataGenerator_liverMask_wholeVolume():
    param = lits_util.Param(data_dir='kaggle\\input\\whole_vol', partial_data = True, resize_option="by_vol")
    param.num_classes = 1
    gen = lits_util.DataGenerator_liverMask_wholeVolume(param, subset_ids)
    gen.batch_size = 2
    X, y = gen.__getitem__(0)
    assert X.shape == (2, *param.resized_vol_shape, 1)
    assert y.shape == (2, *param.resized_vol_shape, 1)
    assert X.dtype == 'float32'
    assert y.dtype == 'float16'
    
