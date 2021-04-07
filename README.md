# LiTS - Liver Tumor Segmentation Challenge

## Steps:
1. Preprocessing: 
    - Windowing the HU values to [-100, 400] (ref [1])
    - Reduce width and height dimension to (168, 168) for fast iteration
    - Set scan spacings to 2mm for fast iteration 
    - Max-min normalization 
    - Saved data to .npy
    - Sample script available at https://www.kaggle.com/klin059/preprocessing
    or preprocessing.py
    
2. Training
    - Used a 3dUnet model as defined by the script for training
    - Predict liver segmentation and lesion segmentation at the same time
    - Used patch size of (168, 168, 16)
    - Defined a generator class to generate samples for training
        - each batch will sequentially load 3 (batch_size) volume samples and 
        for each volume sample 4 different randomly selected patches
        will be used as the output (hence there will be 12 patches in one batch)
    - Sample script at lits_training.py or https://www.kaggle.com/klin059/lits-2mm-v2-load-weights
    
3. Results
    - Currently the model get a dice score of ~0.9 for liver segmentation and 0.00036 for lesion segmentation
    
    




## refs:
[1] https://arxiv.org/pdf/1702.05970.pdf  

        