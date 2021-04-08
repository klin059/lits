# LiTS - Liver Tumor Segmentation Challenge

## Steps:
1. Preprocessing: 
    - Windowing the HU values to [-100, 400] (ref [1])
    - Reduce width and height dimension to (168, 168) for fast iteration
    - Set scan spacings to 2mm for fast iteration 
    - Max-min normalization 
    - Saved data to .npy
    - Sample script available at https://www.kaggle.com/klin059/preprocessing
    or at preprocessing.py
    
2. Training
    - Used a 3d Unet model as defined by the script for training. The model developed with reference 
    on the 2d unet model (ref [2]) with 3d counterparts.
    - Predict liver segmentation and lesion segmentation at the same time
    - Used patch size of (168, 168, 16)
    - Defined a generator class to generate samples for training
        - each batch will sequentially load 3 (batch_size) volume samples and 
        for each volume sample 4 different randomly selected patches
        will be used as the output (hence there will be 12 patches in one batch)
    - Sample script at lits_training.py or at https://www.kaggle.com/klin059/lits-2mm-v2-load-weights
    - No data augmentation were used but it was fairly easy to implement augmentations in the generator class
    
3. Results
    - https://www.kaggle.com/klin059/lits-2mm-v3
        - this version trained for 300 epochs
        - average dice scores for liver and lesion are 0.85 and 0.058 consecutively on the test set
        - this version provides visualization 
    - I stopped training as I ran out of time
    
## Questions
### Why choose the model?
3d unet was chosen to produce a baseline model. It is shown in ref [3] that
in the KITS 2019 challenge the top solutions all use 3d unet-like 
architecture so there is no reason to reject using 3d unet for the LiTs challenges.

Two alternative model options are 3d unet with residual connections and cascaded 3d unet. 
Major considerations for not choosing them would be memory consumption. Currently I 
am training on Google colab and Kaggle and with the current setting I can barely increase the model 
size. (Although it might be possible to increase model size and reduce the batch size, which is something
 left to try.)

However, consider the nature of the problem and the results we have so far, 
it may be better to tailor the network for each subproblem at hand, e.g., instead 
of predicting liver and tumor segmentation in one model, we could have one model 
predicts liver segmentation and another predicts tumor segmentation, with predicted 
liver mask as an additional input.

### What to do next for hyperparameter tuning? 
The results show that lesion segmentation performed far inferior than liver segmentation. I have 
tried tuning the loss_weights but it only led to mild improvement on lesion segmentation but at the 
cost of around 0.2 decrease in liver segmentation dice score. [https://www.kaggle.com/klin059/lits-2mm]

We will refer to [https://www.kaggle.com/klin059/lits-2mm-v3] for the remaining discussion.

The training, validation and testing loss are 59.9, 60.7 and 58.5, respectively (with checkpoint metric being dice score). 
From the history plots, we do not see signs of overfitting. If I have the time and GPU quota, I would try to train the model 
until it overfits. In addition, consider the inputs (volume patches) are selected randomly, I would increase 
the patience parameter of the ReduceLROnPlateau callback, so that the model receives more samples before reducing the 
learning rate.

From the result visualization it seems like liver segmentation for the top and 
buttom section of the liver were doing much inferior than the mid sections. It may be due to inadequate 
training (e.g. top and buttom sections were chosen as inputs less frequently than the mid section) so I would 
train the model for longer. Alternatively I could also constrain the generator such that the input patch 
must contain liver (or increase the probability of sampling patches containing the liver and/or liver lesion).

Before excessively tune the hyperparameters, I would try other model architectures as discussed in the previous section 
and see how it goes.

### Any insights gained?

- 3d unet might not be the best model for liver lesion segmentation. 3d unet is great for 
capturing 3d relationship between scans but it seems like there is not much 3d 
relationships in lesion volumes (as observed from exploratory analysis). Alternatively, 
2d unet might be more suitable for lesion segmentation.
    
- Exploratory analysis shows that by using contrast enhencement (i.e. histogram equalization) on each of the 2d slices, 
it is easy to visually differentiate liver and tumor lesion. Volume (3d) based Contrast enhancement were not used in the preprocessing due 
to the histogram dominated by the non-patient volume (the volumes wrapping around the patient volume). 
leading to little contrast between liver and liver lesion. I was reluctuant to do 2d contrast enhancement since it 
leads to different value scaling between slices. However, perhaps its ok to do 2d contrast enhancement 
since adjacent slices are likely to have similar value scalings. Ref [1] also used contrast enhancement but it was not 
clear wheather it was 2d based or 3d based.
    
- Due to memory limitation I only use small patches as model input (currently the patch volume depth is only 32 mm). 
I am not sure how easy it is for the model to capture 3d information with such a small depth. I would be keen to 
try a model that receive the whole volume as input (i.e. scale to whole volume to a certain input dimension) 
then the 3d relationships would be more appearant to the model. The model then use the 3d relatioship to find liver 
mask and the mask can then be rescaled back to the original dimension and then feed to another 2d model that does liver 
and liver lesion segmentation.

## References:
[1] https://arxiv.org/pdf/1702.05970.pdf  
[2] https://arxiv.org/pdf/1505.04597.pdf  
[3] https://arxiv.org/pdf/1904.08128.pdf
        