# Semantic segmentation
## Overview
### Data
##### The original dataset is from [2018 Data Science Bowl kaggle competition](https://www.kaggle.com/c/data-science-bowl-2018). It was downloaded and preprocessed.
### Model
##### The provided model has an U-Net like architecture. It is based on [U-Net: Convolutional Networks for Biomedical Image Segmentation ](https://arxiv.org/pdf/1505.04597.pdf).
![U-Net architecture](http://robocraft.ru/files/neuronet/u-net/u-net-architecture.png)
### Train
#### Data Generator for train data
##### Sometimes, it is impossible to fit all data into memory. In this case, it is good idea to use data generator. Here, I created custom image data generator by using keras class Sequence. I also added some augmentations with library 'albumentations'. 
#### Model
##### As a model Unet was used. Additionally, I added Batch normalization and Droupouts to avoid/reduce overfitting.
#### Callbacks
* Early stopping - combats overfitting interrupting the training procedure once model’s performance on a validation set gets worse. A validation set is a set of examples that we never use for gradient descent, but which is also not a part of the test set. The validation examples are considered to be representative of future test examples. Early stopping is effectively tuning the hyper-parameter number of epochs/steps.
* ReduceLROnPlateau - reduce learning rate when a metric has stopped improving.
##### As function of loss was chosen binary cross-entropy. Not best choice for image segmentation. Better choice is Tversky Loss (It takes into account imbalance of dataset).
##### As metric was chosen Dice coefficient
## How to use
### To run training use script Train.py
### To run inference on test date use script Predict-masks.py
### To see some insights about dataset use notebook Exploratory_analysis.ipynb
## Results
Dice coefficient was nearly 0.2. 
![Result Example](https://github.com/AlinaAlbasova/semantic_segmentation/blob/master/some_results/result1.png)
