# Semantic segmentation
## Overview
### Data
##### The original dataset is from [2018 Data Science Bowl kaggle competition](https://www.kaggle.com/c/data-science-bowl-2018). It was downloaded and preprocessed.
### Model
##### The provided model has an U-Net like architecture. It is based on [U-Net: Convolutional Networks for Biomedical Image Segmentation ](https://arxiv.org/pdf/1505.04597.pdf).
![U-Net architecture](http://robocraft.ru/files/neuronet/u-net/u-net-architecture.png)
### Train
##### As we have small dataset I decided to use regularization to overcome overfitting. As regularization techniques were chosen:
* Dropout - At each training iteration a dropout layer randomly removes some nodes in the network along with all of their incoming and outgoing connections.
* Early stopping - combats overfitting interrupting the training procedure once model’s performance on a validation set gets worse. A validation set is a set of examples that we never use for gradient descent, but which is also not a part of the test set. The validation examples are considered to be representative of future test examples. Early stopping is effectively tuning the hyper-parameter number of epochs/steps.
##### As function of loss was chosen binary cross-entropy
##### As metric was chosen Dice coefficient
## How to use
### To run training use script Train.py
### To run inference on test date use script Predict-masks_v1.1.py
### To see some insights about dataset use notebook Exploratory_analysis.ipynb
## Results
![Result Example](https://github.com/AlinaAlbasova/semantic_segmentation/blob/master/some_results/result1.png)
