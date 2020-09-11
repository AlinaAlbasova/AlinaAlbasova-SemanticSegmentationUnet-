import os
import cv2
import numpy as np
from albumentations import (
    Compose, HorizontalFlip, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate
)

from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

from DataGen import TrainImageGenerator
from Metrics import dice_coef
from Model import unet

TRAIN_PATH = 'input/stage1_train/'


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Get train IDs
train_ids = next(os.walk(TRAIN_PATH))[1]

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    # RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    # RandomBrightness(limit=0.2, p=0.5),
    # HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
    #                    val_shift_limit=10, p=.9),
    # CLAHE(p=1.0, clip_limit=2.0),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255)
])

AUGMENTATIONS_TEST = Compose([
    # CLAHE(p=1.0, clip_limit=2.0),
    ToFloat(max_value=255)
])

# Get train IDs
ids = next(os.walk(TRAIN_PATH))[1]

train_ids, val_ids = train_test_split(ids, test_size=0.1)


train_gen = TrainImageGenerator(train_ids, augmentations=AUGMENTATIONS_TRAIN)
val_gen = TrainImageGenerator(train_ids, augmentations=AUGMENTATIONS_TEST)
#
# train_gen = TrainImageGenerator(train_ids)
# val_gen = TrainImageGenerator(train_ids)

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

model = unet(inputs, 'binary_crossentropy', dice_coef)

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-1.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

results = model.fit(train_gen, validation_data=val_gen, epochs=50,
                    callbacks=[earlystopper, checkpointer, reduce_lr])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
ax1.plot(results.history['loss'], '-', label = 'Loss')
ax1.plot(results.history['val_loss'], '-', label = 'Validation Loss')
ax1.legend()

ax2.plot(np.array(results.history['dice_coef']), '-',  label = 'dice_coef')
ax2.plot(np.array(results.history['val_dice_coef']), '-', label = 'Validation dice_coef')
ax2.legend()
model.save('best_model.h5')