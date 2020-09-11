import os
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.python.keras.utils.data_utils import Sequence

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = 'Input/stage1_train/'


class TrainImageGenerator(Sequence):
    def __init__(self, train_ids,augmentations=None, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.train_ids = train_ids
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        return len(self.train_ids) // self.batch_size

    def get_mask(self, mask_path):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(mask_path))[2]:
            mask_ = imread(mask_path + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        return mask

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.train_ids[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.train_ids))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        images = np.zeros((len(batch), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        masks = np.zeros((len(batch), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

        for idx, id_ in enumerate(batch):
            path = TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            # images[idx] = self.augment(**img)
            mask = self.get_mask(path + '/masks/')
            if self.augment is not None:
                transformed = self.augment(image=img, mask=mask)
                images[idx] = transformed["image"]
                masks[idx] = transformed["mask"]

            # masks[idx] = self.augment(**mask)
            # images[idx] = img
            # masks[idx] = self.get_mask(path + '/masks/')

        return images, masks







