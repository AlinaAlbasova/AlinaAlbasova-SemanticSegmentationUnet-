import os
import sys
from keras.models import load_model
from tqdm import tqdm
import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

from Metrics import dice_coef

import matplotlib.pyplot as plt

dependencies = {
    'dice_coef': dice_coef
}

model = load_model('model-1.h5', custom_objects=dependencies)
model.summary()

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TEST_PATH = 'Input/stage1_test/'

test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

for i in range(5):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(X_test[i])
    ax1.set_title('Test image')
    # plt.show()
    ax2.imshow(np.squeeze(preds_test_t[i]) ,)
    ax2.set_title('Predicted mask')
    plt.show()