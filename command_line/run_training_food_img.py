import gc
import datetime

import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

import cv2
import matplotlib
import matplotlib.pyplot as plt

import sys
import os
if os.name == 'nt':
    path_prefix = 'D:/workspace'
else:
    path_prefix = '/mnt/d/workspace'
    #path_prefix = '/workspace'
sys.path.append('{}/PConv-Keras'.format(path_prefix))

from libs.pconv_model import PConvUnet
from libs.util import random_mask

plt.ioff()

# SETTINGS
TRAIN_DIR = r"{}/PConv-Keras/data/food_images_set/train".format(path_prefix)
VAL_DIR = r"{}/PConv-Keras/data/food_images_set/validation".format(path_prefix)
TEST_DIR = r"{}/PConv-Keras/data/food_images_set/validation".format(path_prefix)

BATCH_SIZE = 16


class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            # Get augmented image samples
            ori = next(generator)

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori


# Create training generator
train_datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(256, 256), batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = DataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(256, 256), batch_size=BATCH_SIZE, seed=1
)

# Instantiate the model
model = PConvUnet(weight_filepath="{}/PConv-Keras/data/model/".format(path_prefix))
# Run training for certain amount of epochs
model.fit(
    train_generator,
    steps_per_epoch=10,
    validation_data=val_generator,
    validation_steps=100,
    epochs=5,
    plot_callback=None,
    callbacks=[
        TensorBoard(log_dir="{}/PConv-Keras/data/model/initial_training".format(path_prefix), write_graph=False)
    ]
)

