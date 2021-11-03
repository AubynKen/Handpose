import sys

sys.settrace

import numpy as np
import json
import cv2
import Config as cfg
import Dataset
import Model
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context
from tqdm import tqdm
import matplotlib.pyplot as plt

if True:
    with tf.device('/device:GPU:0'):
        model = Model.unetModel()
        model.compile(optimizer="Adam", loss="mse", run_eagerly=True)
        train_dataset = Dataset.FreiHANDDataset()
        validation_dataset = Dataset.FreiHANDDataset(range_imgs = cfg.range_imgs_test)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.PATH_SAVED,
                                                         save_weights_only=True,
                                                         verbose=1)

        results = model.fit(train_dataset, validation_data=validation_dataset, epochs=cfg.workers, verbose=1, workers=8, callbacks=[cp_callback])


if False:
    with open(cfg.PATH_TRAIN_XYZ) as x, open(cfg.PATH_TRAIN_K) as k:
        labels_list = json.load(x)
        K_list = json.load(k)

    def projectPoints(xyz, K):
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return np.clip(uv[:, :2] / uv[:, -1:], a_min=None, a_max=cfg.original_height)

    labels = [projectPoints(xyz, k) for xyz, k in zip(labels_list, K_list)]
    #for img_id, xyz in enumerate(labels_list):
    #pos2d = self.projectPoints(xyz, K_list[img_id])
    #labels_dict[img_id] = pos2d*scaling_factor

    img1 = np.array(labels[19770]).T

    fig, ax = plt.subplots()
    img = cv2.imread('/Users/shawnsidbon/Documents/Learning/projects-venv/HandPose/FreiHAND_pub_v2/training/rgb/00019770.jpg')
    ax.imshow(img)
    ax.scatter(img1[0], img1[1])
    plt.show()
