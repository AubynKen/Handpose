import numpy as np
import json
import Config as cfg
import tensorflow as tf
from tensorflow import keras
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


class FreiHANDDataset(keras.utils.Sequence):

    def __init__(self, batch_size=cfg.batch_size, img_paths=cfg.PATH_TRAIN, xyz_path=cfg.PATH_TRAIN_XYZ, K_path=cfg.PATH_TRAIN_K, \
                 img_height=cfg.img_height, img_width=cfg.img_width, img_scaling_factor=cfg.img_scaling, MEAN = cfg.MEAN, STD = cfg.STD,\
                 range_imgs = cfg.range_imgs_train):
        self.img_paths_list = [filename for filename in os.listdir(img_paths)[min(range_imgs):max(range_imgs)]]
        self.labels = self.get_label(xyz_path=xyz_path, K_path=K_path, scaling_factor = cfg.position_scaling)
        for i, lab in enumerate(self.labels):
            for ent in lab:
                if ent[0] > 128 or ent[1] > 128:
                    print(i, ent)
                    err=err


        self.batch_size = batch_size
        self.img_path = img_paths
        self.img_height = img_height
        self.img_width = img_width
        self.img_scaling_factor = img_scaling_factor
        self.MEAN = MEAN
        self.STD = STD
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return len(self.img_paths_list) // self.batch_size

    def __getitem__(self, idx):

        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batchImPaths = [self.img_paths_list[k] for k in indexes]
        batchLabels = [self.labels[k] for k in indexes]

        batchImgs = [self.decode_img(im) for im in batchImPaths]

        batchLabelsTransformed = [self.decode_labels(lab) for lab in batchLabels]
        batchLabelsTransformed = tf.reshape(batchLabelsTransformed, [self.batch_size, self.img_height, self.img_width, 21])

        return np.array(batchImgs), np.array(batchLabelsTransformed)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths_list))
        #if self.shuffle == True:
            #np.random.shuffle(self.indexes)

    def calc_mean_std(self):
        count = 0
        means = 0
        stdev = 0

        pbar = tqdm(self.img_paths_list)

        for img in pbar:
            img = self.img_path + '/' + img
            img = cv2.imread(img)
            img = tf.image.resize(img, [self.img_height, self.img_width])
            img /= self.img_scaling_factor
            img = tf.reshape(img, [-1, 3])
            count += 1
            means += np.mean(img, axis=0)
            stdev += np.std(img, axis=0)

        return means / count, stdev / count

    def get_label(self, xyz_path=cfg.PATH_TRAIN_XYZ, K_path=cfg.PATH_TRAIN_K, scaling_factor=cfg.position_scaling):
        #labels_dict = {}
        with open(xyz_path) as x, open(K_path) as k:
            labels_list = json.load(x)
            K_list = json.load(k)

        labels = [self.projectPoints(xyz, k)*scaling_factor for xyz, k in zip(labels_list, K_list)]
        #for img_id, xyz in enumerate(labels_list):
            #pos2d = self.projectPoints(xyz, K_list[img_id])
            #labels_dict[img_id] = pos2d*scaling_factor

        return labels

    def projectPoints(self, xyz, K):
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return np.clip(uv[:, :2] / uv[:, -1:], a_min=None,a_max=cfg.original_height-1)

    def decode_img(self, img, debug=False):
        img = self.img_path+'/'+ img
        img = cv2.imread(img)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img /= self.img_scaling_factor
        img = (img - self.MEAN)/self.STD

        if debug == True:

            cv2.imshow('image', np.uint8((self.STD*img + self.MEAN)*self.img_scaling_factor))

            plt.imshow(img)
            plt.show()

            key = cv2.waitKey(3000)  # pauses for 3 seconds before fetching next image
            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()

            err = err

        return img

    def decode_labels(self, label):
        label = np.array(label).astype(int)
        zeros = [np.zeros((self.img_height, self.img_width))]*21


        for i, pos in enumerate(label):
            zeros[i][pos[0]][pos[1]] = 1
            zeros[i] = cv2.GaussianBlur(src = zeros[i], ksize = (5,5), sigmaX = 1, sigmaY = 1)
            zeros[i] /= np.max(zeros[i])
        return zeros