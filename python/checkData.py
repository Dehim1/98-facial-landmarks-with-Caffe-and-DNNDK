from LandmarkDataUnit import LandmarkDataUnit
import numpy as np
import cv2
import os
import copy
import h5py
import sklearn.utils
import random
import resource

from BBox import BBox 

def ReadHDF5(i, h5_dir, h5_prefix):
    h5_file = os.path.join(h5_dir, h5_prefix + '_' + str(i) + '.h5')
    f = h5py.File(h5_file, 'r')

    F_images = f['X']
    F_landmarks = f['landmarks']
    return F_images, F_landmarks

h5_dir = '/home/dehim/Downloads/datasets/WFLW/WFLW_h5/h5_aug_thread_3'
prefix_test = 'test_aug'
prefix_train = 'train_aug'
F_images, F_landmarks = ReadHDF5(0, h5_dir, prefix_test)

i = 119

img = F_images[i]
img = cv2.merge(img)
lm = F_landmarks[i]
lm = np.reshape(lm, (98, 2))
bbox = BBox(np.array([0, 0, 80, 80]))

ldu = LandmarkDataUnit(img, None, lm, bbox)
ldu.ProjectBBoxLandmarksToImg()
# ldu.DrawLandmarks((0, 0, 255))

cv2.imshow('image', img)
cv2.waitKey(0)