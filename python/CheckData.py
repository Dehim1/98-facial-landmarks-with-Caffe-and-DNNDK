from LandmarkDataUnit import LandmarkDataUnit
import numpy as np
import cv2
import os
import copy
import h5py
import random
# import resource

from BBox import BBox 

def ReadHDF5(i, h5_dir, h5_prefix):
    h5_file = os.path.join(h5_dir, h5_prefix + '_' + str(i) + '.h5')
    F_data = h5py.File(h5_file, 'r')
    return F_data

h5_dir = '/home/dehim/Downloads/datasets/landmark_h5'
prefix_test = 'test_aug'
prefix_train = 'train_aug'
F_data = ReadHDF5(0, h5_dir, prefix_train)

i = 21
for d in F_data:
    print(d)
img = F_data['data'][i]
img = cv2.merge(img)
if F_data['lossmult_98'][i] != 0.0:
    F_landmarks = F_data['landmarks_98'][i]
else:
    F_landmarks = F_data['landmarks_68'][i]
lm = F_landmarks
lm = np.reshape(lm, (len(F_landmarks)//2, 2))
bbox = BBox(np.array([0, 0, 400, 400]))

img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
ldu = LandmarkDataUnit(img, None, lm, bbox)
ldu.ProjectBBoxLandmarksToImg()
ldu.DrawLandmarks((0, 0, 255))

cv2.imshow('image', img)
cv2.waitKey(0)