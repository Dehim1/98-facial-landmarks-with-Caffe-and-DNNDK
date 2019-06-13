from LandmarkDataUnit import LandmarkDataUnit
import numpy as np
import cv2
import sys
import os
import copy
import h5py
import sklearn.utils
import random
import threading
import math
import GetData

def WriteHDF5(F_data, i, maxFileSize, h5_dir, h5_prefix, txt_file):
    #13.
    for d in F_data:
        F_data[d] = sklearn.utils.shuffle(F_data[d], random_state=i)
    #14.
    while(len(F_data['data']) != 0):
        h5_path = os.path.join(h5_dir, h5_prefix + '_' + str(i) + '.h5')
        print('Writing ' + h5_path)

        with h5py.File(h5_path, 'w') as h5_file:
            for d in F_data:
                h5_file[d] = F_data[d][:maxFileSize]
        with open(txt_file, 'a') as f:
            f.write(h5_path + '\n')
        
        for d in F_data:
            F_data[d] = F_data[d][maxFileSize:]
        i = i+1

    return F_data, i

class AugmentDataThread(threading.Thread):
    def __init__(self, threadID, data, augmentationRange, imgSize, F_data):
        threading.Thread.__init__(self)
        self.threadID = copy.deepcopy(threadID)
        self.data = copy.deepcopy(data)
        self.augmentationRange = copy.deepcopy(augmentationRange)
        self.imgSize = copy.deepcopy(imgSize)
        self.F_data = F_data
    def run(self):
        print('Starting thread ' + str(self.threadID))

        i = 0
        # Loop over every entry in data. Each entry contains a string containing the path to the image as well as a numpy array containing the landmarks.
        for d in range(len(self.data)):
            N_landmarks = len(self.data[d][0][1])
            for (imgPath, landmarks, attributes) in self.data[d]:
                # Randomly initialize augmentation parameters in a range defined by augmentationRange.
                # This assures every data entry will be augmented differently.
                if self.augmentationRange[0]:
                    mirror = random.getrandbits(1)
                else:
                    mirror = False
                angle = random.uniform(self.augmentationRange[1][0], self.augmentationRange[1][1])
                xTranslate = random.uniform(self.augmentationRange[2][0][0], self.augmentationRange[2][0][1])
                yTranslate = random.uniform(self.augmentationRange[2][1][0], self.augmentationRange[2][1][1])
                translate = (xTranslate, yTranslate)
                xScale = random.uniform(self.augmentationRange[3][0][0], self.augmentationRange[3][0][1])
                yScale = random.uniform(self.augmentationRange[3][1][0], self.augmentationRange[3][1][1])
                scale = (xScale, yScale)

                # Load image into RAM and generate a LandmarkDataUnit object. This class defines functions required to process the dataset including data augmentation.
                img = cv2.imread(imgPath) #1.
                LDU_aug = LandmarkDataUnit(img, landmarks) #1.

                # Augment the current data entry and append the result to F_images and F_landmarks. 
                # LDU_aug = copy.deepcopy(LDU)
                LDU_aug.Rotate(angle) #2.
                LDU_aug.BBoxFromLandmarks() #3.
                LDU_aug.TranslateBBox(translate) #4.
                LDU_aug.ScaleBBox(scale) #5.
                LDU_aug.ClipBBox() #6.
                LDU_aug.ProjectImgLandmarksToBBox() #7.
                LDU_aug.CropImg() #8.
                LDU_aug.ResizeImg(self.imgSize) #9.
                #10.
                if mirror:
                    LDU_aug.Mirror()
                IOD = LDU_aug.CalcInterocularDistance()
                # LDU_aug.CalcLossMult() #11.
                img_aug = LDU_aug.img 
                img_aug = cv2.split(img_aug) #Split the different color channels of the image. This changes the shape from (imgSize[1], imgSize[0], 3) to (3, imgSize[1], imgSize[0]). This is required for Caffe.
                img_aug = np.reshape(np.asarray(img_aug), (3, self.imgSize[1], self.imgSize[0]))
                landmarks_aug = np.reshape(LDU_aug.landmarks_bbox, ((LDU_aug.landmarks_bbox.size)))
                # lossmult_aug = np.reshape(LDU_aug.lossmult, ((LDU_aug.lossmult.size)))
                #12.
                self.F_data['data'][i] = img_aug
                self.F_data['lossweight'][i] = 1.0/IOD
                self.F_data['landmarks_' + str(N_landmarks)][i] = landmarks_aug
                # self.F_data['lossmult_' + str(N_landmarks)][i] = lossmult_aug
                self.F_data['lossgate_' + str(N_landmarks)][i] = 1.0
                if not(attributes is None):
                    self.F_data['attributes_' + str(N_landmarks)][i] = attributes
                i = i+1
                # Print length of F_landmarks if it's divisible by 1000 to indicate progress.
                if(i%100 == 0):
                    print('thread ' + str(self.threadID) + ' length: ' + str(i))


def GenerateDataset(data, N_threads_max, N_augmentations, augmentationRange, imgSize, maxFileSize, h5_dir, h5_prefix):
    txt_file = os.path.join(h5_dir, h5_prefix + '.txt')
    open(txt_file, 'w').close()
    
    data_length = 0
    for d in data:
        data_length += len(d)
    
    N_iters = int(math.ceil(float(N_augmentations) / float(N_threads)))
    N_threads_0 = int(math.ceil(float(N_augmentations) / float(N_iters)))
    rem = ((N_iters)*N_threads_0) - N_augmentations
    iters = []
    for _ in range(N_iters):
        iters.append(N_threads_0)
    for i in range(rem):
        iters[len(iters)-1-i] = iters[len(iters)-1-i] - 1
    print(iters)
    
    ###
    reference_landmarks = GetData_2.GetReferenceLandmarks_98()
    ###
    i = 0
    for N_threads_iter in iters:
        F_data = {}
        F_data['data'] = np.zeros((N_threads_iter*data_length, 3, imgSize[1], imgSize[0]), np.uint8)
        F_data['lossweight'] = np.zeros((N_threads_iter*data_length), np.float32)
        for d in data:
            N_landmarks = len(d[0][1])
            F_data['landmarks_' + str(N_landmarks)] = np.zeros((N_threads_iter*data_length, 2*N_landmarks), np.float32)
            F_data['lossgate_' + str(N_landmarks)] = np.zeros((N_threads_iter*data_length), np.uint8)
            if not(d[0][2] is None):
                N_attributes = len(d[0][2])
                F_data['attributes_' + str(N_landmarks)] = np.zeros((N_threads_iter*data_length, N_attributes), np.uint8)
        ###
        F_data['reference_landmarks_98'] = np.empty((N_threads_iter*data_length, 196), np.float32)
        F_data['reference_landmarks_98'][:] = np.reshape(reference_landmarks, (196))
        ###

        threads = []

        for tid in range(N_threads_iter):  
            F_data_thread = {}
            for d in F_data:
                F_data_thread[d] = F_data[d][tid*data_length:(tid+1)*data_length, ...]

            threads.append(AugmentDataThread(tid, data, augmentationRange, imgSize, F_data_thread))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        F_data, i = WriteHDF5(F_data, i, maxFileSize, h5_dir, h5_prefix, txt_file) #13. and 14.

#Get dataset
wflw_dir = '/home/dehim/Downloads/datasets/WFLW/WFLW_images'
ibug_dir = '/home/dehim/Downloads/datasets/68_landmark'
h5_dir = '/home/dehim/Downloads/datasets/landmark_h5'
prefix_test = 'test_aug'
prefix_train = 'train_aug'

imgSize = (80, 80)
maxFileSize = 7500

N_threads = 4
N_testAugmentations = 8
N_trainAugmentations = 128

mirror = True
angleRange = (-55.0, 55.0)
xTranslateRange = (-0.35, 0.35)
yTranslateRange = (-0.35, 0.35)
xScaleRange = (0.8, 1.8)
yScaleRange = (0.8, 1.8)
augmentationRange = (mirror, angleRange, (xTranslateRange, yTranslateRange), (xScaleRange, yScaleRange))

test_data = []
train_data = []

test_data.append(GetData.GetData_98(os.path.join(wflw_dir, 'list_98pt_rect_attr_test.txt')))
train_data.append(GetData.GetData_98(os.path.join(wflw_dir, 'list_98pt_rect_attr_train.txt')))
train_data.append(GetData.GetData_68(ibug_dir))

GenerateDataset(test_data, N_threads, N_testAugmentations, augmentationRange, imgSize, maxFileSize, h5_dir, prefix_test)
GenerateDataset(train_data, N_threads, N_trainAugmentations, augmentationRange, imgSize, maxFileSize, h5_dir, prefix_train)