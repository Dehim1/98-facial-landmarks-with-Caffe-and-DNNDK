from LandmarkDataUnit import LandmarkDataUnit
import numpy as np
import cv2
import os
import copy
import h5py
import sklearn.utils
import random
# import resource
import threading
import math

def GetAvailableMemory():
    with open('/proc/meminfo', 'r') as mem:
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemAvailable:'):
                availableMemory = float(sline[1])
    return availableMemory/1024.0/1024.0

def GetData_300W(filepath):
    ...

def GetData_WFLW(filepath):
    dirname = os.path.dirname(filepath)
    f = open(filepath, 'r')
    data = []
    for line in f.readlines():
        s = line.strip().split(' ')
        imgPath = os.path.join(dirname, s[206].replace('\\', '/'))
        landmarks = np.zeros((98,2))
        for i in range(0,98):
            landmarks[i] = (float(s[i*2]), float(s[i*2+1]))
        data.append((imgPath, landmarks))
    return data

def WriteHDF5(F_images, F_landmarks, i, maxFileSize, h5_dir, h5_prefix, txt_file):
    F_images, F_landmarks = sklearn.utils.shuffle(F_images, F_landmarks, random_state=42)
    while(len(F_landmarks) != 0):
        h5_file = os.path.join(h5_dir, h5_prefix + '_' + str(i) + '.h5')
        print('Writing ' + h5_file)

        with h5py.File(h5_file, 'w') as f:
            f['X'] = F_images[:maxFileSize].astype(np.uint8)
            f['landmarks'] = F_landmarks[:maxFileSize].astype(np.float32)
        with open(txt_file, 'a') as f:
            f.write(h5_file + '\n')

        F_images = F_images[maxFileSize:]
        F_landmarks = F_landmarks[maxFileSize:]
        i = i+1

    return F_images, F_landmarks, i

class GenerateDataThread(threading.Thread):
    def __init__(self, threadID, data, augmentationRange, imgSize, F_images, F_landmarks):
        threading.Thread.__init__(self)
        self.threadID = copy.deepcopy(threadID)
        self.data = copy.deepcopy(data)
        self.augmentationRange = copy.deepcopy(augmentationRange)
        self.imgSize = copy.deepcopy(imgSize)
        self.F_images = F_images
        self.F_landmarks = F_landmarks
    def run(self):
        print('Starting thread ' + str(self.threadID))

        # Create empty numpy arrays to hold dataset images and annotations
        i = 0
        # Loop over every entry in data. Each entry contains a string containing the path to the image as well as a numpy array containing the landmarks.
        for (imgPath, landmarks) in self.data:
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
            img = cv2.imread(imgPath)
            LDU_aug = LandmarkDataUnit(img, landmarks)

            # Augment the current data entry and append the result to F_images and F_landmarks. 
            # LDU_aug = copy.deepcopy(LDU)
            if mirror:
                LDU_aug.Mirror()
            LDU_aug.Rotate(angle)
            LDU_aug.BBoxFromLandmarks()
            LDU_aug.TranslateBBox(translate)
            LDU_aug.ScaleBBox(scale)
            LDU_aug.ClipBBox()
            LDU_aug.ProjectImgLandmarksToBBox()
            img_aug = LDU_aug.BBoxCroppedImg()
            img_aug = cv2.resize(img_aug, self.imgSize)
            img_aug = cv2.split(img_aug) #Split the different color channels of the image. This changes the shape from (imgSize[1], imgSize[0], 3) to (3, imgSize[1], imgSize[0]). This is required for Caffe.
            img_aug = np.reshape(np.asarray(img_aug), (3, self.imgSize[1], self.imgSize[0]))
            lm_aug = np.reshape(LDU_aug.landmarks_bbox, ((LDU_aug.landmarks_bbox.size)))
            self.F_images[i] = img_aug
            self.F_landmarks[i] = lm_aug
            i = i+1
            # Print length of F_landmarks if it's divisible by 1000 to indicate progress.
            if(i%500 == 0):
                print('thread ' + str(self.threadID) + ' length: ' + str(i))


def GenerateDataset(data, N_threads_max, N_augmentations, augmentationRange, imgSize, maxFileSize, h5_dir, h5_prefix):
    txt_file = os.path.join(h5_dir, h5_prefix + '.txt')
    open(txt_file, 'w').close()
    N_landmarks = len(data[0][1])
    
    N_iters = int(math.ceil(float(N_augmentations) / float(N_threads)))
    N_threads_0 = int(math.ceil(float(N_augmentations) / float(N_iters)))
    rem = ((N_iters)*N_threads_0) - N_augmentations
    iters = []
    for _ in range(N_iters):
        iters.append(N_threads_0)
    for i in range(rem):
        iters[len(iters)-1-i] = iters[len(iters)-1-i] - 1
    print(iters)
    print(len(data))
    i = 0

    for N_threads_iter in iters:
        F_images = np.zeros((N_threads_iter*len(data), 3, imgSize[1], imgSize[0]), np.uint8)
        F_landmarks = np.zeros((N_threads_iter*len(data), 2*N_landmarks), np.float32)
        print(F_images.shape)
        print(F_landmarks.shape)

        threads = []

        for tid in range(N_threads_iter):
            threads.append(GenerateDataThread(tid, data, augmentationRange, imgSize, F_images[tid*len(data):(tid+1)*len(data), :], F_landmarks[tid*len(data):(tid+1)*len(data), :]))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        F_images, F_landmarks, i = WriteHDF5(F_images, F_landmarks, i, maxFileSize, h5_dir, h5_prefix, txt_file)

#Get dataset
dataset_dir = '/home/dehim/Downloads/datasets/WFLW/WFLW_images'
h5_dir = '/home/dehim/Downloads/datasets/WFLW/WFLW_h5/h5_aug_thread_4'
prefix_test = 'test_aug'
prefix_train = 'train_aug'

imgSize = (80, 80)
maxFileSize = 7500

N_threads = 4
N_testAugmentations = 9
N_trainAugmentations = 80

mirror = True
angleRange = (-55.0, 55.0)
xTranslateRange = (-0.3, 0.3)
yTranslateRange = (-0.3, 0.3)
xScaleRange = (0.75, 1.6)
yScaleRange = (0.75, 1.6)
augmentationRange = (mirror, angleRange, (xTranslateRange, yTranslateRange), (xScaleRange, yScaleRange))

test_data = GetData_WFLW(os.path.join(dataset_dir, 'list_98pt_rect_attr_test.txt'))
train_data = GetData_WFLW(os.path.join(dataset_dir, 'list_98pt_rect_attr_train.txt'))
GenerateDataset(test_data, N_threads, N_testAugmentations, augmentationRange, imgSize, maxFileSize, h5_dir, prefix_test)
GenerateDataset(train_data, N_threads, N_trainAugmentations, augmentationRange, imgSize, maxFileSize, h5_dir, prefix_train)