import sys
import os

import time
import pprint
from LandmarkDataUnit import LandmarkDataUnit
from BBox import BBox

import caffe
import dlib
import cv2
import numpy as np
import math

def GetImgPaths(parent):
    img_path_list = []
    files = os.listdir(parent)
    for f in files:
        if os.path.isfile(os.path.join(parent, f)) and (f.endswith('.png') or f.endswith('.jpg')):
            img_path_list.append(f)
    return img_path_list

python_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(python_path, '..')
zoo_path = os.path.join(root_path, 'zoo')
network_path = os.path.join(zoo_path, '12_layer_deploy.prototxt')
weight_path = os.path.join(zoo_path, '12_layer_weights.caffemodel')
images_path = os.path.join(root_path, 'images')
results_path = os.path.join(root_path, 'LDU_test')
img_path = os.path.join(images_path, '31_Waiter_Waitress_Waiter_Waitress_31_484.jpg')

net = cv2.dnn.readNetFromCaffe(network_path, weight_path)
net_input_size = (80, 80)
detector = dlib.get_frontal_face_detector()

img = cv2.imread(img_path)
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(os.path.join(results_path, 'raw.png'), img)
dets = detector(img, 0)

LDUs = []

for index, det in enumerate(dets):
    bbox = BBox((det.left(), det.top(), det.right(), det.bottom()))
    LDU = LandmarkDataUnit(img, None, None, bbox)
    LDU.ClipBBox()
    
    img_cropped = LDU.CroppedImg()
    size = int(math.sqrt(img_cropped.shape[0]*img_cropped.shape[1]))
    cv2.imwrite(os.path.join(results_path, str(index) + '_cropped.png'), img_cropped)
    
    img_cropped = cv2.resize(img_cropped, net_input_size, interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.resize(img_cropped, (40, 40), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.resize(img_resized, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(results_path, str(index) + '_resized.png'), img_resized)


    blob = cv2.dnn.blobFromImage(img_cropped, 1.0, net_input_size, )
    net.setInput(blob)
    out = net.forward()

    out = np.reshape(out[0], (len(out[0])/2, 2))
    LDU.landmarks_bbox = out
    LDU.ProjectBBoxLandmarksToImg()
    LDUs.append(LDU)
    # LDU.DrawBBox((0, 255, 0))
    # cv2.imwrite
    # LDU.DrawLandmarks((0, 0, 255))

    LDU_cropped = LandmarkDataUnit(img_resized, None, out, BBox((0, 0, size, size)))
    LDU_cropped.ProjectBBoxLandmarksToImg()
    LDU_cropped.DrawLandmarks((0, 0, 255))
    cv2.imwrite(os.path.join(results_path, str(index) + '_landmarks.png'), img_resized)

for LDU in LDUs:
    LDU.DrawBBox((0, 255, 0))

cv2.imwrite(os.path.join(results_path, 'bboxes.png'), img)

for LDU in LDUs:
    LDU.DrawLandmarks((0, 0, 255))

cv2.imwrite(os.path.join(results_path, 'processed.png'), img)



