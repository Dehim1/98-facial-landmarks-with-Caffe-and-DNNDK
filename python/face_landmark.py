import sys
import os

import time
from LandmarkDataUnit import LandmarkDataUnit
from BBox import BBox

import caffe
import dlib
import cv2
import numpy as np

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
network_path = os.path.join(zoo_path, '15_layer_deploy.prototxt')
weight_path = os.path.join(zoo_path, '15_layer_weights.caffemodel')
images_path = os.path.join(root_path, 'images')
results_path = os.path.join(root_path, 'results')

img_path_list = GetImgPaths(images_path)
net = cv2.dnn.readNetFromCaffe(network_path, weight_path)
net_input_size = (80, 80)
detector = dlib.get_frontal_face_detector()

total_detecting_time = 0.0
total_landmark_time = 0.0
face_total = 0.0

for img_path in img_path_list:
    print("Processing file: {}".format(img_path))
    img = cv2.imread(os.path.join(images_path, img_path))
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)

    det_start_time = time.time()
    dets = detector(img, 0)
    det_end_time = time.time()
    det_time = det_end_time - det_start_time
    total_detecting_time += det_time
    print("Detecting time is {}".format(det_time))
    print("Number of faces detected: {}".format(len(dets)))

    for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))

    for index, det in enumerate(dets):
        face_total += 1
        bbox = BBox((det.left(), det.top(), det.right(), det.bottom()))
        LDU = LandmarkDataUnit(img, None, None, bbox)
        LDU.ClipBBox()
        img_cropped = LDU.CroppedImg()
        img_cropped = cv2.resize(img_cropped, net_input_size, interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img_cropped, 1.0, net_input_size, )
        net.setInput(blob)

        landmark_time_start = time.time()
        out = net.forward()
        landmark_time_end = time.time()
        landmark_time = landmark_time_end - landmark_time_start
        total_landmark_time += landmark_time
        print("landmark time is {}".format(landmark_time))

        out = np.reshape(out[0], (len(out[0])/2, 2))
        LDU.landmarks_bbox = out
        LDU.ProjectBBoxLandmarksToImg()
        LDU.DrawBBox((0, 255, 0))
        LDU.DrawLandmarks((0, 0, 255))

    cv2.imwrite(os.path.join(results_path, img_path), img)
    print("Writing image to " + os.path.join(results_path, img_path))

print('total detecting time is ' + str(total_detecting_time))
print('total landmark time is ' + str(total_landmark_time))
print('total number of faces is ' + str(face_total))
per_face_det_time = total_detecting_time / (0.000001 + face_total)
per_face_landmark_time = total_landmark_time / (0.000001 + face_total)

per_image_det_time = total_detecting_time / (0.000001 + len(img_path_list))
per_image_landmark_time = total_landmark_time / (0.000001 + len(img_path_list))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("per face detecting time is {}".format(per_face_det_time))
print("per face landmark time is {}".format(per_face_landmark_time))
print("per image detecting time is {}".format(per_image_det_time))
print("per image detecting time is {}".format(per_image_landmark_time))



