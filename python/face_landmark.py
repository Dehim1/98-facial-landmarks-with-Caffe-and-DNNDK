import sys
import os

import time
import pprint

# import caffe
import dlib
import cv2
import numpy as np

def file_list_fn(path):

    file_list = []
    files = os.listdir(path)
    for f in files:
        file_list.append(f)
    return file_list

install_path = os.environ['HOME'] + '/Downloads/Neural_nets/98_Landmark_CNN'
network_path = install_path + '/ZOO/vanilla_deploy_relu_no_pool.prototxt'
weight_path = install_path + '/caffeData/snapshots/relu_no_pool_aug/snap_iter_20000.caffemodel'
images_dir = install_path + '/images/'
result_dir = install_path + '/results/'

#MEAN_TRAIN_SET = cv2.imread(install_path + '/trainMean.png').astype('f4')
#STD_TRAIN_SET  = cv2.imread(install_path + '/trainSTD.png').astype('f4')

image_list = file_list_fn(images_dir)
net = cv2.dnn.readNetFromCaffe(network_path, weight_path)

detector = dlib.get_frontal_face_detector()

total_detecting_time = 0.0
total_landmark_time = 0.0
face_total = 0.0
for image in image_list:
    print("Processing file: {}".format(image))
    img = cv2.imread(images_dir + image)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    det_start_time = time.time()
    dets = detector(img, 1)
    det_end_time = time.time()
    det_time = det_end_time - det_start_time
    total_detecting_time += det_time
    print "Detecting time is {}".format(det_time)
    print "Number of faces detected: {}".format(len(dets))
    for i, d in enumerate(dets):
            print "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom())

    for index, det in enumerate(dets):
        face_total += 1
        x1 = det.left()
        y1 = det.top()
        x2 = det.right()
        y2 = det.bottom()
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img.shape[1]: x2 = img.shape[1]
        if y2 > img.shape[0]: y2 = img.shape[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        roi = img[y1:y2 + 1, x1:x2 + 1, ]
        w = 80
        h = 80

        print image
        res = cv2.resize(roi, (w, h), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
        #new_img = (res - MEAN_TRAIN_SET) / (0.000001 + STD_TRAIN_SET)

        #blob = cv2.dnn.blobFromImage(new_img, 1.0, (w, h), )
        blob = cv2.dnn.blobFromImage(res, 1.0, (w, h), )
        net.setInput(blob)

        landmark_time_start = time.time()
        out = net.forward()
        landmark_time_end = time.time()

        landmark_time = landmark_time_end - landmark_time_start
        total_landmark_time += landmark_time
        print "landmark time is {}".format(landmark_time)

        point_pair_l = len(out[0])
        for i in range(point_pair_l / 2):
            x = out[0][2*i] * (x2 - x1) + (x2 + x1)/2
            y = out[0][2*i+1] * (y2 - y1) + (y2 + y1)/2
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)

    cv2.imwrite(result_dir + image, img)
    print("Writing image to " + result_dir + image)

print total_detecting_time
print total_landmark_time
print face_total
per_face_det_time = total_detecting_time / (0.000001 + face_total)
per_face_landmark_time = total_landmark_time / (0.000001 + face_total)

per_image_det_time = total_detecting_time / (0.000001 + len(image_list))
per_image_landmark_time = total_landmark_time / (0.000001 + len(image_list))

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print "per face detecting time is {}".format(per_face_det_time)
print "per face landmark time is {}".format(per_face_landmark_time)
print "per image detecting time is {}".format(per_image_det_time)
print "per image detecting time is {}".format(per_image_landmark_time)



