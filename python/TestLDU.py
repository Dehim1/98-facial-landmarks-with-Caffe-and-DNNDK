from LandmarkDataUnit import LandmarkDataUnit
from BBox import BBox
import GetData
import os
import math
import copy
import numpy as np
import cv2

dataset_dir = '/home/dehim/Downloads/datasets/WFLW/WFLW_images'
output_dir = '/home/dehim/Downloads/Neural_nets/98-facial-landmarks-with-DNNDK/ldu_test'
data = GetData.GetData_98(os.path.join(dataset_dir, 'list_98pt_rect_attr_train.txt'))
i = 220
LDU = LandmarkDataUnit(cv2.imread(data[i][0]), data[i][1])
LDU.Scale((2.0, 2.0))

augmentations = []
# for angle in [-30.0, 0, 30.0]:
#     for x_translate in [-0.25, 0, 0.25]:
#         for y_translate in [-0.25, 0, 0.25]:
#             for x_scale in [0.7, 1.0, 1.3]:
#                 for y_scale in [0.7, 1.0, 1.3]:
#                     for mirror in[True, False]:
#                         augmentations.append((angle, (x_translate, y_translate), (x_scale, y_scale), mirror))
augmentations.append((-30.0, (0.0, 0.0), (1.0, 1.0), False))
augmentations.append((0.0, (0.0, 0.0), (1.0, 1.0), False))
augmentations.append((30.0, (0.0, 0.0), (1.0, 1.0), False))
augmentations.append((-30.0, (-0.25, 0.0), (1.0, 1.0), False))
augmentations.append((-30.0, (0.25, 0.0), (1.0, 1.0), False))
augmentations.append((-30.0, (0.0, -0.25), (1.0, 1.0), False))
augmentations.append((-30.0, (0.0, 0.25), (1.0, 1.0), False))
augmentations.append((-30.0, (0.25, 0.0), (0.7, 1.3), False))
augmentations.append((-30.0, (0.25, 0.0), (1.3, 0.7), False))
augmentations.append((-30.0, (0.25, 0.0), (1.3, 0.7), True))



for (angle, translate, scale, mirror) in augmentations:
    LDU_aug = copy.deepcopy(LDU)
    LDU_aug.Rotate(angle)
    LDU_aug.BBoxFromLandmarks()
    LDU_aug.TranslateBBox(translate)
    LDU_aug.ScaleBBox(scale)
    LDU_aug.ClipBBox()
    LDU_aug.ProjectImgLandmarksToBBox()

    midfix = '_angle_' + str(int(angle)) + '_translate_' + str(int(translate[0]*100.0)) + 'x_' + str(int(translate[1]*100.0)) + 'y_scale_' + str(int(scale[0]*100.0)) + 'x_' + str(int(scale[1]*100.0)) + 'y_mirror_' + str(mirror).lower() + '_'
    
    LDU_crop = copy.deepcopy(LDU_aug)
    LDU_crop.Crop()
    LDU_resize = copy.deepcopy(LDU_crop)
    LDU_crop.DrawLandmarks((0, 0, 255))
    cv2.imwrite(os.path.join(output_dir, 'ldu' + midfix + 'crop.png'), LDU_crop.img)
    
    (h, w) = LDU_resize.img.shape[:2]
    size = int(math.sqrt(h*w))
    LDU_resize.Resize((40, 40))
    LDU_resize.Resize((size, size), cv2.INTER_NEAREST)
    if mirror:
        LDU_resize.Mirror()
    LDU_resize.ProjectBBoxLandmarksToImg()
    LDU_resize.DrawLandmarks((0, 0, 255))
    cv2.imwrite(os.path.join(output_dir, 'ldu' + midfix + 'resize.png'), LDU_resize.img)
    
    LDU_aug.DrawLandmarks((0, 0, 255))
    cv2.imwrite(os.path.join(output_dir, 'ldu' + midfix + 'landmarks.png'), LDU_aug.img)
    
    LDU_aug.DrawBBox((0, 255, 0))
    cv2.imwrite(os.path.join(output_dir, 'ldu' + midfix + 'bbox.png'), LDU_aug.img)