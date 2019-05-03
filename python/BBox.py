import math
import numpy as np
import cv2
import copy
class BBox(object):
    x1 = None
    y1 = None
    x2 = None
    y2 = None
    x_center = None
    y_center = None
    width = None
    height = None
    
    # def __init__(self, bbox: np.array):
    def __init__(self, bbox):
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
        self.x_center = (self.x1 + self.x2)/2
        self.y_center = (self.y1 + self.y2)/2
        self.width  = self.x2 - self.x1
        self.height = self.y2 - self.y1

    # def scale(self, scale: tuple) -> 'BBox':
    def Scale(self, scale):
        self.width = self.width*scale[0]
        self.height = self.height*scale[1]
        self.x1 = self.x_center - self.width/2
        self.y1 = self.y_center - self.height/2
        self.x2 = self.x_center + self.width/2
        self.y2 = self.y_center + self.height/2
        return self

    # def translate(self, translate: tuple) -> 'BBox':
    def Translate(self, translate):
        self.x_center = self.x_center + translate[0]*self.width
        self.y_center = self.y_center - translate[1]*self.height
        self.x1 = self.x1 + translate[0]*self.width
        self.y1 = self.y1 - translate[1]*self.height
        self.x2 = self.x2 + translate[0]*self.width
        self.y2 = self.y2 - translate[1]*self.height
        return self

    @classmethod
    # def CalcArea(cls, bbox: BBox):
    def CalcArea(cls, bbox):
        return bbox.width*bbox.height
    
    @classmethod
    #def CalcIoU(cls, bbox1: BBox, bbox2: BBox) -> 'float':
    def CalcIoU(cls, bbox1, bbox2):
        x1_I = max(bbox1.x1, bbox2.x1)
        y1_I = max(bbox1.y1, bbox2.y1)
        x2_I = min(bbox1.x2, bbox2.x2)
        y2_I = min(bbox1.y2, bbox2.y2)
        if x1_I >= x2_I or y1_I >= y2_I:
            return 0
        box_I = BBox(np.array([x1_I, y1_I, x2_I, y2_I]))
        area_I = BBox.CalcArea(box_I)
        area_U = BBox.CalcArea(bbox1) + BBox.CalcArea(bbox2) - area_I
        return float(area_I) / float(area_U)

    @classmethod
    # def DrawOnMat(cls, bbox: BBox, img: cv2.Mat, color: tuple) -> cv2.Mat:
    def DrawBBoxOnImg(cls, bbox, img, color):
        cv2.rectangle(img, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), color, 2)
        return img