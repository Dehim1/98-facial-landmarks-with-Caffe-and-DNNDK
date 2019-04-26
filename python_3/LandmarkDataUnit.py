from BBox import BBox
import numpy as np
import cv2
import math
import copy
class LandmarkDataUnit(object):
    # img: cv2.Mat = None
    img = None
    # landmarks_img: np.array = None
    landmarks_img = None
    # landmarks_bbox: np.array = None
    landmarks_bbox = None
    # bbox: BBox = None
    bbox = None

    # def __init__(self, img: cv2.Mat, landmarks: np.array):
    def __init__(self, img = None, landmarks_img = None, landmarks_bbox = None, bbox = None):
        self.img = img
        self.landmarks_img = landmarks_img
        self.landmarks_bbox = landmarks_bbox
        self.bbox = bbox

    def BBoxFromLandmarks(self):
        x1 = self.landmarks_img[0][0]
        y1 = self.landmarks_img[0][1]
        x2 = self.landmarks_img[0][0]
        y2 = self.landmarks_img[0][1]
        for lm in self.landmarks_img:
            if lm[0] < x1:
                x1 = lm[0]
            if lm[1] < y1:
                y1 = lm[1]
            if lm[0] > x2:
                x2 = lm[0]            
            if lm[1] > y2:
                y2 = lm[1]
        self.bbox = BBox(np.array([
            math.floor(x1),
            math.floor(y1),
            math.ceil(x2), 
            math.ceil(y2)])
            .astype(int))

    def ProjectImgLandmarksToBBox(self):
        self.landmarks_bbox = self.landmarks_img - np.array([[self.bbox.x_center, self.bbox.y_center]])
        self.landmarks_bbox = self.landmarks_bbox / np.array([[self.bbox.width, self.bbox.height]])

    def ProjectBBoxLandmarksToImg(self):
        self.landmarks_img = self.landmarks_bbox * np.array([[self.bbox.width, self.bbox.height]])
        self.landmarks_img = self.landmarks_img + np.array([[self.bbox.x_center, self.bbox.y_center]])

    # def Rotate(self, angle: float) -> 'LandmarkDataUnit':
    def Rotate(self, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = self.img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        self.img = cv2.warpAffine(self.img, M, (nW, nH))
        N_landmarks   = len(self.landmarks_img)
        self.landmarks_img = np.reshape(self.landmarks_img, (N_landmarks, 1, 2))
        self.landmarks_img = np.reshape(cv2.transform(self.landmarks_img, M), (N_landmarks, 2))
        return self

    def Shear(self, shear):
        shear_x = shear[0]*float(self.bbox.width)/float(self.bbox.height)
        shear_y = shear[1]*float(self.bbox.height)/float(self.bbox.width)
        M = np.array([[1.0+0.25*shear_x*shear_y, shear_x-0.25*shear_x*shear_y, 0.0], 
                      [shear_y-0.25*shear_x*shear_y, 0.25*shear_x*shear_y+1.0, 0.0]])
        M = np.array([[shear_x*shear_y+1.0, shear_x, 0.0], [shear_y, 1.0, 0.0]])

    # def Mirror(self) -> 'LandmarkDataUnit':
    def Mirror(self):
        (h, w) = self.img.shape[:2]
        M = np.array([[-1.0, 0.0, w], [0.0, 1.0, 0.0]])
        self.img = cv2.warpAffine(self.img, M, (w, h))
        N_landmarks   = len(self.landmarks_img)
        landmarks_mirror = np.reshape(self.landmarks_img, (N_landmarks, 1, 2))
        landmarks_mirror = np.reshape(cv2.transform(landmarks_mirror, M), (N_landmarks, 2))
        if len(landmarks_mirror) == 98:
            for i in range(33):
                self.landmarks_img[i] = landmarks_mirror[32-i]
            
            for i in range(5):
                self.landmarks_img[33+i] = landmarks_mirror[46-i]
                self.landmarks_img[46-i] = landmarks_mirror[33+i]

            for i in range(4):
                self.landmarks_img[38+i] = landmarks_mirror[50-i]
                self.landmarks_img[50-i] = landmarks_mirror[38+i]

            for i in range(4):
                self.landmarks_img[51+i] = landmarks_mirror[51+i]
            
            for i in range(5):
                self.landmarks_img[55+i] = landmarks_mirror[59-i]

            for i in range(5):
                self.landmarks_img[60+i] = landmarks_mirror[72-i]
                self.landmarks_img[72-i] = landmarks_mirror[60+i]
            
            for i in range(3):
                self.landmarks_img[65+i] = landmarks_mirror[75-i]
                self.landmarks_img[75-i] = landmarks_mirror[65+i]

            for i in range(7):
                self.landmarks_img[76+i] = landmarks_mirror[82-i]

            for i in range(5):
                self.landmarks_img[83+i] = landmarks_mirror[87-i]

            for i in range(5):
                self.landmarks_img[88+i] = landmarks_mirror[92-i]

            for i in range(3):
                self.landmarks_img[93+i] = landmarks_mirror[95-i]

            for i in range(2):
                self.landmarks_img[96+i] = landmarks_mirror[97-i]
        elif len(landmarks_mirror) == 68:
            for i in range(17):
                self.landmarks_img[i] = landmarks_mirror[16-i]
            
            for i in range(5):
                self.landmarks_img[17+i] = landmarks_mirror[26-i]
                self.landmarks_img[26-i] = landmarks_mirror[17+i]

            for i in range(4):
                self.landmarks_img[27+i] = landmarks_mirror[27+i]

            for i in range(5):
                self.landmarks_img[31+i] = landmarks_mirror[35-i]

            for i in range(4):
                self.landmarks_img[36+i] = landmarks_mirror[45-i]
                self.landmarks_img[45-i] = landmarks_mirror[36+i]

            for i in range(2):
                self.landmarks_img[40+i] = landmarks_mirror[47-i]
                self.landmarks_img[47-i] = landmarks_mirror[40+i]
            
            for i in range(7):
                self.landmarks_img[48+1] = landmarks_mirror[54-i]

            for i in range(5):
                self.landmarks_img[55+1] = landmarks_mirror[59-i]

            for i in range(5):
                self.landmarks_img[60+1] = landmarks_mirror[64-i]

            for i in range(3):
                self.landmarks_img[65+1] = landmarks_mirror[67-i]
        else:
            raise ValueError('No mirror function for these landmarks defined.')
        return self

    def CalcInterocularDistance(self):
        if len(self.landmarks_bbox) == 98:
            left_eye = self.landmarks_bbox[60]
            right_eye = self.landmarks_bbox[72]
        elif len(self.landmarks_bbox) == 68:
            left_eye = self.landmarks_bbox[37]
            right_eye = self.landmarks_bbox[46]
        else:
            raise ValueError('Eye location not defined for these landmarks.')
        return np.linalg.norm(right_eye-left_eye)+1e-3

    def TranslateBBox(self, translate):
        self.bbox.Translate(translate)
        return self

    # def ScaleBBox(self, scale: tuple) -> 'LandmarkDataUnit':
    def ScaleBBox(self, scale):
        self.bbox.Scale(scale)
        return self

    def ClipBBox(self):
        (h, w) = self.img.shape[:2]
        x1 = math.floor(self.bbox.x1)
        y1 = math.floor(self.bbox.y1)
        x2 = math.ceil(self.bbox.x2)
        y2 = math.ceil(self.bbox.y2)
        if x1 < 0:
            x1 = 0
        if x1 > w:
            x1 = w
        if y1 < 0:
            y1 = 0
        if y1 > h:
            y1 = h
        if x2 > w:
            x2 = w
        if x2 < 0:
            x2 = 0
        if y2 > h:
            y2 = h
        if y2 < 0:
            y2 = 0
        self.bbox = BBox(np.array([x1, y1, x2, y2]).astype(int))

    def BBoxCroppedImg(self):
        return self.img[self.bbox.y1:self.bbox.y2, self.bbox.x1:self.bbox.x2]

    # def DrawLandmarks(self, color: tuple):
    def DrawLandmarks(self, color):
        (h, w) = self.img.shape[:2]
        for lm in self.landmarks_img:
            if lm[0] >= 0.0 and lm[0] <= w and lm[1] >= 0.0 and lm[1] <= h:
                cv2.circle(self.img, (int(lm[0]), int(lm[1])), 1, color, 2)

    # def DrawBBox(self, color: tuple):
    def DrawBBox(self, color):
        self.img = BBox.DrawnBBoxMat(self.bbox, self.img, color)