from BBox import BBox
import numpy as np
import cv2
import math
import copy
class LandmarkDataUnit(object):
    img = None
    landmarks_img = None
    landmarks_bbox = None
    bbox = None
    lossmult = None

    def __init__(self, img = None, landmarks_img = None, landmarks_bbox = None, bbox = None, lossmult = None):
        self.img = img
        self.landmarks_img = landmarks_img
        self.landmarks_bbox = landmarks_bbox
        self.bbox = bbox
        self.lossmult = lossmult

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

    @classmethod
    def MirrorLandmarks(cls, landmarks):
        landmarks_mirror = np.zeros(landmarks.shape, np.float32)
        if len(landmarks) == 98:
            for i in range(33):
                landmarks_mirror[i] = landmarks[32-i]
            
            for i in range(5):
                landmarks_mirror[33+i] = landmarks[46-i]
                landmarks_mirror[46-i] = landmarks[33+i]

            for i in range(4):
                landmarks_mirror[38+i] = landmarks[50-i]
                landmarks_mirror[50-i] = landmarks[38+i]

            for i in range(4):
                landmarks_mirror[51+i] = landmarks[51+i]
            
            for i in range(5):
                landmarks_mirror[55+i] = landmarks[59-i]

            for i in range(5):
                landmarks_mirror[60+i] = landmarks[72-i]
                landmarks_mirror[72-i] = landmarks[60+i]
            
            for i in range(3):
                landmarks_mirror[65+i] = landmarks[75-i]
                landmarks_mirror[75-i] = landmarks[65+i]

            for i in range(7):
                landmarks_mirror[76+i] = landmarks[82-i]

            for i in range(5):
                landmarks_mirror[83+i] = landmarks[87-i]

            for i in range(5):
                landmarks_mirror[88+i] = landmarks[92-i]

            for i in range(3):
                landmarks_mirror[93+i] = landmarks[95-i]

            for i in range(2):
                landmarks_mirror[96+i] = landmarks[97-i]
        elif len(landmarks) == 68:
            for i in range(17):
                landmarks_mirror[i] = landmarks[16-i]
            
            for i in range(5):
                landmarks_mirror[17+i] = landmarks[26-i]
                landmarks_mirror[26-i] = landmarks[17+i]

            for i in range(4):
                landmarks_mirror[27+i] = landmarks[27+i]

            for i in range(5):
                landmarks_mirror[31+i] = landmarks[35-i]

            for i in range(4):
                landmarks_mirror[36+i] = landmarks[45-i]
                landmarks_mirror[45-i] = landmarks[36+i]

            for i in range(2):
                landmarks_mirror[40+i] = landmarks[47-i]
                landmarks_mirror[47-i] = landmarks[40+i]
            
            for i in range(7):
                landmarks_mirror[48+i] = landmarks[54-i]

            for i in range(5):
                landmarks_mirror[55+i] = landmarks[59-i]

            for i in range(5):
                landmarks_mirror[60+i] = landmarks[64-i]

            for i in range(3):
                landmarks_mirror[65+i] = landmarks[67-i]
        else:
            raise Exception('No mirror function for these landmarks defined.')
        return landmarks_mirror

    def Mirror(self):
        (h, w) = self.img.shape[:2]
        M = np.array([[-1.0, 0.0, w], [0.0, 1.0, 0.0]])
        self.img = cv2.warpAffine(self.img, M, (w, h))
        N_landmarks   = len(self.landmarks_img)
        # self.landmarks_img = np.reshape(self.landmarks_img, (N_landmarks, 1, 2))
        # self.landmarks_img = np.reshape(cv2.transform(self.landmarks_img, M), (N_landmarks, 2))
        # self.landmarks_img = LandmarkDataUnit.MirrorLandmarks(self.landmarks_img)
        if not(self.landmarks_bbox is None):
            M = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            self.landmarks_bbox = np.reshape(self.landmarks_bbox, (N_landmarks, 1, 2))
            self.landmarks_bbox = np.reshape(cv2.transform(self.landmarks_bbox, M), (N_landmarks, 2))
            self.landmarks_bbox = LandmarkDataUnit.MirrorLandmarks(self.landmarks_bbox)
        return self

    def TranslateBBox(self, translate):
        self.bbox.Translate(translate)
        return self

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
        return self

    def CroppedImg(self):
        return self.img[self.bbox.y1:self.bbox.y2, self.bbox.x1:self.bbox.x2]

    def CropImg(self):
        self.img = self.img[self.bbox.y1:self.bbox.y2, self.bbox.x1:self.bbox.x2]

    def Crop(self):
        self.img = self.img[self.bbox.y1:self.bbox.y2, self.bbox.x1:self.bbox.x2]
        x1 = 0
        y1 = 0
        x2 = self.bbox.x2 - self.bbox.x1
        y2 = self.bbox.y2 - self.bbox.y1
        self.bbox = BBox((x1, y1, x2, y2))
        self.ProjectBBoxLandmarksToImg()
        return self

    def ResizeImg(self, size, inter=cv2.INTER_LINEAR):
        self.img = cv2.resize(self.img, size, interpolation=inter)

    def Resize(self, size, inter=cv2.INTER_LINEAR):
        (h, w) = self.img.shape[:2]
        M = np.array([[float(size[0])/float(w), 0.0, 0.0], [0.0, float(size[1])/float(h), 0.0]])
        self.ResizeImg(size, inter)

        N_landmarks = len(self.landmarks_img)
        self.landmarks_img = np.reshape(self.landmarks_img, (N_landmarks, 1, 2))
        self.landmarks_img = np.reshape(cv2.transform(self.landmarks_img, M), (N_landmarks, 2))
        if not(self.bbox is None):
            bbox = np.array([[[self.bbox.x1,self.bbox.y1]],[[self.bbox.x2,self.bbox.y2]]])
            bbox = cv2.transform(bbox, M)
            self.bbox = BBox((bbox[0][0][0], bbox[0][0][1], bbox[1][0][0], bbox[1][0][1]))
        return self

    def Scale(self, scale, inter=cv2.INTER_LINEAR):
        (h, w) = self.img.shape[:2]
        size = (int(float(w)*scale[0]), int(float(h)*scale[1]))
        self.Resize(size, inter)

    def CalcInterocularDistance(self):
        if len(self.landmarks_bbox) == 98:
            left_eye = self.landmarks_bbox[60]
            right_eye = self.landmarks_bbox[72]
        elif len(self.landmarks_bbox) == 68:
            left_eye = self.landmarks_bbox[37]
            right_eye = self.landmarks_bbox[46]
        else:
            raise ValueError('No eye location defined for these landmarks.')
        return np.linalg.norm(right_eye-left_eye)+1e-3

    def CalcLossMult(self):
        N_landmarks = len(self.landmarks_bbox)
        self.lossmult = np.zeros((N_landmarks, 2), np.float32)
        IOD = self.CalcInterocularDistance()
        self.lossmult.fill(1.0/IOD)

    # def CalcLossMult(self):
    #     N_landmarks = len(self.landmarks_bbox)
    #     self.lossmult = np.zeros((N_landmarks, 2), np.float32)
    #     IOD = self.CalcInterocularDistance()

    #     for index in range(N_landmarks):
    #         if abs(self.landmarks_bbox[index][0])<0.5 and abs(self.landmarks_bbox[index][1])<0.5:
    #             self.lossmult[index].fill(1.0/IOD)
    #         else:
    #             self.lossmult[index].fill(0.5/IOD)

    def DrawLandmarks(self, color):
        (h, w) = self.img.shape[:2]
        for lm in self.landmarks_img:
            if lm[0] >= 0.0 and lm[0] <= w and lm[1] >= 0.0 and lm[1] <= h:
                cv2.circle(self.img, (int(lm[0]), int(lm[1])), 1, color, 2)

    def DrawBBox(self, color):
        self.img = BBox.DrawBBoxOnImg(self.bbox, self.img, color)