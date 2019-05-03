import os
import sys
import numpy as np

def GetData_98(filepath):
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

def GetData_68(root):
    N_landmarks = 68
    data = []
    for root, _, files in os.walk(root):
        for f in files:
            if not f.endswith('.pts'):
                continue
            lm_path = os.path.join(root, f)

            jpg = f.split('.')[0] + '.jpg'
            png = f.split('.')[0] + '.png'
            jpg_path = os.path.join(root, jpg)
            png_path = os.path.join(root, png)
            if os.path.isfile(jpg_path):
                img_path = jpg_path
            elif os.path.isfile(png_path):
                img_path = png_path
            else:
                continue

            lm_file = open(lm_path, 'r')

            line = lm_file.readline()
            while not line.startswith('n_points'):
                line = lm_file.readline()
            num_pts = line.split(':')[1]
            num_pts = num_pts.strip()
            if int(num_pts) != N_landmarks:
                continue
            line = lm_file.readline()

            landmarks = np.zeros((N_landmarks, 2))
            line = lm_file.readline()
            i = 0
            while not line.startswith('}'):
                lm = line.split(' ')
                landmarks[i] = (float(lm[0]), float(lm[1]))
                line = lm_file.readline()
                i += 1

            data.append((img_path, landmarks))
    return data