# -*- coding: utf-8 -*
# environment is python3.5  others aren't support
from skimage import io
import os
import dlib
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_landmarks(img):
    dets = detector(img, 1)
    landmarks = np.zeros((17,2))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmarks[0] = (shape.part(48).x, shape.part(48).y)
        for i in range(6):
            landmarks[1+i] = (shape.part(59-i).x, shape.part(59-i).y)    
        for i in range(10):
            landmarks[7+i] = (shape.part(26-i).x, shape.part(26-i).y)
    return landmarks


def inside(X,Y,Region): 
    j=len(Region)-1
    flag=False
    for i in range(len(Region)):
        if (Region[i][1]<Y and Region[j][1]>=Y or Region[j][1]<Y and Region[i][1]>=Y):  
            if (Region[i][0] + (Y - Region[i][1]) / (Region[j][1] - Region[i][1]) * (Region[j][0] - Region[i][0]) < X):
                flag =not flag
        j=i
    return flag

# collect all images to crop
paths = []
picpath = os.path.abspath(os.path.dirname(os.getcwd())) + '/securityAI_round1_images'
dire = None
for root, dirs, files in os.walk(picpath): 
    for f in files:
        paths.append(os.path.join(root, f))

num = 1
for path in paths:
    print("processing image  =========>")
    print(num)
    img = io.imread(path)
    region = get_landmarks(img)
    shape = list(img.shape) + [3]  # Convert images into lists for easy access
    img1 = img.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not inside(j, i, region): 
                img1[i, j] = (0, 0, 0)
            else:
                img1[i, j] = (255, 255, 255)
    io.imsave(os.path.abspath(os.path.dirname(os.getcwd())) + '/mask/' + path.split('/')[-1], img1)
    num += 1


