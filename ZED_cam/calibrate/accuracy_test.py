import cv2
import numpy as np
from zed_config import ZED_config

"""
zed = ZED_config()
objPoints = []

a = cv2.imread('test.jpg')
a = cv2.resize(a, (2222, 1200))
gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('a', 0)
cv2.imshow('a', gray)
cv2.waitKey(0)
ret, corners = cv2.findCirclesGrid(gray, (7, 7))
cv2.drawChessboardCorners(a, (7, 7), corners, None)
cv2.namedWindow('a', 0)
cv2.imshow('a', a)
cv2.waitKey(0)
corners.resize(49, 2)
print(corners)
print(corners.shape)
"""

zed = ZED_config()
objPoints = np.mgrid[0:260:14j, 0:120:7j,  0:200:10j].T.reshape(-1,7,3)
print(objPoints)

