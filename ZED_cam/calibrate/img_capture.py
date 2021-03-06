#!/user/bin/env python
# -*- coding:utf-8 -*-
# author：wangzhen
# create：2022/6/9 009 10:15
# Software: PyCharm
##################################
# 读取相机中的图片并把左右视图分别保存
##################################

import cv2
import numpy
import os

# Open the ZED camera
cap = cv2.VideoCapture(0)
if cap.isOpened() == 0:
    exit(-1)

# Set the video resolution to HD2k (4416*1242)
# 1080p	3840x1080
# 720p	2560x720
# WVGA	1344x376
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)

path = "..\\src_video_and_img\\image_capture\\"  # 图片存储路径
cv2.namedWindow("left")
cv2.namedWindow("right")
counter = 0
z_distance = 0
path_left = path + 'left' + "_" + str(counter) + ".bmp"
path_right = path + 'right' + "_" + str(counter) + ".bmp"
if not os.path.exists(path_left):
    os.makedirs(path_left)
if not os.path.exists(path_right):
    os.makedirs(path_right)


def shot(image_left, image_right):
    global counter
    cv2.imwrite(path_left, image_left)
    cv2.imwrite(path_right, image_right)
    print("图片保存于: " + path + '--------' + str(counter))


while True:
    # Get a new frame from camera
    ret, frame = cap.read()
    # Extract left and right images from side-by-side
    left_right_image = numpy.split(frame, 2, axis=1)
    image_left = left_right_image[0]
    image_right = left_right_image[1]

    # Display images
    # cv2.imshow("frame", frame)q
    # cv2.namedWindow("left", 0)
    # cv2.resizeWindow("left", 1104, 621)
    cv2.imshow("left", image_left)
    # cv2.namedWindow("right", 0)
    # cv2.resizeWindow("right", 1104, 621)
    cv2.imshow("right", image_right)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot(image_left, image_right)
        counter += 1

cap.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")
