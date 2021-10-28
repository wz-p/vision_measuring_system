import cv2
import numpy
import os

# Open the ZED camera
cap = cv2.VideoCapture(1)
if cap.isOpened() == 0:
    exit(-1)

# Set the video resolution to HD2k (4416*1242)
# 1080p	3840x1080
# 720p	2560x720
# WVGA	1344x376
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)

path = ".\\accuracyTest\\"  # 图片存储路径
cv2.namedWindow("left")
cv2.namedWindow("right")
counter = 0
z_distance = 0
folder_left = ".\\detect_pic\\pic_1\\left\\"  # 照片存储路径
folder_right = ".\\detect_pic\\pic_1\\right\\"
if not os.path.exists(folder_left):
    os.makedirs(folder_left)
if not os.path.exists(folder_right):
    os.makedirs(folder_right)


def shot(image_left, image_right):
    global counter
    path_left = folder_left + 'left' + "_" + str(counter) + ".bmp"
    path_right = folder_right + 'right' + "_" + str(counter) + ".bmp"
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
