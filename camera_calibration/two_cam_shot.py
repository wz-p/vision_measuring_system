import cv2
import time
from datetime import datetime


left_camera = cv2.VideoCapture(1)
left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

right_camera = cv2.VideoCapture(2)
right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

path = "C:\\Users\\WeiCJ\\Desktop\\OpenCV_Study\\pic_calibration\\"  # 图片存储路径

AUTO = False  # True自动拍照，False则手动按s键拍照
INTERVAL = 0.0000005  # 调整自动拍照间隔

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.moveWindow("left", 0, 0)

counter = 0
utc = time.time()
folder = "C:\\Users\\WeiCJ\\Desktop\\OpenCV_Study\\pic_calibration\\"  # 照片存储路径


def shot(pos, frame):
    global counter
    timestr = datetime.now()
    path = folder + pos + "_" + str(counter) + ".jpg"
    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)


while True:
    ret, left_frame = left_camera.read()
    _, right_frame = right_camera.read()

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)

    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
        utc = now

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1

left_camera.release()
right_camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")
