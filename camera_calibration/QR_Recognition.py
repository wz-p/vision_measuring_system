import cv2
from pyzbar.pyzbar import decode
from camera_configs import Config
import numpy as np


coordinate = []


def video_capture(c_1=1, c_2=2):
    camera_l = cv2.VideoCapture(c_1, cv2.CAP_DSHOW)
    camera_l.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera_r = cv2.VideoCapture(c_2, cv2.CAP_DSHOW)
    camera_r.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    return camera_l, camera_r


def xyz_list(xyz):
    """
    输出相对位移
    """
    global coordinate
    coordinate.append(xyz[1].reshape(3))
    if len(coordinate) >= 2:
        print(coordinate)
        print(np.linalg.norm(coordinate[-1] - coordinate[0]))
        print(np.linalg.norm(coordinate[-1] - coordinate[-2]), '\n')
        # coordinate.pop(0)


left_camera, right_camera = video_capture()

while True:
    ret, left_frame = left_camera.read()
    _, right_frame = right_camera.read()
    cv2.namedWindow("left", 0)
    cv2.namedWindow("right", 0)
    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        # print(left_frame.shape, right_frame.shape)
        result_l = decode(left_frame)
        result_r = decode(right_frame)

        center_l_x = 0
        center_l_y = 0
        center_r_x = 0
        center_r_y = 0
        for item_l in result_l:
            #         print(item.type)
            #         print(item.data)
            #         print(item.rect)
            # print(item_l.polygon)

            for i in range(4):
                center_l_x += item_l.polygon[i][0]
                center_l_y += item_l.polygon[i][1]
            center_l_x = center_l_x / 4
            center_l_y = center_l_y / 4
            # print('l_x:', center_l_x)
            # print('l_y:', center_l_y)

        for item_r in result_r:
            # print(item_r.polygon)
            for i in range(4):
                center_r_x += item_r.polygon[i][0]
                center_r_y += item_r.polygon[i][1]
            center_r_x = center_r_x / 4
            center_r_y = center_r_y / 4

        config = Config()
        A, b = config.get_Ab(center_l_x, center_l_y, center_r_x, center_r_y)
        xyz = cv2.solve(A, b, flags=cv2.DECOMP_QR)
        # print('X:', x, "\tY:", y, "\tZ:", z)
        print('X:', xyz[1][0], '\tY:', xyz[1][1], '\tZ:', xyz[1][2])
        xyz_list(xyz)
