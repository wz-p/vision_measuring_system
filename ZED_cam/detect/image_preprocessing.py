#!/user/bin/env python
# -*- coding:utf-8 -*-
# author：wangzhen
# create：2022/6/9 009 10:15
# Software: PyCharm

from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import interpolate
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import glob
from scipy.integrate import quad
import math as m
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


class ZED_config:
    def __init__(self):
        """
        相机参数为预标定数据，精度较高
        """
        self.left_camera_matrix = np.array([[1061.3500, 0., 1103.0200],  # 左相机内参
                                            [0., 1060.2500, 605.5170],
                                            [0., 0., 1.]])
        self.left_distortion = np.array(
            [[-0.0421, 0.0118, -0.0004, 0, -0.0057]])  # 左相机畸变系数k1, k2, p1, p2, k3=[-0.0577]
        self.right_camera_matrix = np.array([[1058.9000, 0., 1097.3900],  # 右相机内参
                                             [0., 1057.1801, 607.2450],
                                             [0., 0., 1.]])
        self.right_distortion = np.array([-0.0431, 0.0133, -0.0004, 0, -0.0057])  # 右相机畸变系数k1, k2, p1, p2, k3=[-0.0686]

        self.essential_matrix = np.array([[-0.0004, 0.1757, -0.3630],  # 本征矩阵
                                          [0.1335, 0.3338, 120.1385],
                                          [0.0046, -120.1390, 0.3332]])
        self.fundamental_matrix = np.array([[0., 0., -0.0004],  # 基础矩阵
                                            [0., 0., 0.1133],
                                            [-0.0001, -0.1135, 0.4642]])

        # self.R = np.array([[1., 0.0030, 0.0026],
        #                    [-0.003, 1., 0.0028],
        #                    [-0.0026, -0.0028, 1.]])
        # self.T = np.array([[-120.1389], [-0.3625], [-0.1767]])  # 平移关系向量,第一个为基线长度

        om = np.array([0.0021, 0.0025, 0.0028])  # 旋转关系向量
        self.R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
        self.T = np.array([[-119.4570], [-0.3341], [0.2413]])

        self.size = (2208, 1242)  # 图像尺寸

        self.f = 1061.83297  # 左相机焦距
        self.B = 120.1389  # 基线距离

        """进行立体更正
        cameraMatrix1-第一个摄像机的摄像机矩阵
        distCoeffs1-第一个摄像机的畸变向量
        cameraMatrix2-第二个摄像机的摄像机矩阵
        distCoeffs1-第二个摄像机的畸变向量
        imageSize-图像大小
        R- stereoCalibrate() 求得的R矩阵
        T- stereoCalibrate() 求得的T矩阵
        R1-输出矩阵，第一个摄像机的校正变换矩阵（旋转变换）
        R2-输出矩阵，第二个摄像机的校正变换矩阵（旋转矩阵）
        P1-输出矩阵，第一个摄像机在新坐标系下的投影矩阵
        P2-输出矩阵，第二个摄像机在想坐标系下的投影矩阵
        Q-4*4的深度差异映射矩阵
        flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，
        该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大"""

        self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, self.validPixROI2 = cv2.stereoRectify(
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            self.size, self.R, self.T,
            flags=0)
        # 计算更正map
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(self.left_camera_matrix, self.left_distortion,
                                                                     self.R1, self.P1, self.size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(self.right_camera_matrix, self.right_distortion,
                                                                       self.R2, self.P2, self.size, cv2.CV_16SC2)

        self.M_left = np.hstack((self.left_camera_matrix, [[0.], [0.], [0.]]))
        self.RT = np.hstack((self.R, self.T))
        self.M_right = np.dot(self.right_camera_matrix, self.RT)

    def get_Ab(self, u_left, v_left, u_right, v_right):
        """通过相机内外参数得到光轴汇聚模型 线性方程中的A和b
           解超定线性方程组可得世界坐标系下的坐标
        """
        A = np.mat(np.zeros((4, 3)))
        A[0, 0] = u_left * self.M_left[2, 0] - self.M_left[0, 0]
        A[0, 1] = u_left * self.M_left[2, 1] - self.M_left[0, 1]
        A[0, 2] = u_left * self.M_left[2, 2] - self.M_left[0, 2]

        A[1, 0] = v_left * self.M_left[2, 0] - self.M_left[1, 0]
        A[1, 1] = v_left * self.M_left[2, 1] - self.M_left[1, 1]
        A[1, 2] = v_left * self.M_left[2, 2] - self.M_left[1, 2]

        A[2, 0] = u_right * self.M_right[2, 0] - self.M_right[0, 0]
        A[2, 1] = u_right * self.M_right[2, 1] - self.M_right[0, 1]
        A[2, 2] = u_right * self.M_right[2, 2] - self.M_right[0, 2]

        A[3, 0] = v_right * self.M_right[2, 0] - self.M_right[1, 0]
        A[3, 1] = v_right * self.M_right[2, 1] - self.M_right[1, 1]
        A[3, 2] = v_right * self.M_right[2, 2] - self.M_right[1, 2]

        b = np.array([[0.], [0.], [0.], [0.]])
        b[0, 0] = self.M_left[0, 3] - u_left * self.M_left[2, 3]
        b[1, 0] = self.M_left[1, 3] - v_left * self.M_left[2, 3]
        b[2, 0] = self.M_right[0, 3] - u_right * self.M_right[2, 3]
        b[3, 0] = self.M_right[1, 3] - u_right * self.M_right[2, 3]
        return A, b

    def undistort(self, img, map1, map2):
        """
        畸变矫正和立体矫正
        """
        undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img


def cv_show(img_name, img):
    """
    显示图片
    """
    cv2.namedWindow(img_name, 0)
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getPixPoint(image):
    """
    获取长抓手的标志点像素坐标
    """
    pinkLower = (165, 0, 0)
    pinkUpper = (180, 255, 255)
    rectangle = np.zeros((1242, 2208, 3), dtype="uint8")
    cv2.rectangle(rectangle, (552, 621), (1656, 1242), (255, 255, 255), -1)
    #     cv_show("Rectangle", rectangle)
    masked = cv2.bitwise_and(image, rectangle)
    #     cv_show('masked', masked)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, pinkLower, pinkUpper)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    canny = cv2.Canny(blurred, 80, 150)
    #     cv_show('canny', canny)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    point = []
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)  # ellipse = [ (x, y) , (a, b), angle ]
            point.append(ellipse[0])
            cv2.ellipse(image_copy, ellipse, (0, 255, 0), 1)
    cv_show("image_copy", image_copy)
    point = np.array(point)
    #     cv_show('image_copy', image_copy)

    # 点排序
    # left_point = point[point[:, 0] < point[0, 0]]
    # left_point = left_point[::-1]
    # right_point = point[point[:, 0] >= point[0, 0]]
    # new_point = np.vstack((left_point, right_point))

    return point


def getPixPoint2(img):
    """
        分割出检测板图像
        检测7×7圆点阵列获取圆心像素坐标值
        只在精度验证实验中用
    """
    img_copy = img.copy()
    img_canny = cv2.Canny(img, 50, 200)
    point = []
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, 1)
        approx = cv2.approxPolyDP(cnt, epsilon, 1)
        array_approx = np.array(approx).reshape(-1, 2)
        if array_approx.shape[0] == 4 and cv2.contourArea(cnt) > 40000:
            # 掩模
            mask = np.zeros(img.shape[:], dtype="uint8")
            mask = cv2.fillPoly(mask, [array_approx], (255, 255, 255))
            masked = cv2.bitwise_and(img, mask)
    #       masked = cv2.morphologyEx(masked,cv2.MORPH_OPEN,(5, 5))

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 10000:
            ellipse = cv2.fitEllipse(cnt)  # ellipse = [ (x, y) , (a, b), angle ]
            point.append(ellipse[0])
            cv2.ellipse(img_copy, ellipse, (0, 255, 0), 2)
    # cv_show('img_copy', img_copy)
    #     point = np.array(point)
    point = sorted(point, key=lambda point: -point[1])
    point = np.array(point)
    point = point.reshape(7, 7, 2)

    return point


def imageSplit(img):
    """
    把双目图片分割成左右视图图片
    """
    left_right_image = np.split(img, 2, axis=1)
    image_left = left_right_image[0]
    image_right = left_right_image[1]
    return image_left, image_right


def get3DPoint():
    """
    获取圆点标定板初位置的世界坐标
    """
    objPoints = np.mgrid[0:120:7j, 0:120:7j, 0:1].T.reshape(-1, 7, 3)
    return objPoints


def getRT(circle_img):
    """
    获取世界坐标系的RT
    circle_img为经过矫正的圆点标定板图片
    """
    zed = ZED_config()
    objp = get3DPoint()
    point_left = getPixPoint2(circle_img)
    retval, rvec, tvec = cv2.solvePnP(np.array([objp[0, 0, :], objp[0, 6, :], objp[6, 0, :], objp[6, 6, :]]),
                                      np.array([point_left[0, 0, :], point_left[0, 6, :], point_left[6, 0, :],
                                                point_left[6, 6, :]]),
                                      zed.left_camera_matrix, zed.left_distortion)
    R = cv2.Rodrigues(rvec)[0]  # 使用Rodrigues变换将om变换为R
    T = tvec.reshape(1, 3)
    return R, T


def getCameraPoint1(imgPoint_left, imgPoint_right):
    """
    光轴汇聚模型
    像素坐标-->相机坐标系下的坐标
    imgPoint_left：[u, v]
    imgPoint_right: [u, v]
    """
    zed = ZED_config()
    A, b = zed.get_Ab(imgPoint_left[0], imgPoint_left[1], imgPoint_right[0], imgPoint_right[1])
    _, xyz = cv2.solve(A, b, flags=cv2.DECOMP_QR)
    return xyz


def getCameraPoint2(imgPoint_left, imgPoint_right):
    """
    光轴平行模型
    由-像素坐标-获取-相机坐标-系下的坐标
    """
    delta = imgPoint_left[0] - imgPoint_right[0]
    f = 1061.35  # 1057.1801
    B = 120  # 117.8   119.4570
    u0 = 1103.0200
    v0 = 605.5170
    # B= np.linalg.norm((-120.1389, -0.3625, -0.1767))
    z = (f * B / delta)
    x = (imgPoint_left[0] - u0) * B / delta
    y = (imgPoint_left[1] - v0) * B / delta
    cameraPoint = np.array([x, y, z])
    return cameraPoint


def getWorldPoint(cameraPoint, R, T):
    """
    相机坐标系---->世界坐标系
    未经验证，可能有误！！！！
    """
    worldPoint = np.dot(np.linalg.inv(R), (cameraPoint - T).T)
    return worldPoint


def accuracyTest1():
    """

    """
    zed = ZED_config()
    a0 = []
    # image = cv2.imread(r'E:\experiment\img_lr.bmp')
    # image_left, image_right=imageSplit(image)
    for i in range(22):
        image_left = cv2.imread(r'.\pic_circle\left_0\left_' + str(i) + '.bmp')
        image_right = cv2.imread(r'.\pic_circle\right_0\right_' + str(i) + '.bmp')
        left_rectified = zed.undistort(image_left, zed.left_map1, zed.left_map2)
        right_rectified = zed.undistort(image_right, zed.right_map1, zed.right_map2)

        point_left = getPixPoint2(left_rectified).reshape(49, -1)
        point_right = getPixPoint2(right_rectified).reshape(49, -1)
        R_w, T_w = getRT(image_left)

        # print('point_left[0,0,:]:',point_left[0,0,:])
        # print('point_right[0,0,:]:', point_right[0, 0, :])
        for j in [0, 7, 14, 21, 28, 35]:
            b1 = []
            for i in range(j, j + 6):
                a1 = getCameraPoint2(point_left[i, :], point_right[i, :])
                a2 = getCameraPoint2(point_left[i + 1, :], point_right[i + 1, :])
                # print(a1)
                # print(a2)
                dis = abs(np.linalg.norm(a1 - a2) - 20)
                print('dis_' + str(i), dis)
                b1.append(dis)
                a0.append(dis)
            print("average_dis:", sum(b1) / len(b1))
            print('----------------------')
        print("average:", sum(a0) / len(a0))
    print(len(a0))
    print("max_error=", max(a0))
    print("min_error=", min(a0))
    print("end:", sum(a0) / len(a0))


def accuracyTest2():
    time3=time.time()
    zed = ZED_config()
    p_left = []
    p_right = []
    coord = []
    for i in [0, 21, 1, 20, 2, 19, 3, 18, 4, 17, 5, 16, 6, 15, 7, 14, 8, 13, 9, 12, 10, 11]:
        image_left = cv2.imread(r'.\pic_circle\left_0\left_' + str(i) + '.bmp')
        image_right = cv2.imread(r'.\pic_circle\right_0\right_' + str(i) + '.bmp')
        left_rectified = zed.undistort(image_left, zed.left_map1, zed.left_map2)
        right_rectified = zed.undistort(image_right, zed.right_map1, zed.right_map2)

        point_left = getPixPoint2(left_rectified).reshape(49, -1)
        point_right = getPixPoint2(right_rectified).reshape(49, -1)
        # R_w, T_w = getRT(image_left)
        p_left.append(point_left)
        p_right.append(point_right)

    p_left = np.array(p_left).reshape(1078, -1)
    p_right = np.array(p_right).reshape(1078, -1)

    time1 = time.time()
    for i in range(1078):
        xyz = getCameraPoint1(p_left[i, :], p_right[i, :])
        coord.append(xyz)

    time2 = time.time()

    point= np.array(coord).reshape((1078, 3))
    coord = np.array(coord).reshape((11, 2, 49, 3))
    error=[]
    absError=[]


    for i in range(11):
        for j in range(2):
            for k in [8,9,10,11,12,  15,16,17,18,19,  22,23,24,25,26,  29,30,31,32,33,  36,37,38,39,40]:
                d = (np.linalg.norm(coord[i][j][k] - coord[i][j][k+1]) + np.linalg.norm(coord[i][j][k] - coord[i][j][k-1]) + np.linalg.norm(coord[i][j][k] - coord[i][j][k+7]) + np.linalg.norm(coord[i][j][k] - coord[i][j][k-7]))/4
                error.append(d-20)
                absError.append(abs(d-20))

    print(time2-time1)
    print((time2-time1)/1078)
    print(time2-time3)
    # print(error)
    # print(absError)
    #
    # print('max:',max(error))
    # print('min:', min(error))
    return error, absError,point

        # for j in [0, 7, 14, 21, 28, 35]:
        #     b1 = []
        #     for i in range(j, j + 6):
        #         a1 = getCameraPoint2(point_left[i, :], point_right[i, :])
        #         a2 = getCameraPoint2(point_left[i + 1, :], point_right[i + 1, :])
        #         # print(a1)
        #         # print(a2)
        #         dis = abs(np.linalg.norm(a1 - a2) - 20)
        #         # print('dis_' + str(i), dis)
        #         b1.append(dis)
        #         a0.append(dis)
    #         print("average_dis:", sum(b1) / len(b1))
    #         print('----------------------')
    #     print("average:", sum(a0) / len(a0))
    # print(len(a0))
    # print("max_error=", max(a0))
    # print("min_error=", min(a0))
    # print("end:", sum(a0) / len(a0))


def plt3d():
    """
    光轴汇聚模型画3D图
    """
    zed = ZED_config()
    image_left = cv2.imread(r'..\src_video_and_img\img_left.bmp')
    image_right = cv2.imread('..//src_video_and_img//img_right.bmp')
    left_rectified = zed.undistort(image_left, zed.left_map1, zed.left_map2)
    right_rectified = zed.undistort(image_right, zed.right_map1, zed.right_map2)
    left_uv = getPixPoint(left_rectified)
    right_uv = getPixPoint(right_rectified)
    point = []

    for i in range(12):
        A, b = zed.get_Ab(left_uv[i][0], left_uv[i][1], right_uv[i][0], right_uv[i][1])
        # print(A,'\n')
        # print(b)
        # print('\n\n')
        _, xyz = cv2.solve(A, b, flags=cv2.DECOMP_QR)
        point.append(xyz)
    point = np.array(point).reshape((-1, 3)).T

    # plt.scatter(point[:][0], -point[:][1], marker='8', c='b', s=100)
    # plt.show()

    x = point[:][0]
    y = point[:][1]
    z = point[:][2]

    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, marker='8', c='b', s=100)
    ax.set_xlim3d(xmin=0, xmax=150)
    ax.set_ylim3d(ymin=30, ymax=180)
    ax.set_zlim3d(zmin=350, zmax=500)

    plt.show()


def litijiaozheng():
    """立体校正测试"""
    zed = ZED_config()
    img_left = cv2.imread(r'..\src_video_and_img\img_left.bmp')
    img_right = cv2.imread('..//src_video_and_img//img_right.bmp')
    left_rectified = zed.undistort(img_left, zed.left_map1, zed.left_map2)
    right_rectified = zed.undistort(img_right, zed.right_map1, zed.right_map2)
    e1 = np.around(getPixPoint(left_rectified), 4)
    e2 = np.around(getPixPoint(right_rectified), 4)
    print(e1)
    print(e2)


def getPixAndPoint():
    """获取长抓手像素坐标和相机坐标系下的坐标"""
    zed = ZED_config()
    image_left = cv2.imread(r'..\src_video_and_img\img_left.bmp')
    image_right = cv2.imread('..//src_video_and_img//img_right.bmp')
    left_rectified = zed.undistort(image_left, zed.left_map1, zed.left_map2)
    right_rectified = zed.undistort(image_right, zed.right_map1, zed.right_map2)
    left_uv = getPixPoint(left_rectified)
    right_uv = getPixPoint(right_rectified)
    print(left_uv)
    print(right_uv)
    point = []
    for i in range(len(left_uv)):
        point.append(getCameraPoint2(left_uv[i], right_uv[i]))
    a = np.around((np.array(point).reshape(-1, 3)), decimals=4)
    print(a)


def plt4D():
    """
    运行时间稍长，耐心等待
    """
    a, b, e = accuracyTest2()
    c = []
    for i in range(22):
        d = np.array(e)[[8+49*i, 9 +49*i,10+49*i, 11+49*i, 12+49*i,15+49*i, 16+49*i, 17+49*i, 18+49*i, 19+49*i, 22+49*i, 23+49*i, 24+49*i, 25+49*i, 26+49*i, 29+49*i, 30+49*i, 31+49*i, 32+49*i, 33+49*i, 36+49*i, 37+49*i, 38+49*i, 39+49*i, 40+49*i]]
        c.append(d)
    c = np.array(c).reshape(550,3)
    X,Y,Z=c[:,0],c[:,1],c[:,2]
    fig = plt.figure()  # 创建一个图
    ax = fig.add_subplot(111, projection='3d')
    cm = plt.cm.get_cmap('jet')      # 颜色映射，为jet型映射规则
    fig = ax.scatter3D(X,Y,Z, c = a, cmap=cm)
    cb = plt.colorbar(fig)  # 设置坐标轴

    cb.ax.tick_params(labelsize=12)
    cb.set_label('error/mm', size=16)
    plt.show()

def armProcess(img):
    """获取人工肌肉机械臂和软体手的像素坐标"""
    pinkLower = (156, 55, 40)
    pinkUpper = (180, 255, 255)
    rectangle = np.zeros((1242, 2208, 3), dtype="uint8")
    # cv2.rectangle(rectangle, (552, 400), (1500, 1242), (255, 255, 255), -1)  # 机械臂区域
    cv2.rectangle(rectangle, (500, 500), (2000, 1000), (255, 255, 255), -1)  # 软体手区域
    #     cv_show("Rectangle", rectangle)
    masked = cv2.bitwise_and(img, rectangle)
    k = np.ones((5, 5), np.uint8)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    # cv_show('hsv', hsv)
    mask = cv2.inRange(hsv, pinkLower, pinkUpper)
    cv_show('mask', mask)
    open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cv_show('open', open)
    blurred = cv2.GaussianBlur(open, (5, 5), 0)
    cv_show('blurred', blurred)
    canny = cv2.Canny(open, 80, 150)
    cv_show('canny', canny)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_copy = img.copy()
    point = []
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)  # ellipse = [ (x, y) , (a, b), angle ]
            point.append(ellipse[0])
            cv2.ellipse(image_copy, ellipse, (0, 255, 0), 1)
    cv_show("image_copy", image_copy)
    point = np.array(point)
    return point


def arm_plt3D():
    """绘制人工肌肉机械臂的三维图"""
    zed = ZED_config()
    finger = cv2.imread('../src_video_and_img/finger.bmp')
    arm = cv2.imread('../src_video_and_img/arm.bmp')
    finger_left, finger_right = imageSplit(finger)
    arm_left, arm_right = imageSplit(arm)

    left_rectified = zed.undistort(arm_left, zed.left_map1, zed.left_map2)
    right_rectified = zed.undistort(arm_right, zed.left_map1, zed.left_map2)
    # cv_show('arm_left', arm_left)
    # cv_show('arm_right', arm_right)
    # cv_show('finger_left', finger_left)
    # cv_show('finger_right', finger_right)
    point = armProcess(right_rectified)

    p = point[np.lexsort(point[:, ::-1].T)]

    a_p_l_1 = np.array([[1230.08300781, 462.66082764],  # arm_point_left_1
                        [1230.58227539, 420.14849854],
                        [1232.23803711, 506.37677002],
                        [1235.75024414, 549.35766602],
                        [1240.72192383, 596.16363525],
                        [1246.16174316, 639.16589355],
                        [1253.92211914, 678.80755615],
                        [1261.0625, 720.03692627]])
    a_p_l_2 = np.array([[1346.45227051, 575.61010742],  # arm_point_left_2
                        [1346.73571777, 522.98468018],
                        [1348.63122559, 466.45211792],
                        [1348.8013916, 626.77453613],
                        [1351.49267578, 678.76776123],
                        [1353.11108398, 410.79669189],
                        [1356.00500488, 732.90618896],
                        [1363.89892578, 780.85369873]])
    a_p_l_3 = np.array([[1412.34277344, 498.03695679],  # arm_point_left_3
                        [1412.55212402, 421.12759399],
                        [1413.60266113, 460.06747437],
                        [1415.07995605, 537.57165527],
                        [1418.47033691, 580.20916748],
                        [1421.27502441, 624.33856201],
                        [1424.67822266, 664.55932617],
                        [1433.54272461, 701.8059082]])

    a_p_l_1 = a_p_l_1[np.lexsort(a_p_l_1.T)]
    a_p_l_2 = a_p_l_2[np.lexsort(a_p_l_2.T)]
    a_p_l_3 = a_p_l_3[np.lexsort(a_p_l_3.T)]

    a_p_r_1 = p[:8][np.lexsort(p[:8].T)]
    a_p_r_2 = p[8:16][np.lexsort(p[8:16].T)]
    a_p_r_3 = p[17:][np.lexsort(p[17:].T)]

    pw_1 = []
    pw_2 = []
    pw_3 = []
    for i in range(8):
        pw_1.append(getCameraPoint2(a_p_l_1[i], a_p_r_1[i]))
        pw_2.append(getCameraPoint2(a_p_l_2[i], a_p_r_2[i]))
        pw_3.append(getCameraPoint2(a_p_l_3[i], a_p_r_3[i]))
    pw_1 = np.around((np.array(pw_1).reshape(-1, 3)), decimals=4)
    pw_2 = np.around((np.array(pw_2).reshape(-1, 3)), decimals=4)
    pw_3 = np.around((np.array(pw_3).reshape(-1, 3)), decimals=4)

    x1, x2, x3 = pw_1[:, 0], pw_2[:, 0], pw_3[:, 0]
    y1, y2, y3 = pw_1[:, 1], pw_2[:, 1], pw_3[:, 1]
    z1, z2, z3 = pw_1[:, 2], pw_2[:, 2], pw_3[:, 2]
    tck1, u1 = interpolate.splprep([x1, y1, z1], s=0)
    tck2, u2 = interpolate.splprep([x2, y2, z2], s=0)
    tck3, u3 = interpolate.splprep([x3, y3, z3], s=0)

    xnew1, ynew1, znew1 = interpolate.splev(np.linspace(0, 1, 1000), tck1, der=0)
    xnew2, ynew2, znew2 = interpolate.splev(np.linspace(0, 1, 1000), tck2, der=0)
    xnew3, ynew3, znew3 = interpolate.splev(np.linspace(0, 1, 1000), tck3, der=0)

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    # ax = plt.axes(projection='3d')

    ax.scatter3D(x1, y1, z1, marker='8', c='b', s=50)
    ax.scatter3D(x2, y2, z2, marker='8', c='b', s=50)
    ax.scatter3D(x3, y3, z3, marker='8', c='b', s=50)
    temp = getCameraPoint2([1401.50769043 , 949.40478516], [1108.97143555  ,950.04034424])
    ax.scatter3D(temp[0], temp[1],temp[2], marker='8', c='b', s=50)

    ax.plot(xnew1, ynew1, znew1, 'r-')
    ax.plot(xnew2, ynew2, znew2, 'r-')
    ax.plot(xnew3, ynew3, znew3, 'r-')

    ax.set_xlim3d(xmin=-50, xmax=200)
    ax.set_ylim3d(ymin=-100, ymax=150)
    ax.set_zlim3d(zmin=300, zmax=550)

    print(pw_1)
    print(pw_2)
    print(pw_3)
    plt.show()


def figure_plt3D():
    """绘制软体机械手的三维坐标"""
    zed = ZED_config()
    finger = cv2.imread('../src_video_and_img/finger.bmp')
    arm = cv2.imread('../src_video_and_img/arm.bmp')
    finger_left, finger_right = imageSplit(finger)
    arm_left, arm_right = imageSplit(arm)

    left_rectified = zed.undistort(finger_left, zed.left_map1, zed.left_map2)
    right_rectified = zed.undistort(finger_right, zed.left_map1, zed.left_map2)
    # cv_show('arm_left', arm_left)
    # cv_show('arm_right', arm_right)
    # cv_show('finger_left', finger_left)
    # cv_show('finger_right', finger_right)
    point_l = armProcess(left_rectified)
    point_r = armProcess(right_rectified)

    pl = point_l[np.lexsort(point_l[:, ::-1].T)]
    pr = point_r[np.lexsort(point_r[:, ::-1].T)]

    pl_1 = pl[:6]
    pl_2 = pl[6:]
    pr_1 = pr[:6]
    pr_2 = pr[6:]

    pw_1 = []
    pw_2 = []
    for i in range(6):
        pw_1.append(getCameraPoint2(pl_1[i], pr_1[i]))
        pw_2.append(getCameraPoint2(pl_2[i], pr_2[i]))

    pw_1 = np.around((np.array(pw_1).reshape(-1, 3)), decimals=4)
    pw_2 = np.around((np.array(pw_2).reshape(-1, 3)), decimals=4)

    x1, x2 = pw_1[:, 0], pw_2[:, 0]
    y1, y2 = pw_1[:, 1], pw_2[:, 1]
    z1, z2 = pw_1[:, 2], pw_2[:, 2]
    tck1, u1 = interpolate.splprep([x1, y1, z1], s=0)
    tck2, u2 = interpolate.splprep([x2, y2, z2], s=0)

    xnew1, ynew1, znew1 = interpolate.splev(np.linspace(0, 1, 1000), tck1, der=0)
    xnew2, ynew2, znew2 = interpolate.splev(np.linspace(0, 1, 1000), tck2, der=0)

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    # ax = plt.axes(projection='3d')

    ax.scatter3D(x1, y1, z1, marker='8', c='b', s=50)
    ax.scatter3D(x2, y2, z2, marker='8', c='b', s=50)

    ax.plot(xnew1, ynew1, znew1, 'r-')
    ax.plot(xnew2, ynew2, znew2, 'r-')

    ax.set_xlim3d(xmin=-50, xmax=200)
    ax.set_ylim3d(ymin=-100, ymax=150)
    ax.set_zlim3d(zmin=300, zmax=550)
    plt.show()
    print(pw_1)
    print(pw_2)



if __name__ == '__main__':
    arm_plt3D()