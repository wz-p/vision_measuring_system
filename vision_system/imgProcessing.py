#!/usr/bin/python
# -*- coding: utf-8 -*-
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

from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout
from PySide6.QtCore import QTimer
import sys, time

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.cbook as cbook


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

    @staticmethod
    def undistort(img, map1, map2):
        undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img


class ImgProcessing:
    def __init__(self):
        pass

    @staticmethod
    def imageSplit(img):
        """
        把双目图片分割成左右视图图片
        """
        left_right_image = np.split(img, 2, axis=1)
        image_left = left_right_image[0]
        image_right = left_right_image[1]
        return image_left, image_right

    @staticmethod
    def getProcessingRet(image, param):
        """
        获取长抓手的标志点像素坐标和检测结果图像
        :param=[(165, 0, 0), (180, 255, 255), (552, 621), (1656, 1242)],HSV上下限和ROI区域
        """

        def cv_show(img_name, img):
            cv2.namedWindow(img_name, 0)
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        pinkLower = param[0]
        pinkUpper = param[1]
        rectangle = np.zeros((1242, 2208, 3), dtype="uint8")
        figure = rectangle.copy()
        cv2.rectangle(rectangle, param[2], param[3], (255, 255, 255), -1)

        masked = cv2.bitwise_and(image, rectangle)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, pinkLower, pinkUpper)
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, iterations=2)
        canny = cv2.Canny(opened, 80, 150)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        point = []
        for cnt in contours:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)  # ellipse = [ (x, y) , (a, b), angle ]
                point.append(ellipse[0])
                # cv2.ellipse(image_copy, ellipse, (0, 255, 0), 1)
                cv2.ellipse(figure, ellipse, (255, 255, 255), -1)

        cv2.rectangle(figure, param[2], param[3], (0, 255, 0), 3)
        point = np.array(point)
        # 点排序
        # left_point = point[point[:, 0] < point[0, 0]]
        # left_point = left_point[::-1]
        # right_point = point[point[:, 0] >= point[0, 0]]
        # new_point = np.vstack((left_point, right_point))

        return figure, point

    @staticmethod
    def getCameraCoordinate(imgPoint_left, imgPoint_right):
        """
        光轴平行模型
        由-像素坐标-获取-相机坐标-系下的坐标
        """
        if imgPoint_left.shape == imgPoint_right.shape and len(imgPoint_left.shape) == 2:
            delta = imgPoint_left[:, 0] - imgPoint_right[:, 0]
            f = 1061.35  # 1057.1801
            B = 120  # 117.8   119.4570
            u0 = 1103.0200
            v0 = 605.5170
            z = (f * B / delta)
            x = (imgPoint_left[:, 0] - u0) * B / delta
            y = (imgPoint_left[:, 1] - v0) * B / delta
            return x, y, z

    @staticmethod
    def un_distort(left_image, right_image):
        zed = ZED_config()
        left_rectified = zed.undistort(left_image, zed.left_map1, zed.left_map2)
        right_rectified = zed.undistort(right_image, zed.left_map1, zed.left_map2)
        return left_rectified, right_rectified

    @staticmethod
    def figure_plt3D(point_left, point_right):
        """绘制软体机械手的三维坐标"""
        zed = ZED_config()
        if len(point_left) == len(point_right):

            # cv_show('arm_left', arm_left)
            # cv_show('arm_right', arm_right)
            # cv_show('finger_left', finger_left)
            # cv_show('finger_right', finger_right)
            pl = point_left[np.lexsort(point_left[:, ::-1].T)]
            pr = point_right[np.lexsort(point_right[:, ::-1].T)]

            pl_1 = pl[:6]
            pl_2 = pl[6:]
            pr_1 = pr[:6]
            pr_2 = pr[6:]

            pw_1 = []
            pw_2 = []
            for i in range(6):
                pw_1.append(ImgProcessing.getCameraCoordinate(pl_1[i], pr_1[i]))
                pw_2.append(ImgProcessing.getCameraCoordinate(pl_2[i], pr_2[i]))

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


class Figure_Canvas(FigureCanvas):
    """定义画板 """

    def __init__(self, parent=None, width=3.9, height=2.7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super(Figure_Canvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def test(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [2, 1, 3, 5, 6, 4, 3]
        self.ax.plot(x, y)
