#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

from PySide6.QtWidgets import QMessageBox, QFileDialog
# from PySide6.QtCore import QTimer
# from PySide6 import QtCore
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation as animation
# import cv2
import random
from visionSystem import *
from imgProcessing import *


class VisionSystem_ui(QMainWindow, Ui_VisionSystem):

    def __init__(self):
        super(VisionSystem_ui, self).__init__()
        self.setupUi(self)

        self.filePath = None
        self.state = None  # ['video', 'image', 'camera']
        self.target = '测量对象1'

        self.button_openfile.clicked.connect(self.openFile)
        self.button_opencamera.clicked.connect(self.openCamera)
        self.buttonGroup.buttonClicked.connect(self.changeType)
        self.ts = time.time()

        for i in [self.H_max, self.S_max, self.V_max, self.H_min, self.S_min, self.V_min,
                  self.area_u, self.area_v, self.area_w, self.area_h]:
            i.valueChanged.connect(self.changeSpinBoxNum)

        # with open('./src/textBrowser_coor.html', 'w') as f:
        #     f.write('')

    def PrepareSamples(self):
        self.x = np.random.randint(-100, 100, 20)
        self.y = np.random.randint(-100, 100, 20)
        self.z = np.sin(self.x)
        self.d = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def save_poly(self):

        plt.figure("3D Scatter", facecolor="lightgray")
        ax3d = plt.subplot(projection="3d")  # 创建三维坐标
        # ax3d.view_init(20, -120)    # 初始化坐标轴视图方向
        ax3d.set_xlim3d(min(self.x) - 50, min(self.x) + 150)
        ax3d.set_ylim3d(min(self.y) - 50, min(self.y) + 150)
        ax3d.set_zlim3d(350, 550)
        # plt.title('3D Scatter', fontsize=20)
        ax3d.set_xlabel('x', fontsize=14)
        ax3d.set_ylabel('y', fontsize=14)
        ax3d.set_zlabel('z', fontsize=14)
        plt.tick_params(labelsize=10)
        ax3d.scatter(self.x, self.y, self.z, s=15, c=self.d, cmap="jet", marker="o")
        plt.savefig('./temp/test2.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

    def remove_poly(self):
        os.remove('./temp/test2.jpg')

    def changeSpinBoxNum(self):
        print(self.H_max.value(), '\t', self.S_max.value(), '\t', self.V_max.value())
        print(self.H_min.value(), '\t', self.S_min.value(), '\t', self.V_min.value())
        print(self.area_u.value(), '\t', self.area_v.value(), '\t', self.area_w.value(), '\t', self.area_h.value(), )

    def changeType(self):
        if self.buttonGroup.checkedId() == -2:
            self.target = '测量对象1'  # 长手爪
        elif self.buttonGroup.checkedId() == -3:
            self.target = '测量对象2'  # 短手抓
        elif self.buttonGroup.checkedId() == -4:
            self.target = '测量对象3'  # 人工肌肉机械臂

    def openCamera(self):
        self.state = 'camera'
        self.read_video('c')

    def openFile(self):
        self.filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "请选择视频或图片",  # 标题
            r"./",  # 起始目录
            "All files (*.*);;Image files(*.bmp, *.png, *.jpg, *.jpeg, *.tif, *.tiff);;Video files(*.avi, *.mp4)"
            # 选择类型过滤项，过滤内容在括号中
        )

        if self.filePath.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            self.state = 'image'
            self.button_begin.clicked.connect(self.begin)

            self.image = cv2.imread(self.filePath, cv2.IMREAD_UNCHANGED)
            img_height, img_width, channels = self.image.shape
            bytesPerLine = channels * img_width
            QImg = QImage(self.image.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)
            self.label_src.clear()
            self.label_src.setPixmap(QPixmap.fromImage(QImg))

        elif self.filePath.lower().endswith(('.avi', 'mp4')):
            self.state = 'video'
            self.read_video('file')

        elif self.filePath == '':
            pass

        else:
            QMessageBox.critical(self, '错误', '所选文件类型错误')

    def read_video(self, src):
        if src == 'file':
            self.cap = cv2.VideoCapture(self.filePath)
        else:
            self.cap = cv2.VideoCapture(0)
        self.frame_num = 1
        self.success, self.frame = self.cap.read()
        if not self.success:
            return
        img_height, img_width, channels = self.frame.shape
        bytesPerLine = channels * img_width
        QImg = QImage(self.frame.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)

        # 显示原始双目图像
        self.label_src.setPixmap(QPixmap.fromImage(QImg))

        self.timer1 = QTimer()

        self.timer1.timeout.connect(self.queryFrame)
        self.button_begin.clicked.connect(self.video_start)
        self.button_pause.clicked.connect(self.video_stop)

    def begin(self):
        if self.state == 'image':
            img_height, img_width, channels = self.image.shape
            bytesPerLine = channels * img_width
            QImg = QImage(self.image.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)

            # 显示检测结果图像
            self.label_ret.clear()
            self.label_ret.setPixmap(QPixmap.fromImage(QImg))

    def video_start(self):
        if not self.timer1.isActive() and not self.success:
            self.cap = cv2.VideoCapture(self.filePath)
            self.frame_num = 1
            self.timer1.start(0)
        else:
            self.timer1.start(0)

    def video_stop(self):
        self.timer1.stop()

    def queryFrame(self):
        """循环获取图片"""

        def put_text(img):
            cv2.putText(img, str('Frame') + str(self.frame_num),
                        (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (255, 255, 255), 3)

        self.frame_num += 1
        self.success, frame = self.cap.read()

        if not self.success:
            self.timer1.stop()
            return
        self.param_value = [(self.H_min.value(), self.S_min.value(), self.V_min.value()),
                            (self.H_max.value(), self.S_max.value(), self.V_max.value()),
                            (self.area_u.value(), self.area_v.value()),
                            (self.area_w.value(), self.area_h.value())]
        self.frequency = random.randint(4, 10)
        if self.frame_num % self.frequency == 0:
            """每隔self.frequency帧处理一次图片"""
            self.img_left, self.img_right = ImgProcessing.imageSplit(frame)

            # 畸变矫正和立体校正
            zed = ZED_config()
            self.img_left = ZED_config.undistort(self.img_left, zed.left_map1, zed.left_map2)
            self.img_right = ZED_config.undistort(self.img_right, zed.right_map1, zed.right_map2)

            put_text(frame)

            ret_left, point_left = ImgProcessing.getProcessingRet(self.img_left, self.param_value)
            put_text(ret_left)
            _, point_right = ImgProcessing.getProcessingRet(self.img_right, self.param_value)
            self.x, self.y, self.z = ImgProcessing.getCameraCoordinate(point_left, point_right)
            self.d = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

            # 网页不刷新，改为textBroser
            self.textBroser.clear()
            coor = np.round(np.array([self.x, self.y, self.z]).T, 2)
            self.textBroser.setText('X(mm)  ' + 'Y(mm)  ' + 'Z(mm)\n' + str(coor))

            img_height, img_width, channels = self.frame.shape
            bytesPerLine = channels * img_width
            QImg = QImage(frame.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)
            self.label_src.setPixmap(QPixmap.fromImage(QImg))

            QImg_ret = QImage(ret_left.data, int(img_width / 2), img_height, int(3 * img_width / 2),
                              QImage.Format_BGR888)

            # self.PrepareSamples()
            self.save_poly()
            aix_img = cv2.imread('./temp/test2.jpg')
            h, w, c = aix_img.shape
            b_P = c * w
            QImg_aix = QImage(aix_img.data, w, h, b_P, QImage.Format_BGR888)

            self.label_ret.setPixmap(QPixmap.fromImage(QImg_ret))
            self.label_aix.setPixmap(QPixmap.fromImage(QImg_aix))
            self.lcdNumber.display(len(point_left))


app = QApplication()
stats = VisionSystem_ui()
stats.show()
app.exec()
