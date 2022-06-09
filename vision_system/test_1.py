import sys, os
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import *
from PySide6.QtGui import *
import cv2


class CV2_PYQT_Window(QDialog):
    def __init__(self, src, target="video", parent=None):
        super(CV2_PYQT_Window, self).__init__()
        self.default_set_height = 660
        if target == "video":
            self.read_video(src)
        else:
            self.read_src(src)

    def read_src(self, src):

        image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        img_height, img_width, channels = image.shape
        if img_height > 800:
            image = cv2.resize(image, dsize=None,fx=(self.default_set_height / img_height), fy=(self.default_set_height / img_height))
        img_height, img_width, channels = image.shape
        bytesPerLine = channels * img_width
        QImg = QImage(image.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)
        label = QLabel()
        label.setPixmap(QtGui.QPixmap.fromImage(QImg))
        button_ok = QPushButton("ok")
        button_no = QPushButton("no")
        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(label)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(button_ok)
        layout2.addWidget(button_no)
        lay = QtWidgets.QVBoxLayout()
        lay.addLayout(layout1)
        lay.addLayout(layout2)
        self.setLayout(lay)
        self.show()

    def read_video(self, file):
        self.frame = 1
        self._file = file
        self.cap = cv2.VideoCapture(file)

        self.success, image = self.cap.read()
        if not self.success:
            return
        cv2.putText(image, str(self.frame), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        img_height, img_width, channels = image.shape
        if img_height > 800:
            image = cv2.resize(image, (int(img_width * self.default_set_height / img_height), self.default_set_height))

        img_height, img_width, channels = image.shape
        bytesPerLine = channels * img_width
        QImg = QImage(image.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)
        self.label = QLabel()
        self.label.setPixmap(QtGui.QPixmap.fromImage(QImg))
        button_ok = QPushButton("ok")
        button_no = QPushButton("no")
        self.timer = QtCore.QTimer()

        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.label)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(button_ok)
        layout2.addWidget(button_no)
        lay = QtWidgets.QVBoxLayout()
        lay.addLayout(layout1)
        lay.addLayout(layout2)
        self.setLayout(lay)
        self.show()

        self.timer.timeout.connect(self.queryFrame)
        button_ok.clicked.connect(self.video_start)
        button_no.clicked.connect(self.video_stop)

    def cv2_display(self, src):
        cv2.imshow('src', src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def video_start(self):
        if (not self.timer.isActive() and not self.success):
            self.cap = cv2.VideoCapture(self._file)
            self.frame = 1
            self.timer.start(35)
        else:
            self.timer.start(35)

    def video_stop(self):
        self.timer.stop()

    def queryFrame(self):
        '''
        循环捕获图片
        '''
        print(self.frame)
        self.frame += 1
        self.success, image = self.cap.read()
        cv2.putText(image, str(self.frame), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        if not self.success:
            self.timer.stop()
            return

        img_height, img_width, channels = image.shape
        if img_height > 800:
            image = cv2.resize(image, (int(img_width * self.default_set_height / img_height), self.default_set_height))

        img_height, img_width, channels = image.shape
        bytesPerLine = channels * img_width
        QImg = QImage(image.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(QImg))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    file = "1.mp4"
    file1 = "pic.webp"
    mainWin = CV2_PYQT_Window(file, target="video")
    mainWin.show()
    sys.exit(app.exec())
