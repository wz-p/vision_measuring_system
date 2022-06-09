# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DataDisplayUI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.LineDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.LineDisplayGB.setObjectName("LineDisplayGB")
        self.gridLayout.addWidget(self.LineDisplayGB, 0, 0, 1, 1)
        self.BarDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.BarDisplayGB.setObjectName("BarDisplayGB")
        self.gridLayout.addWidget(self.BarDisplayGB, 0, 1, 1, 1)
        self.ImageDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.ImageDisplayGB.setObjectName("ImageDisplayGB")
        self.gridLayout.addWidget(self.ImageDisplayGB, 1, 0, 1, 1)
        self.SurfaceDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.SurfaceDisplayGB.setObjectName("SurfaceDisplayGB")
        self.gridLayout.addWidget(self.SurfaceDisplayGB, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.LineDisplayGB.setTitle(_translate("MainWindow", "Line Display"))
        self.BarDisplayGB.setTitle(_translate("MainWindow", "Bar Display"))
        self.ImageDisplayGB.setTitle(_translate("MainWindow", "Image Display"))
        self.SurfaceDisplayGB.setTitle(_translate("MainWindow", "3D Surface Display"))


from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout
from PyQt6.QtCore import QTimer
import sys, time
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.cbook as cbook


class ImgDisp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ImgDisp, self).__init__(parent)
        self.setupUi(self)


class Figure_Canvas(FigureCanvas):
    def __init__(self, parent=None, width=3.9, height=2.7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super(Figure_Canvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def test(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [2, 1, 3, 5, 6, 4, 3]
        self.ax.plot(x, y)


class ImgDisp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ImgDisp, self).__init__(parent)
        self.setupUi(self)
        self.Init_Widgets()

        self.timer = QTimer()
        self.timer.start(1)
        self.ts = time.time()

        self.timer.timeout.connect(self.UpdateImgs)

    def Init_Widgets(self):
        self.PrepareSamples()
        self.PrepareSurfaceCanvas()

    def PrepareSamples(self):
        self.x = np.arange(-4, 4, 1)
        self.y = np.arange(-4, 4, 1)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.z = np.sin(self.x)
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2)
        self.Z = np.sin(self.R)

    def PrepareSurfaceCanvas(self):
        self.SurfFigure = Figure_Canvas()
        self.SurfFigureLayout = QGridLayout(self.SurfaceDisplayGB)
        self.SurfFigureLayout.addWidget(self.SurfFigure)
        self.SurfFigure.ax.remove()
        self.ax3d = self.SurfFigure.fig.add_subplot(projection='3d')
        self.Surf = self.ax3d.plot_surface(self.X, self.Y, self.Z, cmap='rainbow')

    def UpdateImgs(self):
        dt = time.time() - self.ts
        # self.LineUpdate(dt)
        # self.BarUpdate(dt)
        # self.ImgUpdate(dt)
        self.SurfUpdate(dt)

    def SurfUpdate(self, dt):
        X = self.X + dt
        Y = self.Y + dt
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.sin(R)
        polys = self.Get3dVerts(self.X, self.Y, Z)
        self.Surf.set_verts(polys)
        self.SurfFigure.draw()

    def Get3dVerts(self, X, Y, Z):
        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape
        rcount = 50
        ccount = 50
        rstride = int(max(np.ceil(rows / rcount), 1))
        cstride = int(max(np.ceil(cols / ccount), 1))
        # evenly spaced, and including both endpoints
        row_inds = list(range(0, rows - 1, rstride)) + [rows - 1]
        col_inds = list(range(0, cols - 1, cstride)) + [cols - 1]
        polys = []
        for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
            for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
                ps = [
                    # +1 ensures we share edges between polygons
                    cbook._array_perimeter(a[rs:rs_next + 1, cs:cs_next + 1])
                    for a in (X, Y, Z)
                ]
                # ps = np.stack(ps, axis=-1)
                ps = np.array(ps).T
                polys.append(ps)
        return polys


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = ImgDisp()
    ui.show()
    sys.exit(app.exec())
