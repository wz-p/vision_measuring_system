import sys

import time
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QTimer

from PyQt6.QtWidgets import QMainWindow, QApplication, QGridLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 解决坐标轴中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号不显示的问题


class Figure_Canvas(FigureCanvas):
    """
    创建画板类
    """

    def __init__(self, width=3.2, height=2.7):
        self.fig = Figure(figsize=(width, height), dpi=70)
        super(Figure_Canvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)  # 111表示1行1列，第一张曲线图

    def add_line(self, x_data, y_data, y2_data=None):
        self.line = Line2D(x_data, y_data)  # 绘制2D折线图

        # ------------------调整折线图基本样式---------------------#

        self.line.set_ls('')  # 设置连线
        self.line.set_marker('o') # 设置每个点
        self.line.set_color('b')  # 设置线条颜色

        self.ax.grid(True)  # 添加网格
        self.ax.set_title('动态曲线')  # 设置标题

        # 设置xy轴最大最小值,找到x_data, y_data最大最小值
        self.ax.set_xlim(np.min(x_data), np.max(x_data))
        self.ax.set_ylim(np.min(y_data), np.max(y_data) + 2)  # y轴稍微多一点，会好看一点

        self.ax.set_xlabel('x坐标')  # 设置坐标名称
        self.ax.set_ylabel('y坐标')

        # 在曲线下方填充颜色
        self.ax.fill_between(x_data, y_data, color='g', alpha=0.1)

        self.ax.legend([self.line], ['sinx'])  # 添加图例

        # ------------------------------------------------------#
        self.ax.add_line(self.line)

        # 绘制第二条曲线
        self.line2 = Line2D(x_data, y2_data)
        self.ax.add_line(self.line2)
        self.line2.set_color('red')  # 设置线条颜色
        self.ax.legend([self.line, self.line2], ['sinx', 'cosx'])  # 添加图例

        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel('y2坐标')


class linewidget(QMainWindow):
    def __init__(self):
        super(linewidget, self).__init__()
        self.setWindowTitle('绘制动态曲线')
        self.resize(1000, 800)

        # 创建一个groupbox, 用来画动态曲线
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(100, 200, 800, 300))

        self.load_line()  # 加载动态曲线

        # 创建定时器，使曲线图动态更新
        self.timer = QTimer()
        self.timer.start(10)
        self.ts = time.time()
        self.timer.timeout.connect(self.Updatedata)

    def load_line(self):
        self.LineFigure = Figure_Canvas()
        self.LineFigureLayout = QGridLayout(self.groupBox)
        self.LineFigureLayout.addWidget(self.LineFigure)

        # 准备数据，绘制曲线
        x_data = np.arange(-4, 4, 0.5)
        y_data = np.sin(x_data)
        y2_data = np.cos(x_data)
        self.LineFigure.add_line(x_data, y_data, y2_data)

    def Updatedata(self):
        dt = time.time() - self.ts
        x_data = np.arange(-4, 4, 0.5)
        z_data = np.sin(x_data - dt)  # 准备动态数据

        h_data = np.cos(x_data - dt)

        self.LineFigure.line.set_ydata(z_data)  # 更新数据
        self.LineFigure.line2.set_ydata(h_data)

        self.LineFigure.draw()  # 重新画图


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainMindow = linewidget()
    mainMindow.show()
    sys.exit(app.exec())
