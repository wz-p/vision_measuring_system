#!/usr/bin/python
# -*- coding: utf-8 -*-
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
                           QFont, QFontDatabase, QGradient, QIcon,
                           QImage, QKeySequence, QLinearGradient, QPainter,
                           QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox,
                               QFrame, QGroupBox, QHBoxLayout, QLCDNumber,
                               QLabel, QLayout, QMainWindow, QMenuBar,
                               QPushButton, QRadioButton, QSizePolicy, QSpinBox,
                               QStatusBar, QVBoxLayout, QWidget, QGridLayout, QTextBrowser)


class Ui_VisionSystem(object):
    def setupUi(self, VisionSystem):
        if not VisionSystem.objectName():
            VisionSystem.setObjectName(u"VisionSystem")
        VisionSystem.setEnabled(True)
        VisionSystem.resize(1366, 768)
        VisionSystem.setMaximumSize(QSize(16777215, 16777215))
        icon = QIcon()
        icon.addFile(u"./src/wz.png", QSize(), QIcon.Normal, QIcon.Off)
        VisionSystem.setWindowIcon(icon)
        VisionSystem.setIconSize(QSize(30, 30))
        self.centralwidget = QWidget(VisionSystem)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_16 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.groupBox.setFont(font)
        self.horizontalLayout_13 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_src = QLabel(self.groupBox)
        self.label_src.setObjectName(u"label_src")
        sizePolicy.setHeightForWidth(self.label_src.sizePolicy().hasHeightForWidth())
        self.label_src.setSizePolicy(sizePolicy)
        self.label_src.setFont(font)
        self.label_src.setFrameShape(QFrame.NoFrame)

        self.horizontalLayout_13.addWidget(self.label_src)

        self.verticalLayout.addWidget(self.groupBox)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.horizontalLayout_14 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_ret = QLabel(self.groupBox_2)
        self.label_ret.setObjectName(u"label_ret")
        sizePolicy.setHeightForWidth(self.label_ret.sizePolicy().hasHeightForWidth())
        self.label_ret.setSizePolicy(sizePolicy)

        self.horizontalLayout_14.addWidget(self.label_ret)

        self.horizontalLayout_2.addWidget(self.groupBox_2)
##############################################################################
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setFont(font)
        self.horizontalLayout_15 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")

        self.label_aix = QLabel(self.groupBox_3)
        self.label_aix.setObjectName(u"label_aix")
        sizePolicy.setHeightForWidth(self.label_aix.sizePolicy().hasHeightForWidth())
        self.label_aix.setSizePolicy(sizePolicy)
##############################################################################
        self.horizontalLayout_15.addWidget(self.label_aix)

        self.horizontalLayout_2.addWidget(self.groupBox_3)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)

        self.horizontalLayout_16.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setSpacing(8)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.label_logo = QLabel(self.centralwidget)
        self.label_logo.setObjectName(u"label_logo")
        self.label_logo.setMaximumSize(QSize(1111111, 1111111))
        self.label_logo.setFrameShape(QFrame.NoFrame)
        self.label_logo.setPixmap(QPixmap(u"./src/ysulogo.png"))
        self.label_logo.setScaledContents(False)
        self.label_logo.setMargin(10)

        self.verticalLayout_2.addWidget(self.label_logo)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.button_openfile = QPushButton(self.centralwidget)
        self.button_openfile.setObjectName(u"button_openfile")
        self.button_openfile.setMinimumSize(QSize(0, 30))
        self.button_openfile.setMaximumSize(QSize(150, 16777215))
        font1 = QFont()
        font1.setFamilies([u"\u5fae\u8f6f\u96c5\u9ed1"])
        font1.setPointSize(12)
        font1.setBold(True)
        self.button_openfile.setFont(font1)
        self.button_openfile.setCursor(QCursor(Qt.PointingHandCursor))
        self.button_openfile.setAutoFillBackground(False)
        self.button_openfile.setStyleSheet(u"")
        icon1 = QIcon()
        icon1.addFile(u"./src/icons8-openfile-144.png", QSize(), QIcon.Normal, QIcon.Off)
        self.button_openfile.setIcon(icon1)
        self.button_openfile.setIconSize(QSize(20, 20))
        self.button_openfile.setCheckable(False)
        self.button_openfile.setFlat(False)

        self.horizontalLayout.addWidget(self.button_openfile)

        self.button_opencamera = QPushButton(self.centralwidget)
        self.button_opencamera.setObjectName(u"button_opencamera")
        self.button_opencamera.setMinimumSize(QSize(0, 30))
        self.button_opencamera.setMaximumSize(QSize(150, 16777215))
        self.button_opencamera.setFont(font1)
        self.button_opencamera.setCursor(QCursor(Qt.PointingHandCursor))
        self.button_opencamera.setAutoFillBackground(False)
        self.button_opencamera.setStyleSheet(u"")
        icon2 = QIcon()
        icon2.addFile(u"./src/camera.webp", QSize(), QIcon.Normal, QIcon.Off)
        self.button_opencamera.setIcon(icon2)
        self.button_opencamera.setIconSize(QSize(20, 20))
        self.button_opencamera.setFlat(False)

        self.horizontalLayout.addWidget(self.button_opencamera)

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.button_begin = QPushButton(self.centralwidget)
        self.button_begin.setObjectName(u"button_begin")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.button_begin.sizePolicy().hasHeightForWidth())
        self.button_begin.setSizePolicy(sizePolicy1)
        self.button_begin.setMinimumSize(QSize(0, 30))
        self.button_begin.setMaximumSize(QSize(150, 16777215))
        font2 = QFont()
        font2.setFamilies([u"\u5fae\u8f6f\u96c5\u9ed1"])
        font2.setPointSize(12)
        font2.setBold(True)
        font2.setKerning(True)
        font2.setStyleStrategy(QFont.PreferDefault)
        self.button_begin.setFont(font2)
        self.button_begin.setCursor(QCursor(Qt.PointingHandCursor))
        self.button_begin.setMouseTracking(False)
        self.button_begin.setFocusPolicy(Qt.StrongFocus)
        self.button_begin.setAutoFillBackground(False)
        self.button_begin.setStyleSheet(
            u"background:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0.437113 rgba(164, 222, 255, 255), stop:1 rgba(255, 255, 255, 255));\n"
            "")
        icon3 = QIcon()
        icon3.addFile(u"./src/begin.webp", QSize(), QIcon.Normal, QIcon.Off)
        self.button_begin.setIcon(icon3)
        self.button_begin.setIconSize(QSize(22, 22))
        self.button_begin.setFlat(False)

        self.horizontalLayout_8.addWidget(self.button_begin)

        self.button_pause = QPushButton(self.centralwidget)
        self.button_pause.setObjectName(u"button_pause")
        self.button_pause.setMinimumSize(QSize(0, 30))
        self.button_pause.setMaximumSize(QSize(150, 16777215))
        self.button_pause.setFont(font1)
        self.button_pause.setCursor(QCursor(Qt.PointingHandCursor))
        self.button_pause.setAutoFillBackground(False)
        self.button_pause.setStyleSheet(
            u"background:qlineargradient(spread:pad, x1:0, y1:0.501, x2:1, y2:0.495, stop:0.321429 rgba(255, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));\n"
            "")
        icon4 = QIcon()
        icon4.addFile(u"./src/icons8-close-80.png", QSize(), QIcon.Normal, QIcon.Off)
        self.button_pause.setIcon(icon4)
        self.button_pause.setIconSize(QSize(20, 20))
        self.button_pause.setCheckable(False)
        self.button_pause.setChecked(False)
        self.button_pause.setAutoDefault(False)
        self.button_pause.setFlat(False)

        self.horizontalLayout_8.addWidget(self.button_pause)

        self.verticalLayout_2.addLayout(self.horizontalLayout_8)

        self.groupBox_7 = QGroupBox(self.centralwidget)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setFont(font)
        self.horizontalLayout_3 = QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.radioButton = QRadioButton(self.groupBox_7)
        self.buttonGroup = QButtonGroup(VisionSystem)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.setExclusive(True)
        self.buttonGroup.addButton(self.radioButton)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.radioButton.sizePolicy().hasHeightForWidth())
        self.radioButton.setSizePolicy(sizePolicy1)
        font3 = QFont()
        font3.setPointSize(10)
        font3.setBold(True)
        self.radioButton.setFont(font3)
        self.radioButton.setMouseTracking(True)
        self.radioButton.setTabletTracking(True)
        self.radioButton.setIconSize(QSize(16, 16))
        self.radioButton.setCheckable(True)
        self.radioButton.setChecked(True)
        self.radioButton.setAutoRepeat(False)

        self.horizontalLayout_11.addWidget(self.radioButton)

        self.radioButton_2 = QRadioButton(self.groupBox_7)
        self.buttonGroup.addButton(self.radioButton_2)
        self.radioButton_2.setObjectName(u"radioButton_2")
        self.radioButton_2.setFont(font3)

        self.horizontalLayout_11.addWidget(self.radioButton_2)

        self.radioButton_3 = QRadioButton(self.groupBox_7)
        self.buttonGroup.addButton(self.radioButton_3)
        self.radioButton_3.setObjectName(u"radioButton_3")
        self.radioButton_3.setFont(font3)

        self.horizontalLayout_11.addWidget(self.radioButton_3)

        self.horizontalLayout_3.addLayout(self.horizontalLayout_11)

        self.verticalLayout_2.addWidget(self.groupBox_7)

        self.groupBox_5 = QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName(u"groupBox_5")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy2)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
        self.horizontalLayout_9 = QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(-1, 4, -1, 4)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.groupBox_5)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(0, 0))
        self.label.setMaximumSize(QSize(16777215, 16777215))
        font4 = QFont()
        font4.setFamilies([u"\u534e\u6587\u7425\u73c0"])
        font4.setPointSize(12)
        font4.setBold(False)
        font4.setItalic(False)
        font4.setUnderline(False)
        font4.setKerning(True)
        self.label.setFont(font4)
        self.label.setCursor(QCursor(Qt.ArrowCursor))
        self.label.setTextFormat(Qt.AutoText)
        self.label.setMargin(5)

        self.horizontalLayout_4.addWidget(self.label)

        self.label_3 = QLabel(self.groupBox_5)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMaximumSize(QSize(16777215, 16777215))
        font5 = QFont()
        font5.setPointSize(10)
        font5.setBold(True)
        font5.setKerning(True)
        self.label_3.setFont(font5)
        self.label_3.setCursor(QCursor(Qt.ArrowCursor))
        self.label_3.setFrameShape(QFrame.NoFrame)
        self.label_3.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.label_3.setMargin(5)

        self.horizontalLayout_4.addWidget(self.label_3)

        self.H_max = QSpinBox(self.groupBox_5)
        self.H_max.setObjectName(u"H_max")
        self.H_max.setMaximumSize(QSize(16777215, 16777215))
        font6 = QFont()
        font6.setPointSize(11)
        font6.setBold(False)
        self.H_max.setFont(font6)
        self.H_max.setCursor(QCursor(Qt.ArrowCursor))
        self.H_max.setMaximum(255)
        self.H_max.setStepType(QAbstractSpinBox.DefaultStepType)
        self.H_max.setValue(180)
        self.H_max.setDisplayIntegerBase(10)

        self.horizontalLayout_4.addWidget(self.H_max)

        self.label_4 = QLabel(self.groupBox_5)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(16777215, 16777215))
        self.label_4.setFont(font5)
        self.label_4.setCursor(QCursor(Qt.ArrowCursor))
        self.label_4.setFrameShape(QFrame.NoFrame)
        self.label_4.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.label_4.setMargin(5)

        self.horizontalLayout_4.addWidget(self.label_4)

        self.S_max = QSpinBox(self.groupBox_5)
        self.S_max.setObjectName(u"S_max")
        self.S_max.setMaximumSize(QSize(16777215, 16777215))
        self.S_max.setFont(font6)
        self.S_max.setMaximum(255)
        self.S_max.setStepType(QAbstractSpinBox.DefaultStepType)
        self.S_max.setValue(255)
        self.S_max.setDisplayIntegerBase(10)

        self.horizontalLayout_4.addWidget(self.S_max)

        self.label_5 = QLabel(self.groupBox_5)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMaximumSize(QSize(16777215, 16777215))
        self.label_5.setFont(font5)
        self.label_5.setCursor(QCursor(Qt.ArrowCursor))
        self.label_5.setFrameShape(QFrame.NoFrame)
        self.label_5.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.label_5.setMargin(5)

        self.horizontalLayout_4.addWidget(self.label_5)

        self.V_max = QSpinBox(self.groupBox_5)
        self.V_max.setObjectName(u"V_max")
        self.V_max.setMaximumSize(QSize(16777215, 16777215))
        self.V_max.setFont(font6)
        self.V_max.setMaximum(255)
        self.V_max.setStepType(QAbstractSpinBox.DefaultStepType)
        self.V_max.setValue(255)
        self.V_max.setDisplayIntegerBase(10)

        self.horizontalLayout_4.addWidget(self.V_max)

        self.horizontalLayout_4.setStretch(0, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_2 = QLabel(self.groupBox_5)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 0))
        font7 = QFont()
        font7.setFamilies([u"\u534e\u6587\u7425\u73c0"])
        font7.setPointSize(12)
        font7.setBold(False)
        font7.setItalic(False)
        font7.setUnderline(False)
        self.label_2.setFont(font7)
        self.label_2.setTextFormat(Qt.AutoText)
        self.label_2.setMargin(5)

        self.horizontalLayout_5.addWidget(self.label_2)

        self.label_6 = QLabel(self.groupBox_5)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font3)
        self.label_6.setFrameShape(QFrame.NoFrame)
        self.label_6.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.label_6.setMargin(5)

        self.horizontalLayout_5.addWidget(self.label_6)

        self.H_min = QSpinBox(self.groupBox_5)
        self.H_min.setObjectName(u"H_min")
        self.H_min.setFont(font6)
        self.H_min.setMaximum(255)
        self.H_min.setStepType(QAbstractSpinBox.DefaultStepType)
        self.H_min.setValue(165)
        self.H_min.setDisplayIntegerBase(10)

        self.horizontalLayout_5.addWidget(self.H_min)

        self.label_7 = QLabel(self.groupBox_5)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font3)
        self.label_7.setFrameShape(QFrame.NoFrame)
        self.label_7.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.label_7.setMargin(5)

        self.horizontalLayout_5.addWidget(self.label_7)

        self.S_min = QSpinBox(self.groupBox_5)
        self.S_min.setObjectName(u"S_min")
        self.S_min.setFont(font6)
        self.S_min.setMaximum(255)
        self.S_min.setStepType(QAbstractSpinBox.DefaultStepType)
        self.S_min.setValue(55)
        self.S_min.setDisplayIntegerBase(10)

        self.horizontalLayout_5.addWidget(self.S_min)

        self.label_8 = QLabel(self.groupBox_5)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font3)
        self.label_8.setFrameShape(QFrame.NoFrame)
        self.label_8.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.label_8.setMargin(5)

        self.horizontalLayout_5.addWidget(self.label_8)

        self.V_min = QSpinBox(self.groupBox_5)
        self.V_min.setObjectName(u"V_min")
        self.V_min.setFont(font6)
        self.V_min.setMaximum(255)
        self.V_min.setStepType(QAbstractSpinBox.DefaultStepType)
        self.V_min.setValue(90)
        self.V_min.setDisplayIntegerBase(10)

        self.horizontalLayout_5.addWidget(self.V_min)

        self.horizontalLayout_5.setStretch(0, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 1)

        self.horizontalLayout_9.addLayout(self.verticalLayout_3)

        self.verticalLayout_2.addWidget(self.groupBox_5)

        self.groupBox_6 = QGroupBox(self.centralwidget)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setFont(font)
        self.horizontalLayout_10 = QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_9 = QLabel(self.groupBox_6)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setFont(font3)
        self.label_9.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)

        self.horizontalLayout_6.addWidget(self.label_9)

        self.area_u = QSpinBox(self.groupBox_6)
        self.area_u.setObjectName(u"area_u")
        self.area_u.setFont(font6)
        self.area_u.setMaximum(5555)
        self.area_u.setSingleStep(20)
        self.area_u.setStepType(QAbstractSpinBox.DefaultStepType)
        self.area_u.setValue(500)

        self.horizontalLayout_6.addWidget(self.area_u)

        self.label_10 = QLabel(self.groupBox_6)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setFont(font3)
        self.label_10.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)

        self.horizontalLayout_6.addWidget(self.label_10)

        self.area_v = QSpinBox(self.groupBox_6)
        self.area_v.setObjectName(u"area_v")
        self.area_v.setFont(font6)
        self.area_v.setMaximum(5555)
        self.area_v.setSingleStep(20)
        self.area_v.setValue(150)

        self.horizontalLayout_6.addWidget(self.area_v)

        self.label_11 = QLabel(self.groupBox_6)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font3)
        self.label_11.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)

        self.horizontalLayout_6.addWidget(self.label_11)

        self.area_w = QSpinBox(self.groupBox_6)
        self.area_w.setObjectName(u"area_w")
        self.area_w.setFont(font6)
        self.area_w.setMaximum(5555)
        self.area_w.setSingleStep(20)
        self.area_w.setValue(1800)

        self.horizontalLayout_6.addWidget(self.area_w)

        self.label_12 = QLabel(self.groupBox_6)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font3)
        self.label_12.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)

        self.horizontalLayout_6.addWidget(self.label_12)

        self.area_h = QSpinBox(self.groupBox_6)
        self.area_h.setObjectName(u"area_h")
        self.area_h.setFont(font6)
        self.area_h.setMaximum(5555)
        self.area_h.setSingleStep(20)
        self.area_h.setStepType(QAbstractSpinBox.DefaultStepType)
        self.area_h.setValue(1200)

        self.horizontalLayout_6.addWidget(self.area_h)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_6)

        self.verticalLayout_2.addWidget(self.groupBox_6)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setFont(font)
        self.groupBox_4.setStyleSheet(u"")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.checkBox = QCheckBox(self.groupBox_4)
        self.checkBox.setObjectName(u"checkBox")
        font8 = QFont()
        font8.setPointSize(10)
        font8.setBold(False)
        self.checkBox.setFont(font8)
        self.checkBox.setChecked(True)

        self.horizontalLayout_7.addWidget(self.checkBox)

        self.label_13 = QLabel(self.groupBox_4)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setFont(font3)

        self.horizontalLayout_7.addWidget(self.label_13)

        self.lcdNumber = QLCDNumber(self.groupBox_4)
        self.lcdNumber.setObjectName(u"lcdNumber")
        self.lcdNumber.setEnabled(True)
        self.lcdNumber.setMaximumSize(QSize(50, 16777215))
        font9 = QFont()
        font9.setFamilies([u"\u96b6\u4e66"])
        font9.setPointSize(12)
        font9.setBold(True)
        self.lcdNumber.setFont(font9)
        self.lcdNumber.setFrameShape(QFrame.NoFrame)
        self.lcdNumber.setFrameShadow(QFrame.Plain)
        self.lcdNumber.setDigitCount(2)
        self.lcdNumber.setSegmentStyle(QLCDNumber.Flat)

        self.horizontalLayout_7.addWidget(self.lcdNumber)

        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(2, 1)

        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
# TODO 更改文本显示控件

        self.textBroser = QTextBrowser(self.groupBox_4)
        self.textBroser.setObjectName(u"textBroser")
        self.verticalLayout_4.addWidget(self.textBroser)

#####################################################################
        # self.webEngineView = QWebEngineView(self.groupBox_4)
        # self.webEngineView.setObjectName(u"webEngineView")
        # # self.webEngineView.setUrl(QUrl(u"file:///./src/textBrowser_coor.html"))
        # self.webEngineView.setZoomFactor(1.000000000000000)
        #
        # self.verticalLayout_4.addWidget(self.webEngineView)
#############################################################################
        self.verticalLayout_4.setStretch(1, 1)

        self.verticalLayout_2.addWidget(self.groupBox_4)

        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 1)
        self.verticalLayout_2.setStretch(4, 2)
        self.verticalLayout_2.setStretch(5, 1)
        self.verticalLayout_2.setStretch(6, 5)

        self.horizontalLayout_16.addLayout(self.verticalLayout_2)

        self.horizontalLayout_16.setStretch(0, 7)
        self.horizontalLayout_16.setStretch(1, 1)
        VisionSystem.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(VisionSystem)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 746, 22))
        VisionSystem.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(VisionSystem)
        self.statusbar.setObjectName(u"statusbar")
        VisionSystem.setStatusBar(self.statusbar)
        # if QT_CONFIG(shortcut)
        # endif // QT_CONFIG(shortcut)

        self.retranslateUi(VisionSystem)

        self.button_pause.setDefault(False)

        QMetaObject.connectSlotsByName(VisionSystem)

        self.label_src.setScaledContents(True)
        self.label_ret.setScaledContents(True)
        self.label_aix.setScaledContents(True)  # 自适应显示图片大小

    # setupUi

    def retranslateUi(self, VisionSystem):
        VisionSystem.setWindowTitle(QCoreApplication.translate("VisionSystem",
                                                               u"\u8f6f\u4f53\u673a\u5668\u4eba\u7a7a\u95f4\u4f4d\u59ff\u6d4b\u91cf\u7cfb\u7edf",
                                                               None))
        self.groupBox.setTitle(
            QCoreApplication.translate("VisionSystem", u"\u5de6\u53f3\u89c6\u56fe\u56fe\u50cf", None))
        self.label_src.setText("")
        self.groupBox_2.setTitle(
            QCoreApplication.translate("VisionSystem", u"\u6807\u5fd7\u70b9\u8bc6\u522b\u7ed3\u679c", None))
        self.label_ret.setText("")
        self.groupBox_3.setTitle(
            QCoreApplication.translate("VisionSystem", u"\u6807\u5fd7\u70b9\u7a7a\u95f4\u4f4d\u7f6e", None))
        self.label_aix.setText("")
        # if QT_CONFIG(whatsthis)
        self.label_logo.setWhatsThis(
            QCoreApplication.translate("VisionSystem", u"<html><head/><body><p><br/></p></body></html>", None))
        # endif // QT_CONFIG(whatsthis)
        self.label_logo.setText("")
        self.button_openfile.setText(QCoreApplication.translate("VisionSystem", u"\u6253\u5f00\u6587\u4ef6", None))
        self.button_opencamera.setText(QCoreApplication.translate("VisionSystem", u"\u6253\u5f00\u76f8\u673a", None))
        # if QT_CONFIG(tooltip)
        self.button_begin.setToolTip("")
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(whatsthis)
        self.button_begin.setWhatsThis("")
        # endif // QT_CONFIG(whatsthis)
        self.button_begin.setText(QCoreApplication.translate("VisionSystem", u"\u5f00\u59cb\u68c0\u6d4b", None))
        self.button_pause.setText(QCoreApplication.translate("VisionSystem", u"\u6682\u505c", None))
        self.groupBox_7.setTitle(
            QCoreApplication.translate("VisionSystem", u"\u6d4b\u91cf\u5bf9\u8c61\u9009\u62e9", None))
        self.radioButton.setText(QCoreApplication.translate("VisionSystem", u"\u6d4b\u91cf\u5bf9\u8c611", None))
        self.radioButton_2.setText(QCoreApplication.translate("VisionSystem", u"\u6d4b\u91cf\u5bf9\u8c612", None))
        self.radioButton_3.setText(QCoreApplication.translate("VisionSystem", u"\u6d4b\u91cf\u5bf9\u8c613", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("VisionSystem", u"HSV\u9608\u503c", None))
        self.label.setText(QCoreApplication.translate("VisionSystem", u"Max", None))
        self.label_3.setText(QCoreApplication.translate("VisionSystem", u"H", None))
        self.label_4.setText(QCoreApplication.translate("VisionSystem", u"S", None))
        self.label_5.setText(QCoreApplication.translate("VisionSystem", u"V", None))
        self.label_2.setText(QCoreApplication.translate("VisionSystem", u"Min", None))
        self.label_6.setText(QCoreApplication.translate("VisionSystem", u"H", None))
        self.label_7.setText(QCoreApplication.translate("VisionSystem", u"S", None))
        self.label_8.setText(QCoreApplication.translate("VisionSystem", u"V", None))
        self.groupBox_6.setTitle(
            QCoreApplication.translate("VisionSystem", u"\u5f85\u68c0\u533a\u57df\u9884\u8bbe", None))
        self.label_9.setText(QCoreApplication.translate("VisionSystem", u"u", None))
        self.area_u.setSuffix("")
        self.area_u.setPrefix("")
        self.label_10.setText(QCoreApplication.translate("VisionSystem", u"v", None))
        self.label_11.setText(QCoreApplication.translate("VisionSystem", u"w", None))
        self.label_12.setText(QCoreApplication.translate("VisionSystem", u"h", None))
        self.groupBox_4.setTitle(
            QCoreApplication.translate("VisionSystem", u"\u6807\u5fd7\u70b9\u6570\u91cf\u53ca\u7a7a\u95f4\u5750\u6807",
                                       None))
        self.checkBox.setText(
            QCoreApplication.translate("VisionSystem", u"\u8f93\u51fa\u6807\u5fd7\u70b9\u7a7a\u95f4\u5750\u6807", None))
        self.label_13.setText(QCoreApplication.translate("VisionSystem", u"\u6807\u5fd7\u70b9\u6570\u91cf", None))
    # retranslateUi
