import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt
from scipy import interpolate
from image_preprocessing import imageSplit, getPixPoint
from mpl_toolkits.mplot3d import Axes3D


def dynamic_plt():
    x = list(range(1, 100))  # epoch array
    # loss = [10 / (i**2) for i in x]  # loss values array
    y2 = [i ** 2 for i in x]
    y3 = [i ** 3 for i in x]
    y4 = [i ** 4 for i in x]
    y5 = [i ** 5 for i in x]
    y6 = [i ** 6 for i in x]
    y7 = [i ** 7 for i in x]

    y = [y2, y3, y4, y5, y6, y7]
    plt.ion()

    for i in range(6):
        plt.title("loss")
        plt.plot(x, y[i])

        plt.xlabel("epoch")
        plt.ylabel("loss")

        # plt.xlim(0,len(x)) #固定x轴
        if i == 1:
            plt.pause(1)  # 启动时间，方便截屏
        plt.pause(1)
        plt.cla()
    plt.ioff()
    plt.show()


def nothing():
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    img1 = cv.imread('myleft.jpg', 0)  # 索引图像 # left image
    img2 = cv.imread('myright.jpg', 0)  # 训练图像 # right image
    sift = cv.SIFT()  # 使用SIFT查找关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2,
                                      None)  # FLANN 参数FLANN_INDEX_KDTREE = 1index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)search_params = dict(checks=50)flann = cv.FlannBasedMatcher(index_params,search_params)matches = flann.knnMatch(des1,des2,k=2)good = []pts1 = []pts2 = []# 根据Lowe的论文进行比率测试for i,(m,n) in enumerate(matches): if m.distance < 0.8*n.distance: good.append(m) pts2.append(kp2[m.trainIdx].pt) pts1.append(kp1[m.queryIdx].pt)


def show2D():
    point = [[85.0473, 112.7322, 441.2153],
             [97.7168, 110.1424, 440.9912],
             [72.1235, 109.3822, 440.2387],
             [109.4031, 104.2037, 441.7863],
             [60.7102, 102.1624, 439.83],
             [116.6687, 94.8319, 443.4596],
             [52.4, 88.9331, 438.5008],
             [119.0515, 81.0491, 443.171],
             [48.6851, 75.9596, 438.7289],
             [114.5955, 69.8285, 442.1298],
             [49.3119, 63.4969, 438.2798],
             [105.263, 62.1415, 442.2687],
             [50.6698, 50.663, 437.1106]]
    point = np.array(point)
    left_point = point[point[:, 0] < point[0, 0]]
    left_point = left_point[::-1]
    right_point = point[point[:, 0] >= point[0, 0]]
    new_point = np.vstack((left_point, right_point))
    x = new_point[:, 0]
    y = new_point[:, 1]

    tck, u = splprep([x, y], s=0)
    xnew, ynew = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)

    plt.plot(x, y, '8', xnew, ynew)
    plt.legend(['MarkPoint', 'Spline'])

    plt.gca().invert_yaxis()
    # plt.axis( [ x.min() - 1 , x.max() + 1 , y.min() - 1 , y.max() + 2 ] )
    plt.show()


def show3D():
    point = [[85.0473, 112.7322, 441.2153],
             [97.7168, 110.1424, 440.9912],
             [72.1235, 109.3822, 440.2387],
             [109.4031, 104.2037, 441.7863],
             [60.7102, 102.1624, 439.83],
             [116.6687, 94.8319, 443.4596],
             [52.4, 88.9331, 438.5008],
             [119.0515, 81.0491, 443.171],
             [48.6851, 75.9596, 438.7289],
             [114.5955, 69.8285, 442.1298],
             [49.3119, 63.4969, 438.2798],
             [105.263, 62.1415, 442.2687],
             [50.6698, 50.663, 437.1106]]
    point = np.array(point)
    left_point = point[point[:, 0] < point[0, 0]]
    left_point = left_point[::-1]
    right_point = point[point[:, 0] >= point[0, 0]]
    new_point = np.vstack((left_point, right_point))
    x = new_point[:, 0]
    y = new_point[:, 1]
    z = new_point[:, 2]
    tck, u = splprep([x, y, z], s=0)
    xnew, ynew, znew = splev(np.linspace(0, 1, 1000), tck, der=0)

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    # ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z, marker='8', c='b', s=50)

    ax.plot(xnew, ynew, znew, 'r-')
    ax.set_xlim3d(xmin=0, xmax=120)
    ax.set_ylim3d(ymin=30, ymax=150)
    ax.set_zlim3d(zmin=350, zmax=500)
    plt.show()


def cv_show(img_name, img):
    cv2.namedWindow(img_name, 0)
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getCenterPoint(img):
    """
        分割出检测板图像
        检测7×7圆点阵列获取圆心像素坐标值
    """
    img_copy = img.copy()
    img_canny = cv2.Canny(img, 50, 200)
    cv_show('img_canny', img_canny)
    point = []
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, 1)
        approx = cv2.approxPolyDP(cnt, epsilon, 1)
        array_approx = np.array(approx).reshape(-1, 2)
        if array_approx.shape[0] == 5 and cv2.contourArea(cnt) > 40000:
            # 掩模
            mask = np.zeros(img.shape[:], dtype="uint8")
            mask = cv2.fillPoly(mask, [array_approx], (255, 255, 255))
            # cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 3)
            cv_show('img_copy', img_copy)
            masked = cv2.bitwise_and(img, mask)
            cv_show('masked', masked)
    #             masked = cv2.morphologyEx(masked,cv2.MORPH_OPEN,(5, 5))

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 10000:
            ellipse = cv2.fitEllipse(cnt)  # ellipse = [ (x, y) , (a, b), angle ]
            point.append(ellipse[0])
            cv2.ellipse(img_copy, ellipse, (0, 255, 0), 2)
    cv_show('img_copy', img_copy)
    #     point = np.array(point)
    point = sorted(point, key=lambda point: -point[1])
    point = np.array(point)
    point = point.reshape(7, 7, 2)

    plt.tick_params(labelsize=15)
    x = point[:, :, 0]
    y = point[:, :, 1]
    plt.scatter(x, -y, color='b', marker='o', edgecolors='b', s=100, linewidth=2)

    return point


def qt_tittle():
    from PyQt5 import QtWidgets  # 导入PyQt5部件
    from PyQt5.QtGui import QIcon
    import sys

    app = QtWidgets.QApplication(sys.argv)  # 建立application对象
    first_window = QtWidgets.QWidget()  # 建立窗体对象
    first_window.resize(400, 300)  # 设置窗体大小
    first_window.setWindowTitle("软体机器人测量系统")  # 设置窗体标题
    first_window.setWindowIcon(QIcon('../src_video_and_img/1.png'))
    first_window.show()  # 显示窗体
    sys.exit(app.exec())  # 运行程序


def bright():
    import cv2 as cv
    # 引入numpy模块
    import numpy as np
    # 引入sys模块
    import sys

    # 对比度范围：0 ~ 0.3
    alpha = 0.3
    # 亮度范围0 ~ 100
    beta = 100
    img = cv2.imread(r'..\src_video_and_img\img_left.bmp')
    img2 = cv2.imread(r'..\src_video_and_img\arm.bmp')

    def updateAlpha(x):
        global alpha, img, img2
        alpha = cv.getTrackbarPos('Alpha', 'image')
        alpha = alpha * 0.01
        img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))

    def updateBeta(x):
        global beta, img, img2
        beta = cv.getTrackbarPos('Beta', 'image')
        img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))

    def img_test():
        global beta, img, img2
        # 判断是否读取成功
        if img is None:
            print("Could not read the image,may be path error")
            return

        # 创建窗口
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
        cv.createTrackbar('Beta', 'image', 0, 255, updateBeta)
        cv.setTrackbarPos('Alpha', 'image', 100)
        cv.setTrackbarPos('Beta', 'image', 10)
        while (True):
            cv.imshow('image', img)
            if cv.waitKey(1) == ord('q'):
                break
        cv.destroyAllWindows()

    sys.exit(img_test() or 0)


def armProcess(img):
    """获取人工肌肉机械臂的像素坐标"""
    pinkLower = (156, 43, 40)
    pinkUpper = (180, 255, 255)
    rectangle = np.zeros((1242, 2208, 3), dtype="uint8")
    # cv2.rectangle(rectangle, (552, 400), (1500, 1242), (255, 255, 255), -1)  # 机械臂区域
    cv2.rectangle(rectangle, (1000, 500), (2000, 1000), (255, 255, 255), -1)  # 软体手区域
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


def edge_detection():
    def cv_show(img_name, img):
        # cv2.namedWindow(img_name, 0)
        cv2.imshow(img_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = cv2.imread('../src_video_and_img/0001.jpg')
    img = cv2.resize(img, (600, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show('img', img)

    canny = cv2.Canny(gray, 80, 150)
    cv_show('canny', canny)

    # Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv_show('Roberts', Roberts)

    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv_show('Prewitt', Prewitt)

    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv_show('Sobel', dst)

    Laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(Laplacian)
    cv_show('Laplacian', Laplacian)


if __name__ == "__main__":
    show3D()
    show2D()