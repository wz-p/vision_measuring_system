import numpy as np
import cv2


"""
预标定数据：
fx  1061.3500   1058.9000           Tx  119.4570
fy  1060.2500   1057.1801           Ty  -0.3341
cx  1103.0200   1097.3900           Tz  0.2413
cy  605.5170    607.2450            Rx  0.0021
k1  -0.0421     -0.0431             Ry  0.0025
k2  0.0118      0.0133              Rz  0.0028
p1  -0.0004     -0.0004
p2  0           0
k3  -0.0057     -0.0057
"""

"""matlab标定数据
def __init__(self):
    self.left_camera_matrix = np.array([[1061.83297, 0., 1098.43688],   # 左相机内参
                                        [0., 1061.32983, 606.02156],
                                        [0., 0., 1.]])
    self.left_distortion = np.array([-0.0421, 0.0118, -0.0004, 0, -0.0057])    # 左相机畸变系数k1, k2, p1, p2, k3=[-0.0577]
    self.right_camera_matrix = np.array([[1058.80814, 0., 1098.79960],        # 右相机内参
                                         [0., 1057.78468, 608.69985],
                                         [0., 0., 1.]])
    self.right_distortion = np.array([-0.0431, 0.0133, -0.0004, 0, -0.0057])  # 右相机畸变系数k1, k2, p1, p2, k3=[-0.0686]

    self.essential_matrix = np.array([[-0.0004, 0.1757, -0.3630],       # 本征矩阵
                                      [0.1335, 0.3338, 120.1385],
                                      [0.0046, -120.1390, 0.3332]])
    self.fundamental_matrix = np.array([[0., 0., -0.0004],          # 基础矩阵
                                      [0., 0., 0.1133],
                                      [-0.0001, -0.1135, 0.4642]])
"""
# left_distortion = np.array([-0.1051, 0.1190, -0.0001811, -0.00090466, -0.0577])
# right_distortion = np.array([-0.1085, 0.1307, 0.00019474, -0.00039216, -0.0686])


class ZED_config(object):
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

        om = np.array([0.0021, 0.0025, 0.0028])  # 旋转关系向量
        self.R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
        self.T = np.array([[119.4570], [-0.3341], [0.2413]])


        # self.R = np.array([[1., 0.0030, 0.0026],
        #                    [-0.003, 1., 0.0028],
        #                    [-0.0026, -0.0028, 1.]])
        # self.T = np.array([[-120.1389], [-0.3625], [-0.1767]])  # 平移关系向量,第一个为基线长度


        self.size = (2208, 1242)  # 图像尺寸

        # 进行立体更正

        # cameraMatrix1-第一个摄像机的摄像机矩阵
        # distCoeffs1-第一个摄像机的畸变向量
        # cameraMatrix2-第二个摄像机的摄像机矩阵
        # distCoeffs1-第二个摄像机的畸变向量
        # imageSize-图像大小
        # R- stereoCalibrate() 求得的R矩阵
        # T- stereoCalibrate() 求得的T矩阵
        # R1-输出矩阵，第一个摄像机的校正变换矩阵（旋转变换）
        # R2-输出矩阵，第二个摄像机的校正变换矩阵（旋转矩阵）
        # P1-输出矩阵，第一个摄像机在新坐标系下的投影矩阵
        # P2-输出矩阵，第二个摄像机在想坐标系下的投影矩阵
        # Q-4*4的深度差异映射矩阵
        # flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，
        # 该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大
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
        undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img


if __name__ == '__main__':
    zed = ZED_config()
    # 读取标定板图片
    for i in range(0, 36):
        t = str(i)
        image_right = cv2.imread(r'.\pic_calibration\right\right_' + t + '.bmp')  # 右视图
        image_left = cv2.imread(r'.\pic_calibration\left\left_' + t + '.bmp')  # 左视图
        left_rectified = zed.undistort(image_left, zed.left_map1, zed.left_map2)
        right_rectified = zed.undistort(image_right, zed.right_map1, zed.right_map2)


    # 读取圆点图片
    # for i in range(22):
    #     image_left = cv2.imread(r'E:\SynologyDrive\PycharmProject\vision_measuring_system\ZED_cam\detect\picture_circle\left_0\left_' + str(i) + '.bmp')
    #     image_right = cv2.imread(r'E:\SynologyDrive\PycharmProject\vision_measuring_system\ZED_cam\detect\picture_circle\right_0\right_' + str(i) + '.bmp')
    #     left_rectified = zed.undistort(image_left, zed.left_map1, zed.left_map2)
    #     right_rectified = zed.undistort(image_right, zed.right_map1, zed.right_map2)


        # a = np.hstack((image_left, image_right))
        a = np.hstack((left_rectified, right_rectified))
        y = np.arange(0, 1200, 40)
        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 8
        for i in range(len(y)):
            cv2.line(a, (0, y[i]), (4416, y[i]), point_color, thickness)

        cv2.namedWindow('a', 0)
        # cv2.resize(a, ())
        cv2.imshow('a', a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()