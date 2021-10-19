import cv2
import numpy as np


class Config:
    def __init__(self):
        self.left_camera_matrix = np.array([[897.6963, -3.2254, 943.9867],
                                            [0., 899.8549, 556.6663],
                                            [0., 0., 1.]])
        self.left_distortion = np.array([[0.0884, -0.1136, 0.0022, -0.006, 0.0562]])

        self.right_camera_matrix = np.array([[896.2121, -0.4132, 952.979],
                                             [0., 901.92, 557.278],
                                             [0., 0., 1.]])
        self.right_distortion = np.array([[0.0563, -0.0427, 0.003, -0.0048, 0.02]])

        # om = np.array([0.01911, 0.03125, -0.00960])  # 旋转关系向量
        # R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
        self.R = np.array([[0.9999, -0.0012, 0.0147],
                           [0.0015, 0.9998, -0.0177],
                           [-0.0147, 0.0178, 0.9997]])
        self.T = np.array([[-172.88], [-0.3765], [3.0471]])  # 平移关系向量

        self.size = (1920, 1080)  # 图像尺寸

        # 进行立体更正
        self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, self.validPixROI2 = cv2.stereoRectify(
                                                                self.left_camera_matrix, self.left_distortion,
                                                                self.right_camera_matrix, self.right_distortion,
                                                                self.size, self.R, self.T)
        # 计算更正map
        left_map1, left_map2 = cv2.initUndistortRectifyMap(self.left_camera_matrix, self.left_distortion, self.R1,
                                                           self.P1, self.size, cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(self.right_camera_matrix, self.right_distortion, self.R2,
                                                             self.P2, self.size, cv2.CV_16SC2)

        self.M_left = np.array([[897.6963, -3.2254, 943.9867, 0.],
                                [0., 899.8549, 556.6663, 0.],
                                [0., 0., 1., 0.]])
        self.RT = np.array([[0.9999, -0.0012, 0.0147, -172.88],
                           [0.0015, 0.9998, -0.0177, -0.3765],
                           [-0.0147, 0.0178, 0.9997, 3.0471]])
        self.M_right = np.dot(self.right_camera_matrix, self.RT)

    def get_Ab(self, u_left, v_left, u_right, v_right):
        """

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


def undistort(img, map1, map2):
    undistorted_img = cv2.remap(img, map1, map2)
    return undistorted_img
