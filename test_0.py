import numpy as np
import cv2
import glob

w=11
h=8
class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((w*h, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.obj_points = [] # 3d point in real world space
        self.img_points_l = [] # 2d points in image plane.
        self.img_points_r = [] # 2d points in image plane.

        self.cal_path = filepath

    def read_images(self):
        images_right = glob.glob(self.cal_path + 'right\\*.bmp')
        images_left = glob.glob(self.cal_path + 'left\\*.bmp')
        images_left.sort()
        images_right.sort()

        for i, frame in enumerate(images_left):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (w, h), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (w, h), None)

            # If found, add object points, image points (after refining them)
            if ret_l and ret_r:
                self.obj_points.append(self.objp)

            if ret_l:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                self.img_points_l.append(corners_l)
                # Draw and display the corners
                # ret_l = cv2.drawChessboardCorners(img_l, (w, h), corners_l, ret_l)
                # cv2.imshow(images_left[i], img_l)
                # cv2.waitKey(500)

            if ret_r:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                self.img_points_r.append(corners_r)
                # Draw and display the corners
                # ret_r = cv2.drawChessboardCorners(img_r, (w, h), corners_r, ret_r)
                # cv2.imshow(images_right[i], img_r)
                # cv2.waitKey(500)
            self.img_shape = gray_l.shape[::-1]

            rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.obj_points, self.img_points_l, self.img_shape, None, None)
            rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.obj_points, self.img_points_r, self.img_shape, None, None)
            # self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self):
        flags = 0
        # flags = cv2.CALIB_FIX_INTRINSIC
        # flags = cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags = cv2.CALIB_USE_INTRINSIC_GUESS
        # flags = cv2.CALIB_FIX_FOCAL_LENGTH
        # flags = cv2.CALIB_FIX_ASPECT_RATIO
        # flags = cv2.CALIB_ZERO_TANGENT_DIST
        # flags = cv2.CALIB_RATIONAL_MODEL
        # flags = cv2.CALIB_SAME_FOCAL_LENGTH
        # flags = cv2.CALIB_FIX_K3
        # flags = cv2.CALIB_FIX_K4
        # flags = cv2.CALIB_FIX_K5

        stereo_cal_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M_1, d_1, M_2, d_2, R, T, E, F = cv2.stereoCalibrate(self.obj_points, self.img_points_l, self.img_points_r,
                                                              self.M1, self.d1, self.M2, self.d2, self.img_shape,
                                                              None, None, None, None,
                                                              criteria=stereo_cal_criteria, flags=flags)

        print('Intrinsic_mtx_1', M_1)
        print('dist_1', d_1)
        print('Intrinsic_mtx_2', M_2)
        print('dist_2', d_2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        camera_model = dict([('M1', M_1), ('M2', M_2), ('dist1', d_1), ('dist2', d_2),
                             ('rvecs1', self.r1), ('rvecs2', self.r2),
                             ('R', R), ('T', T),('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

cal = StereoCalibration(".\\pic_calibration\\")
cal.read_images()
cal.stereo_calibrate()
# cal.camera_model