import numpy as np
import cv2
import glob

w=11
h=8

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = [] # 3d point in real world space
img_points_l = [] # 2d points in image plane.
img_points_r = [] # 2d points in image plane.

cal_path = "C:\\Users\\WeiCJ\\Desktop\\vision_measuring_system\\ZED_cam\\calibrate\\pic_calibration\\"


images_right = glob.glob(cal_path + 'right\\*.bmp')
images_left = glob.glob(cal_path + 'left\\*.bmp')
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
    # if ret_l and ret_r:
    obj_points.append(objp)

    if ret_l:
        cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        img_points_l.append(corners_l)
        # Draw and display the corners
        # ret_l = cv2.drawChessboardCorners(img_l, (w, h), corners_l, ret_l)
        # cv2.imshow(images_left[i], img_l)
        # cv2.waitKey(500)

    if ret_r:
        cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        img_points_r.append(corners_r)
        # Draw and display the corners
        # ret_r = cv2.drawChessboardCorners(img_r, (w, h), corners_r, ret_r)
        # cv2.imshow(images_right[i], img_r)
        # cv2.waitKey(500)
img_shape = (1242, 2208)

_, M1, d1, r1, t1 = cv2.calibrateCamera(obj_points, img_points_l, img_shape, None, None)
_, M2, d2, r2, t2 = cv2.calibrateCamera(obj_points, img_points_r, img_shape, None, None)
# camera_model = stereo_calibrate(img_shape)

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
ret, M_1, d_1, M_2, d_2, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r,
                                                      M1, d1, M2, d2, img_shape,
                                                      None, None, None, None,
                                                      criteria=stereo_cal_criteria, flags=0)

print('Intrinsic_mtx_1', M_1)
print('dist_1', d_1)
print('Intrinsic_mtx_2', M_2)
print('dist_2', d_2)
print('R', R)
print('T', T)
print('E', E)
print('F', F)

camera_model = dict([('M1', M_1), ('M2', M_2), ('dist1', d_1), ('dist2', d_2),
                     ('rvecs1', r1), ('rvecs2', r2),
                     ('R', R), ('T', T),('E', E), ('F', F)])

cv2.destroyAllWindows()
print(camera_model)
