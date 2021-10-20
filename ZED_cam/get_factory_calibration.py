import pyzed.sl as sl

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K
init_params.camera_fps = 15

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

calibration_params = zed.get_camera_information().calibration_parameters
# Focal length of the left eye in pixels
focal_left_x = calibration_params.left_cam.fx
# First radial distortion coefficient
k1 = calibration_params.left_cam.disto[0]
# Translation between left and right eye on z-axis
# tz = calibration_params.T.z
# # Horizontal field of view of the left eye in degrees
h_fov = calibration_params.left_cam.h_fov

