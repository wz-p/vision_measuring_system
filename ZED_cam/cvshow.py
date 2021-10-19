import cv2


left_camera = cv2.VideoCapture(1)

while True:
    ret, left_frame = left_camera.read()

    cv2.imshow("left", left_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
