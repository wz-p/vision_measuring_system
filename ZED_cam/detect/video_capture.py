import cv2
import numpy as np


# Open the ZED camera
cap = cv2.VideoCapture(1)
if cap.isOpened() == 0:
    exit(-1)

# Set the video resolution to HD2k (4416*1242)
# 1080p	3840x1080
# 720p	2560x720
# WVGA	1344x37611
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output_13.avi',fourcc, 30.0, (4416,1242))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,1)
#        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   将视频转换为灰色的源
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
