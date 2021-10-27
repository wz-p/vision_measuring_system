import cv2
import numpy as np


img=cv2.imread("1-杂物-圆锥体.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(380, 640))
ret,a = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
a = cv2.GaussianBlur(a,(5,5),1)
a = cv2.morphologyEx(a,cv2.MORPH_OPEN,kernel)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

scharrx = cv2.Scharr(a,cv2.CV_64F,1,0)
scharry = cv2.Scharr(a,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

laplacian = cv2.Laplacian(a,cv2.CV_64F, ksize=1)
laplacian = cv2.convertScaleAbs(laplacian)



res = np.hstack((img,a,sobelxy))

# v1=cv2.Canny(img,20,100)
# v2=cv2.Canny(img,80,200)
#
# res = np.hstack((img,v1,v2))
# cv2.namedWindow("Canny_res")
# cv2.resizeWindow("Canny_res", 500, 600)
cv2.imshow('Canny_res', res)
cv2.waitKey(0)
