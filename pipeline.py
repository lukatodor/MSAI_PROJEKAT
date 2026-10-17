import cv2 as cv
import numpy as np
import glob
# import matplotlib.pyplot as plt

############################# Loading in image ######################################
img = cv.imread('test_images/whiteCarLaneSwitch.jpg', 0) # Load as grayscale
assert img is not None, "file could not be read, check with os.path.exists()"
cv.imshow('Original', img)

H, W = img.shape[:2]
######################################################################################


########################### High pass filter ##########################################
gaussian = cv.GaussianBlur(img, (3, 3), 0)

# Make high pass kernel
kernel_3x3 = np.ones((3, 3), np.float32) * -1
kernel_3x3[1, 1] = 9

sharpen = cv.filter2D(gaussian, -1, kernel_3x3)
cv.imshow("3x3 Sharpen", sharpen)
######################################################################################


############################### Distortion #############################################
# Load calibrated camera parameters
calibratio = np.load('img/calib.npz')
mtx = calibratio['mtx']
dist = calibratio['dist']
rvecs = calibratio['rvecs']
tvecs = calibratio['tvecs']

# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (W, H), 1, (W, H))
undistortedImg = cv.undistort(sharpen, mtx, dist, None, newCameraMtx)

# Display the final result
cv.imshow('Undistorted', undistortedImg)
######################################################################################


############################### CANNY ############################################
blur = cv.GaussianBlur(sharpen, (5,5), cv.BORDER_DEFAULT)

ret, thresh = cv.threshold(blur, 125, 225, cv.THRESH_BINARY)
cv.imshow('Thrash', thresh)

canny = cv.Canny(thresh, 25, 175)
cv.imshow('Canny Edged', canny)
######################################################################################


################################ BEV-(Birds Eye View) ###############################################
pts1 = np.float32([[171,520], # bottom left
                   [445,340], # top left
                   [535,340], # top right
                   [934,540]  # bottom right
                   ])
pts2 = np.float32([[200,H],[200,0],[W-200,0],[W-200,H]])

M = cv.getPerspectiveTransform(pts1,pts2)

dst = cv.warpPerspective(thresh,M,(W,H))

cv.imshow('Warped', dst)
######################################################################################



cv.waitKey(0)

cv.destroyAllWindows()