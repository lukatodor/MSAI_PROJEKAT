import cv2 as cv
import numpy as np
import glob
# import matplotlib.pyplot as plt

############################# Loading in image ######################################
img = cv.imread('test_images/solidYellowCurve.jpg') # Load as grayscale
img2 = cv.imread('test_images/test2.jpg') # Load as grayscale

assert img is not None, "file could not be read, check with os.path.exists()"
cv.imshow('Original', img2)

H, W = img.shape[:2]
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
undistortedImg = cv.undistort(img2, mtx, dist, None, newCameraMtx)

# Display the final result
# cv.imshow('Undistorted', undistortedImg)
######################################################################################


################################ BEV-(Birds Eye View) ###############################################
pts1 = np.float32(
        [[0,H],     # bottom left
        [420,340], # top left
        [540,340], # top right
        [W,H]      # bottom right
        ])
pts2 = np.float32([
        [0.10 * W, H],
        [0.10 * W, 0],
        [0.90 * W, 0],
        [0.90 * W, H],
    ])
pts3 = np.float32(
        [[375, 480],     # bottom left
        [905, 480],      # top left
        [1811, 685],     # top right
        [-531, 685]      # bottom right
        ])
pts4 = np.float32([
        [0, 0],
        [500, 0],
        [500, 600],
        [0, 600],
    ])


M = cv.getPerspectiveTransform(pts1,pts2)
N = cv.getPerspectiveTransform(pts3,pts4)

warped = cv.warpPerspective(img,M,(W,H))
warped2 = cv.warpPerspective(img2,N,(W,H))


cv.imshow('Warped', warped2)

######################################################################################
r_thresh=190
r = warped[:, :, 2]
red = (r >= r_thresh).astype(np.uint8)
cv.imshow('White', red*255)

h_range=(15, 40)
s_min=90        
v_min=100
hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
lower = np.array([h_range[0], s_min, v_min], dtype=np.uint8)
upper = np.array([h_range[1], 255, 255], dtype=np.uint8)
yellow = (cv.inRange(hsv, lower, upper) > 0).astype(np.uint8)
cv.imshow('Yellow', yellow*255)

mask = red 
kernel = np.ones((3, 3), np.uint8)
gray =  cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
cv.imshow('GRAY', gray*255)


############################## Fit Viusual 2 ##########################################
def find_lane_pixels(binary, nwindows=9, margin=100, minpix=50):
    h, w = binary.shape
    # Histogram of the bottom half -> peaks are the lane line bases
    histogram = np.sum(binary[h // 2:, :], axis=0)
    midpoint = w // 2
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    window_height = h // nwindows
    nonzeroy, nonzerox = binary.nonzero()
    left_inds, right_inds = [], []

    for i in range(nwindows):
        y_low = h - (i + 1) * window_height
        y_high = h - i * window_height

        in_y = (nonzeroy >= y_low) & (nonzeroy < y_high)
        good_left = (in_y & (nonzerox >= leftx_current - margin) &
                     (nonzerox < leftx_current + margin)).nonzero()[0]
        good_right = (in_y & (nonzerox >= rightx_current - margin) &
                      (nonzerox < rightx_current + margin)).nonzero()[0]
        left_inds.append(good_left)
        right_inds.append(good_right)

        # Recenter next window on the mean x of the pixels found
        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)
    return (nonzerox[left_inds], nonzeroy[left_inds],
            nonzerox[right_inds], nonzeroy[right_inds])


leftx, lefty, rightx, righty = find_lane_pixels(gray)

# Fit x = A*y^2 + B*y + C (x as a function of y, lines are near vertical)
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, H - 1, H)
left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

# Visualization
out_img = np.dstack((gray, gray, gray)) * 255
out_img[lefty, leftx] = [0, 0, 255]
out_img[righty, rightx] = [255, 0, 0]
left_pts = np.int32(np.column_stack((left_fitx, ploty)))
right_pts = np.int32(np.column_stack((right_fitx, ploty)))
cv.polylines(out_img, [left_pts], False, (0, 255, 255), 3)
cv.polylines(out_img, [right_pts], False, (0, 255, 255), 3)
cv.imshow('Lane fit', out_img)

######################################################################################




cv.waitKey(0)

cv.destroyAllWindows()