import collections
import os

import cv2 as cv
import numpy as np

INPUT_PATH = 'test_videos/challenge03.mp4'
OUTPUT_DIR = 'docs'

LANE_WIDTH_M = 3.7
LANE_LENGTH_M = 30.0
MIN_LINE_PIXELS = 100
SMOOTH_FRAMES = 5


def perspective_matrices(W, H):
    # Points are tuned per camera; the two sources differ in resolution
    if W >= 1280:
        src = np.float32([
            [240, 700],   # bottom left
            [605, 445],   # top left
            [680, 445],   # top right
            [1100, 685],  # bottom right
        ])
        left, right = 0.25, 0.75
    else:
        src = np.float32([
            [130, H],     # bottom left
            [438, 335],   # top left
            [530, 335],   # top right
            [900, H],     # bottom right
        ])
        left, right = 0.20, 0.80

    dst = np.float32([
        [left * W, H],
        [left * W, 0],
        [right * W, 0],
        [right * W, H],
    ])
    return cv.getPerspectiveTransform(src, dst), cv.getPerspectiveTransform(dst, src)


def warp(img, M):
    H, W = img.shape[:2]
    return cv.warpPerspective(img, M, (W, H))


def to_binary(warped, r_thresh=190, h_range=(15, 40), s_min=70, v_min=100):
    white = (warped[:, :, 2] >= r_thresh).astype(np.uint8)

    hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
    lower = np.array([h_range[0], s_min, v_min], dtype=np.uint8)
    upper = np.array([h_range[1], 255, 255], dtype=np.uint8)
    yellow = (cv.inRange(hsv, lower, upper) > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    return cv.morphologyEx(white | yellow, cv.MORPH_OPEN, kernel)


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


def fit_lines(binary):
    leftx, lefty, rightx, righty = find_lane_pixels(binary)
    if len(leftx) < MIN_LINE_PIXELS or len(rightx) < MIN_LINE_PIXELS:
        return None
    # Fit x = A*y^2 + B*y + C (x as a function of y, lines are near vertical)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, (leftx, lefty, rightx, righty)


def draw_fit(binary, left_fit, right_fit, pixels):
    leftx, lefty, rightx, righty = pixels
    ploty = np.arange(binary.shape[0])

    out_img = np.dstack((binary, binary, binary)) * 255
    out_img[lefty, leftx] = [0, 0, 255]
    out_img[righty, rightx] = [255, 0, 0]
    for fit in (left_fit, right_fit):
        pts = np.int32(np.column_stack((np.polyval(fit, ploty), ploty)))
        cv.polylines(out_img, [pts], False, (0, 255, 255), 3)
    return out_img


def curvature_and_offset(left_fit, right_fit, W, H):
    y_eval = H - 1  # bottom of the image = closest to the car
    left_x_bottom = np.polyval(left_fit, y_eval)
    right_x_bottom = np.polyval(right_fit, y_eval)

    xm_per_pix = LANE_WIDTH_M / (right_x_bottom - left_x_bottom)
    ym_per_pix = LANE_LENGTH_M / H
    y_m = y_eval * ym_per_pix

    # Scale the pixel fit to meters, then R = (1 + (2Ay + B)^2)^1.5 / |2A|
    radii = []
    for A, B, _ in (left_fit, right_fit):
        A_m = A * xm_per_pix / ym_per_pix**2
        B_m = B * xm_per_pix / ym_per_pix
        radii.append((1 + (2 * A_m * y_m + B_m)**2)**1.5 / max(abs(2 * A_m), 1e-6))
    curvature = np.mean(radii)

    # Camera is mounted at the car's center -> car is at the image center column
    lane_center = (left_x_bottom + right_x_bottom) / 2
    offset_m = (W / 2 - lane_center) * xm_per_pix  # >0: car is right of lane center
    return curvature, offset_m


def draw_lane(img, left_fit, right_fit, Minv, curvature, offset_m):
    H, W = img.shape[:2]
    ploty = np.arange(H)
    pts_left = np.column_stack((np.polyval(left_fit, ploty), ploty))
    pts_right = np.flipud(np.column_stack((np.polyval(right_fit, ploty), ploty)))

    # Lane polygon in bird's-eye view: left curve down, right curve back up
    lane_layer = np.zeros_like(img)
    cv.fillPoly(lane_layer, [np.int32(np.vstack((pts_left, pts_right)))], (0, 255, 0))
    cv.polylines(lane_layer, [np.int32(pts_left)], False, (0, 0, 255), 15)
    cv.polylines(lane_layer, [np.int32(pts_right)], False, (255, 0, 0), 15)

    # Inverse perspective: bird's-eye -> original camera view
    lane_unwarped = cv.warpPerspective(lane_layer, Minv, (W, H))
    result = cv.addWeighted(img, 1, lane_unwarped, 0.4, 0)

    side = 'right' if offset_m > 0 else 'left'
    texts = [f'Curvature: {curvature:.0f} m',
             f'Vehicle is {abs(offset_m):.2f} m {side} of center']
    for i, text in enumerate(texts):
        cv.putText(result, text, (30, 50 + 50 * i), cv.FONT_HERSHEY_SIMPLEX,
                   1.2, (255, 255, 255), 2, cv.LINE_AA)
    return result


def process_image(img, M, Minv, show=False):
    H, W = img.shape[:2]
    warped = warp(img, M)
    binary = to_binary(warped)

    fits = fit_lines(binary)
    if fits is None:
        return img
    left_fit, right_fit, pixels = fits
    curvature, offset_m = curvature_and_offset(left_fit, right_fit, W, H)
    result = draw_lane(img, left_fit, right_fit, Minv, curvature, offset_m)

    if show:
        cv.imshow('Warped', warped)
        cv.imshow('Binary', binary * 255)
        cv.imshow('Lane fit', draw_fit(binary, left_fit, right_fit, pixels))
        print(f'Curvature: {curvature:.0f} m')
        print(f'Vehicle is {abs(offset_m):.2f} m {"right" if offset_m > 0 else "left"} of center')
    return result


def process_video(path_in, path_out):
    cap = cv.VideoCapture(path_in)
    assert cap.isOpened(), f'could not open {path_in}'

    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25

    out_dir = os.path.dirname(path_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = cv.VideoWriter(path_out, cv.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    M, Minv = perspective_matrices(W, H)
    history = collections.deque(maxlen=SMOOTH_FRAMES)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fits = fit_lines(to_binary(warp(frame, M)))
        # Keep the last good fits so a frame with a lost line reuses them
        if fits is not None:
            history.append(fits[:2])

        result = frame
        if history:
            left_fit = np.mean([f[0] for f in history], axis=0)
            right_fit = np.mean([f[1] for f in history], axis=0)
            curvature, offset_m = curvature_and_offset(left_fit, right_fit, W, H)
            result = draw_lane(frame, left_fit, right_fit, Minv, curvature, offset_m)

        writer.write(result)
        cv.imshow('Result', result)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()


def main():
    if INPUT_PATH.lower().endswith(('.mp4')):
        path_out = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_PATH))
        process_video(INPUT_PATH, path_out)
        return

    img = cv.imread(INPUT_PATH)
    assert img is not None, 'file could not be read, check with os.path.exists()'
    H, W = img.shape[:2]
    M, Minv = perspective_matrices(W, H)

    result = process_image(img, M, Minv, show=True)
    cv.imshow('Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()