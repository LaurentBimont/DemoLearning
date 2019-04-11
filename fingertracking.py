import cv2
import numpy as np
import RealSenseClass as Rsc

cam = Rsc.RealCamera()
cam.start_pipe()


def max_contour(contour_list):
    max_i = 0
    max_area = 0
    for i in range(len(contour_list)):
        cnt = contour_list[i]
        area_cnt = cv2.contourArea(cnt)
        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i
    return contour_list[max_i]


while True:
    # Take each frame
    cam.get_frame()
    frame = cam.color_image
    _, frame = cam.get_frame()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # define range for green color in HSV
    lower_green = np.array([80, 60, 60])
    upper_green = np.array([90, 255, 255])

    # define range of skin color in HSV
    lower_skin = np.array([110, 25, 25])
    upper_skin = np.array([130, 200, 200])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_cont = max_contour(cont)

    print(len(cont))

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

## TO UTILIZE ##
contour_list = contours(hist_mask_image)
max_cont = max_contour(contour_list)

cnt_centroid = centroid(max_cont)
cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)
