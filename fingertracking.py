import cv2
import numpy as np
import RealSenseClass as Rsc

cam = Rsc.RealCamera()
cam.start_pipe()

class FingerTracker(object):
    def __init__(self):
        super(FingerTracker, self).__init__()

def max_contour(contour_list):
    first_i, second_i = 0, 0
    max_area = 0
    if len(contour_list) != 0:
        for i in range(len(contour_list)):
            cnt = contour_list[i]
            area_cnt = cv2.contourArea(cnt)
            if area_cnt > max_area:
                max_area = area_cnt
                first_i = i
        first_max = contour_list[first_i]
        contour_list = np.delete(contour_list, first_i)

        max_area = 0
        for i in range(len(contour_list)):
            cnt = contour_list[i]
            area_cnt = cv2.contourArea(cnt)
            if area_cnt > max_area:
                max_area = area_cnt
                second_i = i
        second_max = contour_list[second_i]
        return first_max, second_max
    else:
        return None

def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


while True:
    first_cont, second_cont = None, None
    # Take each frame
    cam.get_frame()
    frame = cam.color_image
    _, frame = cam.get_frame()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # define range for green color in HSV
    lower_green = np.array([60, 40, 40])
    upper_green = np.array([90, 250, 250])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        first_cont, second_cont = max_contour(cont)
        if first_cont is not None:
            cx, cy = centroid(first_cont)
            cv2.circle(frame, (cx, cy), 5, [0, 0, 255], -1)
        if second_cont is not None:
            cx, cy = centroid(second_cont)
            cv2.circle(frame, (cx, cy), 5, [0, 255, 0], -1)
    except:
        print('Pas de contours')
        pass
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()