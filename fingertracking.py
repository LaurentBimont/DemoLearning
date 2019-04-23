import cv2
import numpy as np
from RealSenseClass import RealCamera
import time

class FingerTracker(RealCamera):
    def __init__(self, reset_period = 5):
        super(FingerTracker, self).__init__()
        self.start_pipe()
        self.min_dist = float('Inf')
        self.center_x, self.center_y = 0, 0
        self.reset_period = reset_period
        self.Time = time.time()


    def max_indice(self, contour_list):
        indice = None
        max_area = 0
        for i in range(len(contour_list)):
            cnt = contour_list[i]
            area_cnt = cv2.contourArea(cnt)
            if area_cnt > max_area and area_cnt > 200:
                max_area = area_cnt
                indice = i
        return indice

    def max_contour(self, contour_list):
        first_i, second_i = None, None
        max_area = 0
        if len(contour_list) != 0:
            # Get the first maximum contour
            indice = self.max_indice(contour_list)
            if indice is not None:
                first_max = contour_list[indice]
            else:
                return None, None

            # Delete the max element of the list
            contour_list = np.delete(contour_list, indice)

            # Get the second maximum contour
            indice = self.max_indice(contour_list)
            if indice is not None:
                second_max = contour_list[indice]
            else:
                second_max = None
            return first_max, second_max

        return None, None

    def centroid(self, max_contour):
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None

    def get_dist(self):
        dist = np.linalg.norm(np.array([self.cx1, self.cy1])
                                 -np.array([self.cx2, self.cy2]))
        return(dist)

    def main(self):
        if time.time() - self.Time > self.reset_period:
            self.min_dist = float('Inf')
            self.Time = time.time()

        first_cont, second_cont = None, None
        # Take each frame
        frame = self.color_image
        _, frame = self.get_frame()
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

        first_cont, second_cont = self.max_contour(cont)

        if (first_cont is not None) and (second_cont is not None):
            self.cx1, self.cy1 = self.centroid(first_cont)
            cv2.circle(frame, (self.cx1, self.cy1), 5, [0, 0, 255], -1)
            self.cx2, self.cy2 = self.centroid(second_cont)
            cv2.circle(frame, (self.cx2, self.cy2), 5, [0, 255, 0], -1)

            # Calculate the center of the two closest point
            dist = self.get_dist()
            if (self.min_dist > dist) and (dist > 20):
                self.min_dist = dist
                self.center_x = (self.cx1 + self.cx2) // 2
                self.center_y = (self.cy1 + self.cy2) // 2

        cv2.circle(frame, (self.center_x, self.center_y), 5, [255, 0, 0], -1)
        return frame


if __name__ == '__main__':
    ft = FingerTracker()
    while True:
        frame = ft.main()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
