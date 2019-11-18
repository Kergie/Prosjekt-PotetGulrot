# import the necessary packages
from collections import deque
import cv2
import imutils
from scipy.spatial import distance as dist


class FindTarget:

    def __init__(self):
        pass

    def getTarget(self, bilde):

        center = None
        d = None
        x = None
        y = None
        radius = None

        #bilde = cv2.cvtColor(bilde, cv2.COLOR_BGR2RGB)
        blueLower = (100, 170, 40)
        blueUper = (140, 255, 240)

        yelowLower = (20, 100, 100)
        yellowHiger = (30, 255, 255)
        frame = bilde

        # Filtering of the pictures
        # 1 We blur the picture with Gaussian blur
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # 2 We change the colors to HSV
        color = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # 3 Removes everyting that is not inside the Color range that is chosen
        mask = cv2.inRange(color, yelowLower, yellowHiger)
        # 4 Dilates the picture

        mask = cv2.dilate(mask, None, iterations=3)
        # 5 Errodes the picture
        mask = cv2.erode(mask, None, iterations=3)
        # 6 Finds countors inne the fillterd picture

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if radius > 15:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), -1)
        if x is not None and y is not None:
            d = dist.euclidean((x, y), (0, 0))

        if len(cnts) > 0 and d is not None and center is not None and radius is not None:
            return d, frame, radius, center

        return None


if __name__ == "__main__":
    pass


