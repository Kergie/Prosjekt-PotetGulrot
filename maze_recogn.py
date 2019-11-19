import os
import numpy as np
import cv2

realtime = False
stillPic = False
testing = False



plotted = False


class mazeRecognizer:
    """
    Reads a picture, filters it, gets the lines in the picture from the \n
    HoughLinesP-function, then returns a list of the lines (in coordinates, \n
    (x1y1, x2y2) and a picture with the line drawn on it.
    """
    def __init__(self):
        #self.cam = Camera()
        pass

    def filtering(self, picture):
        """
        Filters a given picture. Changes to grayscale, \n
        then blurs, edge detects, dilates/erodes, uses \n
        morphological closing then returns the picture.

        :param picture: Picture to filter \n
        :return: The picture filtered \n
        """
        grayfilt = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        grayfilt = cv2.GaussianBlur(grayfilt, (9, 9), 0)

        edges = cv2.Canny(grayfilt, 50, 120, apertureSize=3)
        edges = cv2.dilate(edges, None, iterations=3)
        edges = cv2.erode(edges, None, iterations=2)
        kernel = np.ones((7, 7), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    def findMaze(self, picture=None):
        """
        At the moment; reads a picture from the project, \n
        sends it to the filtering-function, runs HoughLinesP \n
        on the picture, then draws these lines on the picture. \n

        Then returns the list of coordinates (x1y1, x2y2) for the \n
        lines, as well as the picture with the lines drawn.
        :return: List of lines (in x1y1, x2y2 coordinates) and picture
        """
        if testing:
            pic2 = cv2.imread(
                os.getcwd() + "\\" + "..\\..\\Pictures\\DeadEnds\\perf2.jpg", -1)
        elif picture is not None:
            # pic2 = self.cam.takePicture()
            pic2 = picture

        edges2 = self.filtering(pic2)

        lines2 = cv2.HoughLinesP(edges2, 1, np.pi / 1000, 50, maxLineGap=90, minLineLength=80)

        for data in lines2:
            x1 = data[0][0]
            y1 = data[0][1]
            x2 = data[0][2]
            y2 = data[0][3]

            cv2.line(pic2, (x1, y1), (x2, y2), (0, 255, 0), 2)



        return lines2, cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    m = mazeRecognizer()
    _, pic = m.findMaze()
    cv2.imshow("Test", pic)

    cv2.waitKey()
    cv2.destroyAllWindows()
