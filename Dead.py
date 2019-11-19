import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from shapely.geometry import LineString
import os


class Dead:

    def getDeadEnds2(self, bilde):
        # img_rgb = cv2.imread(r"C:\Users\marcu\PycharmProjects\Luretriks-Slange\Pictures\DeadEnds\perf2.jpg")
        img_rgb = bilde
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_cannyre = cv2.Canny(img_gray, 50, 120, apertureSize=3)

        path = os.getcwd() + "\\Pictures\\DeadEnds/"
        template = cv2.imread(path + "DeadEnd1.PNG", 0)
        template2 = cv2.imread(path + "DeadEnd2.PNG", 0)
        template3 = cv2.imread(path + "DeadEnd3.PNG", 0)
        template4 = cv2.imread(path + "DeadEnd4.PNG", 0)
        template5 = cv2.imread(path + "DeadEnd5.PNG", 0)
        template6 = cv2.imread(path + "DeadEnd6.PNG", 0)
        template7 = cv2.imread(path + "DeadEnd7.PNG", 0)
        template8 = cv2.imread(path + "DeadEnd8.PNG", 0)
        template9 = cv2.imread(path + "DeadEnd9.PNG", 0)
        template10 = cv2.imread(path + "DeadEnd10.PNG", 0)
        template11 = cv2.imread(path + "DeadEnd11.PNG", 0)
        template12 = cv2.imread(path + "DeadEnd12.PNG", 0)
        template13 = cv2.imread(path + "DeadEnd13.PNG", 0)
        template14 = cv2.imread(path + "DeadEnd14.PNG", 0)
        template15 = cv2.imread(path + "DeadEnd15.PNG", 0)
        template16 = cv2.imread(path + "DeadEnd16.PNG", 0)
        template17 = cv2.imread(path + "DeadEnd17.PNG", 0)
        template18 = cv2.imread(path + "DeadEnd18.PNG", 0)
        template19 = cv2.imread(path + "DeadEnd19.PNG", 0)
        template20 = cv2.imread(path + "DeadEnd20.PNG", 0)
        template21 = cv2.imread(path + "DeadEnd21.PNG", 0)
        template22 = cv2.imread(path + "DeadEnd22.PNG", 0)
        template23 = cv2.imread(path + "DeadEnd23.PNG", 0)
        template24 = cv2.imread(path + "DeadEnd24.PNG", 0)
        template25 = cv2.imread(path + "DeadEnd25.PNG", 0)
        template26 = cv2.imread(path + "DeadEnd26.PNG", 0)
        template27 = cv2.imread(path + "DeadEnd27.PNG", 0)
        template28 = cv2.imread(path + "DeadEnd28.PNG", 0)
        template29 = cv2.imread(path + "DeadEnd29.PNG", 0)
        template30 = cv2.imread(path + "DeadEnd30.PNG", 0)
        template31 = cv2.imread(path + "DeadEnd31.PNG", 0)
        template32 = cv2.imread(path + "DeadEnd32.PNG", 0)
        template33 = cv2.imread(path + "DeadEnd33.PNG", 0)
        template34 = cv2.imread(path + "DeadEnd34.PNG", 0)
        template35 = cv2.imread(path + "DeadEnd35.PNG", 0)
        template36 = cv2.imread(path + "DeadEnd36.PNG", 0)
        template37 = cv2.imread(path + "DeadEnd37.PNG", 0)
        template38 = cv2.imread(path + "DeadEnd38.PNG", 0)
        template39 = cv2.imread(path + "DeadEnd39.PNG", 0)
        template40 = cv2.imread(path + "DeadEnd40.PNG", 0)
        template41 = cv2.imread(path + "DeadEnd41.PNG", 0)
        template42 = cv2.imread(path + "DeadEnd42.PNG", 0)
        template43 = cv2.imread(path + "DeadEnd43.PNG", 0)
        template44 = cv2.imread(path + "DeadEnd44.PNG", 0)
        template45 = cv2.imread(path + "DeadEnd45.PNG", 0)
        template46 = cv2.imread(path + "DeadEnd46.PNG", 0)
        template47 = cv2.imread(path + "DeadEnd47.PNG", 0)
        template48 = cv2.imread(path + "DeadEnd48.PNG", 0)
        template49 = cv2.imread(path + "DeadEnd49.PNG", 0)
        template50 = cv2.imread(path + "DeadEnd50.PNG", 0)
        template51 = cv2.imread(path + "DeadEnd51.PNG", 0)
        template52 = cv2.imread(path + "DeadEnd52.PNG", 0)
        template53 = cv2.imread(path + "DeadEnd53.PNG", 0)
        template54 = cv2.imread(path + "DeadEnd54.PNG", 0)
        template55 = cv2.imread(path + "DeadEnd55.PNG", 0)
        template56 = cv2.imread(path + "DeadEnd56.PNG", 0)
        template57 = cv2.imread(path + "DeadEnd57.PNG", 0)






        """img_canny2 = cv2.Canny(template2, 50, 120, apertureSize=3)
        img_canny3 = cv2.Canny(template3, 50, 120, apertureSize=3)
        img_canny4 = cv2.Canny(template4, 50, 120, apertureSize=3)
        img_canny5 = cv2.Canny(template5, 50, 120, apertureSize=3)
        img_canny6 = cv2.Canny(template6, 50, 120, apertureSize=3)
        img_canny7 = cv2.Canny(template7, 50, 120, apertureSize=3)"""

        templateList = []
        """templateList.append(img_canny)
        templateList.append(img_canny2)
        templateList.append(img_canny3)
        templateList.append(img_canny4)
        templateList.append(img_canny5)
        templateList.append(img_canny6)
        templateList.append(img_canny7)"""

        templateList.append(template)
        templateList.append(template2)
        templateList.append(template3)
        templateList.append(template4)
        templateList.append(template5)
        templateList.append(template6)
        templateList.append(template7)
        templateList.append(template8)
        templateList.append(template9)
        templateList.append(template10)
        templateList.append(template11)
        templateList.append(template12)
        templateList.append(template13)
        templateList.append(template14)
        templateList.append(template15)
        templateList.append(template16)
        templateList.append(template17)
        templateList.append(template18)
        templateList.append(template19)
        templateList.append(template20)
        templateList.append(template21)
        templateList.append(template22)
        templateList.append(template23)
        templateList.append(template24)
        templateList.append(template25)
        templateList.append(template26)
        templateList.append(template27)
        templateList.append(template28)
        templateList.append(template29)
        templateList.append(template30)
        templateList.append(template31)
        templateList.append(template32)
        templateList.append(template33)
        templateList.append(template34)
        templateList.append(template35)
        templateList.append(template36)
        templateList.append(template37)
        templateList.append(template38)
        templateList.append(template39)
        templateList.append(template40)
        templateList.append(template41)
        templateList.append(template42)
        templateList.append(template43)
        templateList.append(template44)
        templateList.append(template45)
        templateList.append(template46)
        templateList.append(template47)
        templateList.append(template48)
        templateList.append(template49)
        templateList.append(template50)
        templateList.append(template51)
        templateList.append(template52)
        templateList.append(template53)
        templateList.append(template54)
        templateList.append(template55)
        templateList.append(template56)
        templateList.append(template57)




        w, h = template.shape[::-1]

        """res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        res1 = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
        res2 = cv2.matchTemplate(img_gray,template3,cv2.TM_CCOEFF_NORMED)
        res3 = cv2.matchTemplate(img_gray,template4,cv2.TM_CCOEFF_NORMED)
        res4 = cv2.matchTemplate(img_gray,template5,cv2.TM_CCOEFF_NORMED)
        res5 = cv2.matchTemplate(img_gray,template6,cv2.TM_CCOEFF_NORMED)"""
        threshold = 0.82

        # loc = np.where( res4 >= threshold)# and np.where(res1 >= threshold) and np.where(res2 >= threshold) and np.where(res3 >= threshold)

        # loop over the scales of the image
        loc = []
        WH = []
        for temp in templateList:

            res = cv2.matchTemplate(img_gray, temp, cv2.TM_CCOEFF_NORMED)
            res = np.where(res >= threshold)
            if res:
                loc.append(res)
                w, h = temp.shape[::-1]
                WH.append([w, h])
            # loc.append(np.where( res >= threshold))

        lastPt = 0
        lastPt2 = 0
        deadEnds = []

        for point, WH in zip(loc, WH):
            w, h = WH
            for pt in zip(*point[::-1]):
                """low = (int(pt[0])-100)
                high = (int(pt[0])+100)
                low2 = (int(pt[1]) - 100)
                high2 = (int(pt[1]) + 100)"""
                deadEnds.append([pt[0] + w / 2, pt[1] + h / 2])
                """
                if low <= lastPt <= high and low2 <= lastPt2 <= high2 :
                    break
                else:
                    
                    lastPt = int(pt[0])
                    lastPt2 = int(pt[1])
                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)"""

        tempDeadEnds = []
        for i in range(len(deadEnds)):
            apApend = True
            if i == len(deadEnds):
                break
            else:

                for j in range(i + 1, len(deadEnds)):
                    if (abs(deadEnds[i][0] - deadEnds[j][0]) <= 100 and abs(deadEnds[i][1] - deadEnds[j][1]) <= 100):
                        apApend = False
                        break
            if apApend:
                tempDeadEnds.append(deadEnds[i])
        deadEnds = tempDeadEnds

        for point in deadEnds:
            cv2.circle(img_rgb, (int(point[0]), int(point[1])),10,(255,0,255), 3)

        cv2.imshow("img", img_rgb)

        cv2.waitKey()
        return deadEnds, img_rgb

if __name__ == "__main__":
    #bilde = c.takePicture()
    bilde = cv2.imread(os.getcwd() + "\\" + "..\\..\\Pictures\\DeadEnds\\perf2.jpg")
    c = Dead()
    c.getDeadEnds2(bilde)
