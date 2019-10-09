import numpy as np
import cv2

import time

snusBoksConstantArea = 38.48  # cm2
snusBoksConstantPerim = 21.99 # cm
pixelPerMetric = 0

cap = cv2.VideoCapture(0)
time.sleep(0.01)

def putTextInFrame(frame, xcord, ycord, variableType:str, variable):
    cv2.putText(frame, variableType + str('%.2f' % round(variable, 2)) + 'cm', (xcord, ycord), color=(255,0,0),
                thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)

while(True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(gray, 100, 180, apertureSize=3)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    cnts_filt = []

    if len(cnts) > 0:
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 1000:
                cnts_filt.append(cnt)
        if len(cnts_filt) > 0:
            pixelPerMetricArea = snusBoksConstantArea / cv2.contourArea(cnts_filt[0])
            pixelPerMetricPerim = snusBoksConstantPerim / cv2.arcLength(cnts_filt[0], True)
            putTextInFrame(frame, 50, 50, 'Omkrets: ', cv2.arcLength(cnts_filt[0], True)*pixelPerMetricPerim)
            putTextInFrame(frame, 50, 100, 'Areal: ', cv2.contourArea(cnts_filt[0])*pixelPerMetricArea)
            if len(cnts_filt) > 1:
                putTextInFrame(frame, 300, 50, 'Omkrets: ', cv2.arcLength(cnts_filt[1], True)*pixelPerMetricPerim)
                putTextInFrame(frame, 300, 100, 'Areal: ', cv2.contourArea(cnts_filt[1])*pixelPerMetricArea)
                cv2.fillPoly(frame, pts=[cnts_filt[1]], color=(0,255,0))
            cv2.fillPoly(frame, pts=[cnts_filt[0]], color=(255,0,0))

            cv2.drawContours(frame, cnts_filt, -1, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()