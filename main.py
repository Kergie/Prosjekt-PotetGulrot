import numpy as np
import cv2

import time


cap = cv2.VideoCapture(0)
time.sleep(0.01)

while(True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    cnts_filt = []

    if len(cnts) > 0:
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 100:
                cnts_filt.append(cnt)
        if len(cnts_filt) > 0:
            cv2.putText(frame, 'Omkrets: ' + str(cv2.arcLength(cnts_filt[0], True)), (50,50), color=(255,0,0), thickness=3,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
            cv2.putText(frame, 'Areal: ' + str(cv2.contourArea(cnts_filt[0])), (50, 100), color=(255, 0, 0),
                        thickness=3,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
            cv2.fillPoly(frame, pts=[cnts_filt[0]], color=(255,0,0))
            cv2.drawContours(frame, cnts_filt, -1, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()