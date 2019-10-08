import numpy as np
import cv2

import time

cap = cv2.VideoCapture(0)
time.sleep(0.01)

while(True):
    ret, frame = cap.read()

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()