import numpy as np
import cv2 as cv
cap = cv.VideoCapture('data/background.mp4')
_, fback = cap.read()
cap = cv.VideoCapture('data/good1.mp4')
while(1):
    ret, frame = cap.read()
    frame = cv.absdiff(frame, fback)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
