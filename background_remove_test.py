import numpy as np
import cv2 as cv
W_HEIGHT = 200
W_WIDTH = W_HEIGHT * 16 // 9
O_HEIGHT = 720
O_WIDTH = 1280
cap = cv.VideoCapture('data/background.mp4')
_, fback = cap.read()

cap = cv.VideoCapture('data/good1.mp4')
fback = cv.resize(fback, (W_WIDTH, W_HEIGHT), cv.INTER_AREA)
while(1):
    ret, frame = cap.read()
    # cv.resize(frame, frame, cv.Size(640, 360), 0, 0, INTER_CUBIC)
    frame = cv.resize(frame, (W_WIDTH, W_HEIGHT), cv.INTER_AREA)
    frame = cv.absdiff(frame, fback)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    nonzeroes = []
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            breakpoint()
            if frame[i][j] < 50:
                frame[i][j] = 0

    frame = cv.resize(frame, (O_WIDTH, O_HEIGHT), cv.INTER_AREA)
    cv.imshow('frame', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
