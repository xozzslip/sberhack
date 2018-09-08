import numpy as np
import cv2 as cv


def main():
    cap = cv.VideoCapture('data/background.mp4')
    _, fback = cap.read()
    fback = prepare_frame(fback)

    cap = cv.VideoCapture('data/good1.mp4')
    while(1):
        _, frame = cap.read()
        if frame is None:
            break
        frame = prepare_frame(frame)
        newframe = cv.absdiff(frame, fback)
        for i in range(len(newframe)):
            for j in range(len(newframe[i])):
                if newframe[i][j] < 50:
                    newframe[i][j] = 0

        newframe = resize_frame_to_out(newframe)
        cv.imshow('frame', newframe)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def resize_frame_to_work(frame):
    W_HEIGHT = 10
    W_WIDTH = W_HEIGHT * 16 // 9
    return cv.resize(frame, (W_WIDTH, W_HEIGHT), cv.INTER_AREA)


def resize_frame_to_out(frame):
    O_HEIGHT = 720
    O_WIDTH = 1280
    return cv.resize(frame, (O_WIDTH, O_HEIGHT), cv.INTER_AREA)


def prepare_frame(frame):
    frame = resize_frame_to_work(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame


if __name__ == '__main__':
    main()
