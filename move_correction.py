import numpy as np
import cv2 as cv


def main():
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('data/output.avia',fourcc, 20.0, (1280, 720))
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
        _, newframe = cv.threshold(newframe, 35, 255, cv.THRESH_TOZERO)

        circles = cv.HoughCircles(newframe, cv.HOUGH_GRADIENT, 1, 1000,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=250)
        assert circles is not None

        _, contours, _ = cv.findContours(np.copy(newframe), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        newframe = resize_frame_to_out(newframe)
        newframe = cv.cvtColor(newframe, cv.COLOR_GRAY2BGR)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(newframe,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(newframe,(i[0],i[1]),2,(0,0,255),3)

        # find largest contour
        main_contour_i = 0
        for i in range(len(contours)):
            if len(contours[main_contour_i]) < len(contours[i]):
                main_contour_i = i
        cv.drawContours(newframe, contours, main_contour_i, (255, 0, 0))

        # find backbone subset
        backbone = []
        main_contour = contours[main_contour_i]
        for i in range(len(main_contour)):
            pixel = main_contour[i][0]
        out.write(newframe)

    cap.release()
    cv.destroyAllWindows()


def resize_frame_to_work(frame):
    W_HEIGHT = 720
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
