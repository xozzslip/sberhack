import numpy as np
from collections import deque
import cv2 as cv


def main():
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('data/output.avi',fourcc, 20.0, (1280, 720))

    cap = cv.VideoCapture(0)
     # wait for static background
    fback = wait_for_static_background(cap)
    while(1):
        
       
        
       
        _, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if frame is None:
            break
        newframe = cv.absdiff(frame, fback)
        _, newframe = cv.threshold(newframe, 5, 255, cv.THRESH_TOZERO)

        circles = cv.HoughCircles(newframe, cv.HOUGH_GRADIENT, 1, 1000,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=250)

        _, contours, _ = cv.findContours(np.copy(newframe), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        newframe = resize_frame_to_out(newframe)
        newframe = cv.cvtColor(newframe, cv.COLOR_GRAY2BGR)
        if circles is not None:
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

            # find backbone subset
            backbone = []
            main_contour = contours[main_contour_i]
            top_pixel_i = 0
            for i in range(len(main_contour)):
                if main_contour[i][0][1] < main_contour[top_pixel_i][0][1]: # y coord
                    top_pixel_i = i
            
            backbone = main_contour[top_pixel_i:500]
            cv.polylines(newframe, backbone, True, (0, 0, 255), thickness=5)
            # out.write(newframe)
        else:
            cv.putText(newframe, "READY", (400, 400), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 200))
        cv.imshow("frame", newframe)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def wait_for_static_background(cap):
    frames_count = 10
    frames_accamulator = deque(maxlen=frames_count)
    while (1):
        _, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if len(frames_accamulator) == frames_count:
            diff_s = 0
            for prev_frame in frames_accamulator:
                diff = cv.absdiff(frame, prev_frame)
                _, diff = cv.threshold(diff, 35, 255, cv.THRESH_TOZERO)
                diff_s += cv.sumElems(diff)[0]
            print(diff_s)
            if diff_s < 3000:
                print("Static captured")
                return frame
        cv.imshow("frame", frame)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        frames_accamulator.append(frame)


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
    
    return frame


def draw_pixel(x, y):
    cv.circle(newframe,(x, y),2,(0,0,255),3)


if __name__ == '__main__':
    main()
