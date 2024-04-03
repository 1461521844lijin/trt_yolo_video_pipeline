import numpy as np
import cv2 as cv
import sys

if len(sys.argv) != 2:
    print('Input video name is missing')
    exit()

cv.namedWindow("tracking")
camera = cv.VideoCapture(sys.argv[1])
ok, image=camera.read()
if not ok:
    print('Failed to read video')
    exit()
bbox = cv.selectROI("tracking", image)
tracker = cv.TrackerMIL_create()
init_once = False

while camera.isOpened():
    ok, image=camera.read()
    if not ok:
        print('no image to read')
        break

    if not init_once:
        ok = tracker.init(image, bbox)
        init_once = True

    ok, newbox = tracker.update(image)
    print(ok, newbox)

    if ok:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(image, p1, p2, (200,0,0))

    cv.imshow("tracking", image)
    k = cv.waitKey(1) & 0xff
    if k == 27 : break # esc pressed
