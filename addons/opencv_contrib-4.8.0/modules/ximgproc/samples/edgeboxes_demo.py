#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sample demonstrates structured edge detection and edgeboxes.
Usage:
  edgeboxes_demo.py [<model>] [<input_image>]
'''

import cv2 as cv
import numpy as np
import sys

if __name__ == '__main__':
    print(__doc__)

    model = sys.argv[1]
    im = cv.imread(sys.argv[2])

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)

    if len(boxes) > 0:
        boxes_scores = zip(boxes, scores)
        for b_s in boxes_scores:
            box = b_s[0]
            x, y, w, h = box
            cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
            score = b_s[1][0]
            cv.putText(im, "{:.2f}".format(score), (x, y), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv.LINE_AA)
            print("Box at (x,y)=({:d},{:d}); score={:f}".format(x, y, score))

    cv.imshow("edges", edges)
    cv.imshow("edgeboxes", im)
    cv.waitKey(0)
    cv.destroyAllWindows()
