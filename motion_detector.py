# Class MotionDetector

'''
-------------Temporal Difference algorithm---------------
Accumulate the weighted average between the current frame
and the previous frames, then compute the pixel-wise differences
between the current frame and running average, 
faster algorithm for real time system
'''

import imutils
import cv2

class MotionDetector:
    def __init__(self, _accum_weight=0.5, _delta_thresh=5, _min_area=2000):
        self.isv2 = imutils.is_cv2() # determine the OpenCV version
        self._accum_weight = _accum_weight # the frame accumulation weight
        self._delta_thresh = _delta_thresh # fixed threshold for the delta image
        self._min_area = _min_area # min area for motion detected

        # initialize the average image for motion detection
        self._avg = None

    def update(self, image_gray):
        # initialize the list of locations containing motion
        locs = []

        # if the average image is None, initialize it
        if self._avg is None:
            self._avg = image_gray.astype("float")
            return locs

        cv2.accumulateWeighted(image_gray, self._avg, self._accum_weight)
        frame_delta = cv2.absdiff(image_gray, cv2.convertScaleAbs(self._avg))

        # threshold the delta image and apply dilations
        image_thresh = cv2.threshold(frame_delta, self._delta_thresh, 255, cv2.THRESH_BINARY)[1]
        image_thresh = cv2.dilate(image_thresh, None, iterations=2)
        # cv2.imshow("Thresh", image_thresh)

        # find contours in the thresholded image
        cnts = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if self.isv2 else cnts[1]

        # loop over the contours
        for c in cnts:
            # only add the contour to the locs list if it > _min_area
            if cv2.contourArea(c) > self._min_area:
                locs.append(c)

        return locs