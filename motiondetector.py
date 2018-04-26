# Class MotionDetector

# -------------Temporal Difference Method---------------
# accumulate the weighted average between
# the current frame and the previous frames, then compute
# the pixel-wise differences between the current frame
# and running average, faster algorithm for real time system

import imutils
import cv2

class MotionDetector:
	def __init__(self, accumWeight=0.5, deltaThresh=5, minArea=5000):
		self.isv2 = imutils.is_cv2() # determine the OpenCV version
		self.accumWeight = accumWeight # the frame accumulation weight
		self.deltaThresh = deltaThresh # fixed threshold for the delta image
		self.minArea = minArea # min area for motion detected

		# initialize the average image for motion detection
		self.avg = None

	def update(self, image):
		# initialize the list of locations containing motion
		locs = []

		# if the average image is None, initialize it
		if self.avg is None:
			self.avg = image.astype("float")
			return locs

		cv2.accumulateWeighted(image, self.avg, self.accumWeight)
		frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))

		# threshold the delta image and apply dilations
		thresh = cv2.threshold(frameDelta, self.deltaThresh, 255,
			cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)

		# find contours in the thresholded image
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if self.isv2 else cnts[1]

		# loop over the contours
		for c in cnts:
			# only add the contour to the locs list if it > minArea
			if cv2.contourArea(c) > self.minArea:
				locs.append(c)
		
		return locs