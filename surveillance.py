# Office surveillance system
# Usage: Python3 surveillance.py
# Press 'q' for quit

from motiondetector import MotionDetector
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import datetime
import imutils
import time
import cv2

# Camera resolution setting
frameWidth = 320
frameHeight = 240

# Start camera videostream
print("[INFO] starting camera...")
# 0 for default webcam, 1/2/3... for external webcam
videoStream = WebcamVideoStream(src=1)
videoStream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
videoStream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
videoStream.start()
time.sleep(1.0) # for warm up camera, 1 second

# Initialize motion detector
motionDetector = MotionDetector()
noframesRead = 0 # no. of frames read

# FPS calculation
fps = FPS().start()

while True:
	# grab frame
	frame = videoStream.read()
	frameShow = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	locs = motionDetector.update(gray)

	# form a nice average before motion detection
	if noframesRead < 60:
		noframesRead += 1
		continue

	if len(locs) > 0:
		# initialize the minimum and maximum (x, y)-coordinates
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		# loop over the locations of motion and accumulate the
		# minimum and maximum locations of the bounding boxes
		for l in locs:
			(x, y, w, h) = cv2.boundingRect(l)
			(minX, maxX) = (min(minX, x), max(maxX, x + w))
			(minY, maxY) = (min(minY, y), max(maxY, y + h))

		# draw the bounding box
		cv2.rectangle(frameShow, (minX, minY), (maxX, maxY),
			(0, 0, 255), 3)

	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%d %b %Y %H:%M:%S")

	cv2.putText(frameShow, ts, (10, frameShow.shape[0] - 10), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
	cv2.imshow("Frame", frameShow)

	fps.update()

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up and release memory
cv2.destroyAllWindows()
videoStream.stop()