# Class FaceCollector

'''
This class is for collecting face images to do training and saving.
'''

from motion_detector import MotionDetector
from face_detector import FaceDetector
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import datetime
import imutils
import time
import cv2

# Camera resolution setting
frameWidth = 640
frameHeight = 480

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

# Initialize face detector
faceDetector = FaceDetector()

# FPS calculation
fps = FPS().start()

while True:
	# grab frame
	frame = videoStream.read()
	frameShow = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	motionLocs = motionDetector.update(gray)

	# form a nice average before motion detection
	if noframesRead < 30:
		noframesRead += 1
		continue

	if len(motionLocs) > 0:
		# may consider to process in every other frame to accelerate
		# !!!!!!!!!!!!!!!!!!!!!
		faceLocs = faceDetector.detect(frame)
		if len(faceLocs) > 0:
			print("[INFO] "+str(len(faceLocs)) + " face found.")
			# Save image with faces detected
			timestamp = datetime.datetime.now()
			ts = timestamp.strftime("%Y%b%d%H%M%S_%f")
			imagePath = "images/" + ts + ".jpg"
			cv2.imwrite(imagePath, frame)

			for top, right, bottom, left in faceLocs:
				# Scale back up face locations
				top *= 1
				right *= 1
				bottom *= 1
				left *= 1
				cv2.rectangle(frameShow,(left, top), (right, bottom), (0, 255, 0), 2)

		# initialize the minimum and maximum (x, y)-coordinates
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		# loop over the locations of motion and accumulate the
		# minimum and maximum locations of the bounding boxes
		for l in motionLocs:
			(x, y, w, h) = cv2.boundingRect(l)
			(minX, maxX) = (min(minX, x), max(maxX, x + w))
			(minY, maxY) = (min(minY, y), max(maxY, y + h))

		# draw the bounding box
		cv2.rectangle(frameShow, (minX, minY), (maxX, maxY),
			(0, 0, 255), 3)

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