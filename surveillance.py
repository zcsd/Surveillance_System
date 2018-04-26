# Office surveillance system
# Usage: Python3 surveillance.py
# Press 'q' for quit

from imutils.video import WebcamVideoStream
from imutils.video import FPS
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
time.sleep(2.0) # for warm up camera
fps = FPS().start()

while True:
	# grab frame
	frame = videoStream.read()
	#frame = imutils.resize(frame, width=320, height=240)

	cv2.imshow("Frame", frame)

	fps.update()

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up and release memory
cv2.destroyAllWindows()
videoStream.stop()