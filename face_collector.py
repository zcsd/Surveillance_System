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
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Start camera videostream
print("[INFO] starting camera...")
# 0 for default webcam, 1/2/3... for external webcam
video_stream = WebcamVideoStream(src=1)
video_stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
video_stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
video_stream.start()
time.sleep(1.0) # for warm up camera, 1 second

# Initialize motion detector
motion_detector = MotionDetector()
num_frame_read = 0 # no. of frames read

# Initialize face detector
face_detector = FaceDetector()

# FPS calculation
fps = FPS().start()

while True:
    # grab frame
    frame = video_stream.read()
    frame_show = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    motion_locs = motion_detector.update(frame_gray)

    # form a nice average before motion detection
    if num_frame_read < 30:
        num_frame_read += 1
        continue

    if len(motion_locs) > 0:
        # may consider to process in every other frame to accelerate
        # @ZC
        face_locs = face_detector.detect(frame)
        if len(face_locs) > 0:
            # print("[INFO] "+str(len(face_locs)) + " face found.")
            # Save image with faces detected
            timestamp = datetime.datetime.now()
            ts = timestamp.strftime("%Y%m%d%H%M%S_%f")
            image_save_path = "images/" + ts + ".jpg"
            cv2.imwrite(image_save_path, frame)

            for top, right, bottom, left in face_locs:
                # Scale back up face locations
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1
                cv2.rectangle(frame_show,(left, top), (right, bottom), (0, 255, 0), 2)

        # initialize the minimum and maximum (x, y)-coordinates
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # loop over the locations of motion and accumulate the
        # minimum and maximum locations of the bounding boxes
        for l in motion_locs:
            (x, y, w, h) = cv2.boundingRect(l)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))

        # draw the bounding box
        cv2.rectangle(frame_show, (minX, minY), (maxX, maxY),(0, 0, 255), 2)

    cv2.imshow("Frame", frame_show)

    fps.update()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up and release memory
cv2.destroyAllWindows()
video_stream.stop()