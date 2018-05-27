# Class FaceCollector

'''
This class is for collecting face images to do training and saving.
'''

from motion_detector import MotionDetector
from face_detector import FaceDetector
from frame_grabber import FrameGrabber
from imutils.video import FPS
import datetime
import cv2

# Start videostream, 0 for webcam, 1 for rtsp
frame_grabber = FrameGrabber(0)
frame_grabber.start()

# Initialize motion detector
motion_detector = MotionDetector()
num_frame_read = 0 # no. of frames read

# Initialize face detector
face_detector = FaceDetector()

# FPS calculation
fps = FPS().start()

print("[INFO] Start collecting face images.")

while True:
    # grab frame
    frame = frame_grabber.read()
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
        face_locs = face_detector.detect(frame, motion_locs)
        if len(face_locs) > 0:
            # print("[INFO] "+str(len(face_locs)) + " face found.")/home/zichun
            # Save image with faces detected
            timestamp = datetime.datetime.now()
            ts = timestamp.strftime("%Y-%m-%d_%H:%M:%S_%f")
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
        (minX, minY) = (999999, 999999)
        (maxX, maxY) = (-999999, -999999)

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
print("[INFO] Collection Done.")
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up and release memory
cv2.destroyAllWindows()
frame_grabber.stop()
