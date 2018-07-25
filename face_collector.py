# Class FaceCollector

'''
This class is for collecting face images to do training and saving.
'''

from motion_detector import MotionDetector
from face_detector import FaceDetector
from frame_grabber import FrameGrabber
from imutils.video import FPS
import imutils
import datetime
import cv2

SHOW_GUI = True

left_offsetX = 900
right_offsetX = 1600
up_offsetY = 550
down_offsetY = 1350

# Start videostream, 0 for webcam, 1 for rtsp
frame_grabber = FrameGrabber(1)
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
    frame_roi = frame[up_offsetY:down_offsetY, left_offsetX:right_offsetX]
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    motion_locs = motion_detector.update(frame_gray)

    # form a nice average before motion detection
    if num_frame_read < 15:
        num_frame_read += 1
        continue

    if len(motion_locs) > 0:
        # @ZC may consider to process in every other frame to accelerate
        face_locs = face_detector.detect(frame_roi, motion_locs)
        if len(face_locs) > 0:
            # Save image with faces detected
            timestamp = datetime.datetime.now()
            ts = timestamp.strftime("%Y-%m-%d_%H:%M:%S_%f")
            print("[INFO] "+str(len(face_locs)) + " face found." + ts)
            image_save_path = "images/" + ts + ".jpg"
            cv2.imwrite(image_save_path, frame_roi)

            for top, right, bottom, left in face_locs:
                # Scale back up face locations
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1
                # draw the bounding box for faces
                cv2.rectangle(frame_show,(left+left_offsetX, top+up_offsetY), (right+left_offsetX, bottom+up_offsetY), (0, 255, 0), 2)

        # initialize the minimum and maximum (x, y)-coordinates
        (minX, minY) = (999999, 999999)
        (maxX, maxY) = (-999999, -999999)

        # loop over the locations of motion and accumulate the
        # minimum and maximum locations of the bounding boxes
        for l in motion_locs:
            (x, y, w, h) = cv2.boundingRect(l)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))

        # draw the bounding box for motion
        cv2.rectangle(frame_show, (minX+left_offsetX, minY+up_offsetY), (maxX+left_offsetX, maxY+up_offsetY),(0, 0, 255), 2)

    
    if SHOW_GUI:
        cv2.rectangle(frame_show, (left_offsetX, up_offsetY), (right_offsetX, down_offsetY), (0, 0, 0), 2)
        frame_show = imutils.resize(frame_show, width=1344, height=760)
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
