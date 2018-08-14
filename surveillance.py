#!/usr/bin/python3
# Python 3.5+
# Surveillance system
# Author: @zichun

'''
Normal Usage(surveillance and face recognition):
python3 surveillance.py

Train a new classifier when face images databse changed:
python3 surveillance.py -t

Collect face images for training and saving:
python3 surveillance.py -c

Show help text:
python3 surveillance.py -h

Press 'q' for quit program
'''

from key_video_writer import KeyVideoWriter
from sql_updater import SqlUpdater
from motion_detector import MotionDetector
from face_detector import FaceDetector
from classifier_train import ClassifierTrain
from face_recognizer import KnnFaceRecognizer
from frame_grabber import FrameGrabber
from imutils.video import FPS
import cv2
import imutils
from subprocess import call
import argparse
import datetime
import time
import os

# True for showing video GUI, change to false on server OS
SHOW_GUI = True

# Set default working directory
HOME_PATH = "/home/zichun/SurveillanceSystem"
os.chdir(HOME_PATH)

# ROI for motion detection and face detection
left_offsetX = 900
right_offsetX = 1550
up_offsetY = 650
down_offsetY = 1200

# set image resize ratio for motion and face detection
motion_resize_ratio = 0.25
faceD_resize_ratio = 0.5

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", help="train a KNN faces classifier",
                action="store_true")
ap.add_argument("-c", "--collect_faces", help="collect faces images",
                action="store_true")
args = ap.parse_args()

if args.train:
    # method: LSVM, KNN, ALL
    classifier_train = ClassifierTrain(method='ALL')
    classifier_train.start()
    exit()
elif args.collect_faces:
    call(["python3", "face_collector.py"])
    exit()

# initialize key video writer and the consecutive number of
# frames that have NOT contained any action
key_video_writer = KeyVideoWriter()
num_consec_frames = 0

# init motion frame counts
num_motion_frames = 0
num_total_frames = 0
start_recording = False

# Initialize SQL Updater
sql_updater = SqlUpdater()
try:
    sql_updater.connect()
    # sql_updater.truncate()
except:
    print("[INFO] Failed to Connect SQL. ")

# Declare user info dictionary
info_dict = {'NAME': '', 'DATETIME': '', 'ACTION': ''}

# Start videostream, 0 for webcam, 1 for rtsp
frame_grabber = FrameGrabber(1)
frame_grabber.start()

# Initialize motion detector
motion_detector = MotionDetector(_resize_ratio=motion_resize_ratio)
num_frame_read = 0  # no. of frames read

# Initialize face detector
face_detector = FaceDetector(_scale=faceD_resize_ratio)
face_detected = False
# Initialize face recognizer
knn_face_recognizer = KnnFaceRecognizer()

# FPS calculation
fps = FPS().start()
current_fps = 0

while True:
    time_start = time.time()

    if not sql_updater.running:
        try:
            sql_updater.connect()
        except:
            pass
        else:
            print("[INFO] Succeed to Connect SQL. ")

    # grab frame
    frame = frame_grabber.read()
    # frame will be used by motion detector, create another show copy
    frame_show = frame.copy()
    # Only interested in this ROI region(door area)
    frame_roi = frame[up_offsetY:down_offsetY, left_offsetX:right_offsetX]
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    motion_locs = motion_detector.update(frame_gray)

    # boolean used to indicate if the consecutive frames
    # counter should be updated
    update_consec_frames = True

    num_total_frames += 1

    face_detected = False

    # form a nice average before motion detection
    if num_frame_read < 15:
        num_frame_read += 1
        continue

    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    ts1 = timestamp.strftime("%Y-%m-%d %H:%M:%S_%f")
 
    if len(motion_locs) > 0:
        num_motion_frames += 1
        # initialize the minimum and maximum (x, y)-coordinates
        (minX, minY) = (999999, 999999)
        (maxX, maxY) = (-999999, -999999)

        # loop over the locations of motion and accumulate the
        # minimum and maximum locations of the bounding boxes
        for l in motion_locs:
            (x, y, w, h) = cv2.boundingRect(l)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))

        known_face_locs = face_detector.detect(frame_roi)

        # reset the number of consecutive frames with NO action to zero
        update_consec_frames = False
        num_consec_frames = 0
        # if we are not already recording, start recording
        if not key_video_writer.recording and start_recording:
            video_save_path = "{}/{}.avi".format("videos", ts)
            key_video_writer.start(
                video_save_path, cv2.VideoWriter_fourcc(*'MJPG'), 15)
        
        if len(known_face_locs) > 0:
            face_detected = True
            image_save_path = "images/" + ts1 + ".jpg"
            cv2.imwrite(image_save_path, frame_roi)

            #print("[INFO] " + str(len(known_face_locs)) + " face found.")
            # Start face recognition
            predictions = knn_face_recognizer.predict(
                x_img=frame_roi, x_known_face_locs=known_face_locs)
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} ".format(name) + ts)
                cv2.rectangle(frame_show, (left+left_offsetX, top+up_offsetY),
                              (right+left_offsetX, bottom+up_offsetY), (0, 255, 0), 2)
                cv2.rectangle(frame_show, (left+left_offsetX, bottom+up_offsetY),
                              (right+left_offsetX, bottom+up_offsetY+15), (0, 255, 0), -1)
                cv2.putText(frame_show, name, (int((right-left)/4)+left+left_offsetX, bottom+up_offsetY+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                info_dict['DATETIME'] = ts
                info_dict['NAME'] = name
                info_dict['ACTION'] = 'NA'

                sql_updater.insert(info_dict)

        # draw red bounding box on moving body
        # because we resize image in motion detector
        cv2.rectangle(frame_show, (int(minX/motion_resize_ratio)+left_offsetX, int(minY/motion_resize_ratio)+up_offsetY),
                      (int(maxX/motion_resize_ratio)+left_offsetX, int(maxY/motion_resize_ratio)+up_offsetY), (0, 0, 255), 3)

    if update_consec_frames:
        num_consec_frames += 1
    
    if current_fps >= 100:
        fpsText = "FPS: " + str(current_fps)
    elif current_fps >= 10:
        fpsText = "FPS: " + " " + str(current_fps)
    else:
        fpsText = "FPS: " + "  " + str(current_fps)

    cv2.putText(frame_show, fpsText, (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    cv2.rectangle(frame_show, (left_offsetX, up_offsetY), (right_offsetX, down_offsetY), (0, 0, 0), 2)
    frame_toVideo = imutils.resize(frame_show, width=1344, height=760)
    # update the key frame video buffer
    key_video_writer.update(frame_toVideo)

    # if we are recording and reached a threshold on consecutive
    # number of frames with no action, stop recording the clip
    if key_video_writer.recording and num_consec_frames == 50:
        key_video_writer.finish()
        start_recording = False
    
    if not key_video_writer.recording and not start_recording and num_motion_frames >= 4:
        motion_ratio = num_motion_frames / num_total_frames
        if motion_ratio > 0.3:
            start_recording = True
    elif face_detected and not start_recording:
        start_recording = True
        
    if num_consec_frames >= 50:
        num_consec_frames = 50
        num_total_frames = 0
        num_motion_frames = 0
        start_recording = False

    if SHOW_GUI:
        frame_show = imutils.resize(frame_show, width=1344, height=760)
        cv2.imshow("Frame", frame_show)

    fps.update()

    time_end = time.time()
    time_each = time_end - time_start
    current_fps = int(1 / time_each)
    # print("{}".format(current_fps))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are in the middle of recording a video, wrap it up
if key_video_writer.recording:
    key_video_writer.finish()

# Clean up and release memory
sql_updater.close()
frame_grabber.stop()
cv2.destroyAllWindows()
