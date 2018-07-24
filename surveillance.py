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
from knn_classifier_train import KnnClassifierTrain
from knn_face_recognizer import KnnFaceRecognizer
from frame_grabber import FrameGrabber
from imutils.video import FPS
import cv2
import imutils
from subprocess import call
from queue import Queue
import argparse
import datetime

# True for showing video GUI, change to false on server OS
SHOW_GUI = True

# ROI for motion detection and face detection
left_offsetX = 900
right_offsetX = 1600
up_offsetY = 550
down_offsetY = 1350

# Write timelog information to text file if sql connection fail
def backup_to_timelog(q):
    seq_list = []
    # put all information in queue to a list
    for i in range(q.qsize()):
        dict = q.get()
        seq = str(dict['NAME']) + "  " + str(dict['DATETIME']) + "  " + str(dict['ACTION']) + "\n"
        seq_list.append(seq)
    # write list information to txt file
    with open('timelog/backup.txt','a') as f:
        f.writelines(seq_list)

    f.close()
    print("[INFO] Wrote to backup timelog.")

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", help="train a KNN faces classifier",
                action="store_true")
ap.add_argument("-c", "--collect_faces", help="collect faces images",
                action="store_true")
args = ap.parse_args()

if args.train:
    knn_classifier_train = KnnClassifierTrain()
    knn_classifier_train.train()
    exit()
elif args.collect_faces:
    call(["python3", "face_collector.py"])
    exit()

# initialize key video writer and the consecutive number of
# frames that have NOT contained any action
key_video_writer = KeyVideoWriter()
num_consec_frames = 0

# create sql thread
info_queue = Queue(100)  # max size of q is 100
sql_updater = SqlUpdater(info_queue)
sql_updater.start()

# Declare user info dictionary
info_dict = {'NAME': '', 'DATETIME': '', 'ACTION': ''}

# Start videostream, 0 for webcam, 1 for rtsp
frame_grabber = FrameGrabber(1)
frame_grabber.start()

# Initialize motion detector
motion_detector = MotionDetector()
num_frame_read = 0  # no. of frames read

# Initialize face detector
face_detector = FaceDetector()
# Initialize face recognizer
knn_face_recognizer = KnnFaceRecognizer()

# FPS calculation
fps = FPS().start()

while True:
    # grab frame
    frame = frame_grabber.read()
    frame_show = frame.copy()
    frame_roi = frame[up_offsetY:down_offsetY, left_offsetX:right_offsetX]
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    motion_locs = motion_detector.update(frame_gray)

    # boolean used to indicate if the consecutive frames
    # counter should be updated
    update_consec_frames = True

    # form a nice average before motion detection
    if num_frame_read < 15:
        num_frame_read += 1
        continue

    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if len(motion_locs) > 0:
        # initialize the minimum and maximum (x, y)-coordinates
        (minX, minY) = (999999, 999999)
        (maxX, maxY) = (-999999, -999999)

        # loop over the locations of motion and accumulate the
        # minimum and maximum locations of the bounding boxes
        for l in motion_locs:
            (x, y, w, h) = cv2.boundingRect(l)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))

        known_face_locs = face_detector.detect(frame_roi, motion_locs)

        if len(known_face_locs) > 0:
            # reset the number of consecutive frames with NO action to zero
            update_consec_frames = False
            num_consec_frames = 0

            # if we are not already recording, start recording
            if not key_video_writer.recording:
                video_save_path = "{}/{}.avi".format("videos",ts)
                key_video_writer.start(video_save_path, cv2.VideoWriter_fourcc(*'MJPG'), 10)
            #print("[INFO] " + str(len(known_face_locs)) + " face found.")
            # Start face recognition
            predictions = knn_face_recognizer.predict(x_img=frame_roi, x_known_face_locs=known_face_locs)
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                cv2.rectangle(frame_show, (left+left_offsetX, top+up_offsetY), (right+left_offsetX, bottom+up_offsetY), (0, 255, 0), 2)
                cv2.rectangle(frame_show, (left+left_offsetX, bottom+up_offsetY), (right+left_offsetX, bottom+up_offsetY+15), (0, 255, 0), -1)
                cv2.putText(frame_show, name, (int((right-left)/3)+left+left_offsetX,bottom+up_offsetY+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                info_dict['DATETIME'] = ts
                info_dict['NAME'] = name
                info_dict['ACTION'] = 'NA'
                info_queue.put(info_dict)
                # print(info_queue.qsize())
                if info_queue.qsize() >= 100:
                    backup_to_timelog(info_queue)

        # draw red bounding box on moving body
        cv2.rectangle(frame_show, (minX+left_offsetX, minY+up_offsetY), (maxX+left_offsetX, maxY+up_offsetY), (0, 0, 255), 3)

    if update_consec_frames:
        num_consec_frames += 1

    #cv2.putText(frame_show, ts, (10, frame_show.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # update the key frame video buffer
    key_video_writer.update(frame_show)

    # if we are recording and reached a threshold on consecutive
    # number of frames with no action, stop recording the clip

    if key_video_writer.recording and num_consec_frames == 32:
        key_video_writer.finish()

    if num_consec_frames > 32:
        num_consec_frames = 32

    if SHOW_GUI:
        cv2.rectangle(frame_show, (left_offsetX, up_offsetY), (right_offsetX, down_offsetY), (0, 0, 0), 2)
        frame_show = imutils.resize(frame_show, width=1344, height=760)
        cv2.imshow("Frame", frame_show)

    fps.update()

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
sql_updater.join()
frame_grabber.stop()
cv2.destroyAllWindows()
