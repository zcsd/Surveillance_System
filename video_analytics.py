#!/usr/bin/python3
# Python 3.5+
# Video analytics system with face detection and recognition
# Author: @zichun

'''
It's used to do face detection and recognition, update info to SQL database.
Usage: python3 video_analytics.py (make sure surveillance.py is running too)
'''

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.sql_updater import SqlUpdater
from queue import Queue
import os
import collections
import pyinotify  # Used to watch file changes
import imutils
import cv2


# True for showing video GUI, change to false on server OS
SHOW_GUI = True
# Set default working directory
HOME_PATH = "/home/zichun/SurveillanceSystem"
os.chdir(HOME_PATH)

# ROI for motion detection
left_offsetX = 850
right_offsetX = 1650
up_offsetY = 550
down_offsetY = 1250

# set image resize ratio for face detection, reduce calculation
faceD_resize_ratio = 0.5

# Initialize face detector
face_detector = FaceDetector(_scale=faceD_resize_ratio)

# Initialize face recognizer, method:SVM(16.0) or KNN(0.50)
face_recognizer = FaceRecognizer(method='KNN', threshold=0.50)

# Initialize SQL Updater
sql_updater = SqlUpdater()
try:
    sql_updater.connect()
    # Delete all data in SQL database
    # sql_updater.truncate()
except:
    print("[INFO] Failed to Connect SQL. ")

# Declare info dictionary
info_dict = {'NAME': '', 'TIMESTAMP': '', 'VIDEO_PATH': ''}

# FIFO queue, used to store original video path
q_path = Queue()
# FIFO queue, used to indicate whether video file finish writing
q_flag = Queue()

# Watch Manager using pyinotify library
wm = pyinotify.WatchManager()

class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        q_path.put(event.pathname)

    def process_IN_CLOSE_WRITE(self, event):
        q_flag.put(1)  # Indicate whether finish writing

# Note: notifier is threaded safe
notifier = pyinotify.ThreadedNotifier(wm, EventHandler())
mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_CREATE
wdd = wm.add_watch('/home/zichun/SurveillanceSystem/videos_temp', mask)
notifier.start()


def process(file_path):
    print("[INFO] Start to process {}".format(file_path))
    file_time = file_path.replace(
        '/home/zichun/SurveillanceSystem/videos_temp/', '').replace('.avi', '')

    # Save another processed video file
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_save_path = "{}/{}.avi".format("videos", file_time)
    out = cv2.VideoWriter(video_save_path, fourcc, 15, (1344, 760))

    # how many face images in all frames in this video
    face_cnt = 0

    names = []
    is_process = False
    is_save = False
    stream = cv2.VideoCapture(file_path)

    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break

        # To process one frame for each 2 frames to speed up
        if not is_process:
            is_process = True
        else:
            is_process = False

        if is_process:
            # Only interested in this ROI region(door area)
            frame_roi = frame[up_offsetY:down_offsetY,
                              left_offsetX:right_offsetX]
            known_face_locs = face_detector.detect(frame_roi)
            if len(known_face_locs) > 0:
                face_cnt += 1
                # to save 1 face image for each 2 processed face image
                if not is_save:
                    is_save = True
                else:
                    is_save = False

                if is_save:
                    image_save_path = "images/" + file_time + \
                        "_" + str(face_cnt) + ".jpg"
                    cv2.imwrite(image_save_path, frame_roi)

                #print("[INFO] " + str(len(known_face_locs)) + " face found.")
                predictions = face_recognizer.predict(
                    x_img=frame_roi, x_known_face_locs=known_face_locs)

                for name, (top, right, bottom, left) in predictions:
                    #print("- Found {} ".format(name))
                    names.append(name)
                    cv2.rectangle(frame, (left+left_offsetX, top+up_offsetY),
                                  (right+left_offsetX, bottom+up_offsetY), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left+left_offsetX, bottom+up_offsetY),
                                  (right+left_offsetX, bottom+up_offsetY+15), (0, 255, 0), -1)
                    cv2.putText(frame, name, (int((right-left)/4)+left+left_offsetX, bottom+up_offsetY+12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw the door area ROI rectangle
        cv2.rectangle(frame, (left_offsetX, up_offsetY),
                      (right_offsetX, down_offsetY), (0, 0, 0), 2)

        frame_to_video = imutils.resize(frame, width=1344, height=760)
        out.write(frame_to_video)

        if SHOW_GUI:
            cv2.imshow("Frame", frame_to_video)
        cv2.waitKey(1)

    # Uses to count how many images for each person
    name_counter = collections.Counter(names)

    # TODO: more strategy to determine muitiple person
    if face_cnt == 0:
        person_id = "None"
    else:
        person_id = name_counter.most_common(1)[0][0]

    full_video_path = HOME_PATH + "/" + video_save_path

    info_dict['NAME'] = person_id
    info_dict['TIMESTAMP'] = file_time.replace('_', ' ')
    info_dict['VIDEO_PATH'] = full_video_path

    print(info_dict)
    sql_updater.insert(info_dict)
    # Delete original video
    os.remove(file_path)
    out.release()
    stream.release()
    cv2.destroyAllWindows()


while True:
    '''
    if not sql_updater.running:
        try:
            sql_updater.connect()
        except:
            pass
        else:
            print("[INFO] Succeed to Connect SQL. ")
    '''
    if not q_path.empty() and not q_flag.empty():
        q_flag.get()
        process(q_path.get())

notifier.stop()
