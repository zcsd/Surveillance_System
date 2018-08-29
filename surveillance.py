#!/usr/bin/python3
# Python 3.5+
# Surveillance system
# Author: @zichun

from key_video_writer import KeyVideoWriter
from motion_detector import MotionDetector
from frame_grabber import FrameGrabber
from imutils.video import FPS
import cv2
import imutils
import datetime
import time
import os


# True for showing video GUI, change to false on server OS
SHOW_GUI = True

# Set default working directory
HOME_PATH = "/home/zichun/SurveillanceSystem"
os.chdir(HOME_PATH)

# ROI for motion detection
left_offsetX = 900
right_offsetX = 1550
up_offsetY = 650
down_offsetY = 1200

# set image resize ratio for motion and face detection
motion_resize_ratio = 0.25

# initialize key video writer and the consecutive number of
# frames that have NOT contained any action
key_video_writer = KeyVideoWriter()
num_consec_frames = 0

# init motion frame counts
num_motion_frames = 0
num_total_frames = 0
start_recording = False

# Start videostream, 0 for webcam, 1 for rtsp
frame_grabber = FrameGrabber(1)
frame_grabber.start()

# Initialize motion detector
motion_detector = MotionDetector(_resize_ratio=motion_resize_ratio)
num_frame_read = 0  # no. of frames read

# FPS calculation
fps = FPS().start()

while True:
    # grab frame
    frame = frame_grabber.read()
    # frame will be used by motion detector, create another show copy
    frame_show = frame.copy()
    # Only interested in this ROI region(door area)
    frame_roi = frame[up_offsetY:down_offsetY, left_offsetX:right_offsetX]
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    motion_locs = motion_detector.update(frame_gray)

    # form a nice average before motion detection
    if num_frame_read < 20:
        num_frame_read += 1
        continue
    # boolean used to indicate if the consecutive frames
    # counter should be updated
    update_consec_frames = True

    num_total_frames += 1

    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
 
    if len(motion_locs) > 0:
        num_motion_frames += 1

        # reset the number of consecutive frames with NO action to zero
        update_consec_frames = False
        num_consec_frames = 0
        # if we are not already recording, start recording
        if not key_video_writer.recording and start_recording:
            print("[INFO] Start video recording...")
            video_save_path = "{}/{}.avi".format("videos_temp", ts)
            key_video_writer.start(
                video_save_path, cv2.VideoWriter_fourcc(*'MJPG'), 30)

    if update_consec_frames:
        num_consec_frames += 1
    
    # update the key frame video buffer
    key_video_writer.update(frame_show)
    
    # print("motion_frame: {}, total_frame: {}, conse_frame: {}".format(num_motion_frames, num_total_frames, num_consec_frames))
    
    if not key_video_writer.recording and not start_recording and num_motion_frames >= 5:
        motion_ratio = num_motion_frames / num_total_frames
        # print("motion_ratio: {}".format(motion_ratio))
        if motion_ratio > 0.15:
            start_recording = True
            print("[INFO] Motion detected.")
        
    if num_consec_frames >= 30:
        num_consec_frames = 30
        num_total_frames = 0
        num_motion_frames = 0
        motion_ratio = 0
        start_recording = False

    # if we are recording and reached a threshold on consecutive
    # number of frames with no action, stop recording the clip
    if key_video_writer.recording and num_consec_frames >= 30:
        key_video_writer.finish()
        num_motion_frames = 0
        num_total_frames = 0
        start_recording = False
    
    if SHOW_GUI:
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
frame_grabber.stop()
cv2.destroyAllWindows()
