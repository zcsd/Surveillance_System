# Class FrameGrabber

from imutils.video import WebcamVideoStream
import imutils
import cv2
import time


RTSP_URL = "rtsp://satcam002:starasia2018@172.19.80.36:554/cam/realmonitor?channel=1&subtype=0"


class FrameGrabber:
    def __init__(self, src_from_rtsp):
        # src_from_rtsp is bool 0(from webcam) or 1(from rtsp)
        self.src_from_rtsp = src_from_rtsp

        if self.src_from_rtsp:
            self.source = RTSP_URL
        else:
            # stream from webcam, 0,1,2...represnt different webcam
            self.source = 0

        self.video_stream = None

    def start(self):
        # Use original opencv capture mode, no buffer
        self.video_stream = cv2.VideoCapture(self.source)
        print("[INFO] Starting Video Stream...")

    def read(self):
        (grabbed, raw_frame) = self.video_stream.read()

        if not grabbed:
            print("[ERROR] Fail to grab frame.")
            # Program will crash if grabbing fail.
            return 0
        else:
            return raw_frame

    def stop(self):
        self.video_stream.release()
        print("[INFO] Stream Closed. ")
