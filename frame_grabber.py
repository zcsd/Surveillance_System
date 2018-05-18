# Class FrameGrabber

from imutils.video import WebcamVideoStream
import imutils
import cv2
import time

RTSP_URL = "rtsp://admin:admin123@172.19.80.30:554/videoMain"
# Camera resolution setting
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

class FrameGrabber:
    def __init__(self, src_from_rtsp):
        self.src_from_rtsp = src_from_rtsp

        if self.src_from_rtsp:
            self.source = RTSP_URL
        else:
            self.source = 1

        self.video_stream = None

    def start(self):
        if self.src_from_rtsp:
            self.video_stream = WebcamVideoStream(self.source)
        else:
            self.video_stream = WebcamVideoStream(self.source)
            self.video_stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.video_stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.video_stream.start()
        time.sleep(1.0)  # for warm up camera, 1 second

    def read(self):
        raw_frame = self.video_stream.read()

        if self.src_from_rtsp:
            frame = imutils.resize(raw_frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
            return frame
        else:
            return raw_frame

    def stop(self):
        self.video_stream.stop()
