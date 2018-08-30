# Class FrameGrabber

from imutils.video import WebcamVideoStream
import imutils
import cv2
import time

RTSP_URL = "rtsp://satcam002:starasia2018@172.19.80.36:554/cam/realmonitor?channel=1&subtype=0"
# Camera resolution setting
FRAME_WIDTH = 320
FRAME_HEIGHT = 240


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
        print("[INFO] Starting Video Stream...")
        if self.src_from_rtsp:
            self.video_stream = WebcamVideoStream(self.source)
        else:
            self.video_stream = WebcamVideoStream(self.source)
            self.video_stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.video_stream.stream.set(
                cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.video_stream.start()
        time.sleep(1.0)  # for warm up camera, 1 second

    def read(self):
        raw_frame = self.video_stream.read()
        '''
        if self.src_from_rtsp:
            # Return same-sized frame for both rtsp and webcam
            resized_frame = imutils.resize(raw_frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
            return resized_frame
        else:
            return raw_frame
        '''
        return raw_frame

    def stop(self):
        self.video_stream.stop()
        print("[INFO] Stream Closed. ")
