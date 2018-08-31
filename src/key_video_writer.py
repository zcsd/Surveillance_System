# Class KeyVideoWriter

'''
Writing frames to video file a few seconds before the action takes place.
Writing frames to file a few seconds after the action finishes.
Utilizing threads to write key video to file.
Using deque and Queue data structures to pass on frames.
'''

from collections import deque
from threading import Thread
from queue import Queue
import time
import cv2


class KeyVideoWriter:
    def __init__(self, buffer_size=30, timeout=1.0):
        # store the maximum buffer size of frames to be kept
        # in memory along with the sleep timeout during threading
        self.buffer_size = buffer_size
        self.timeout = timeout

        self.frames = deque(maxlen=buffer_size)
        self.Q = None
        self.writer = None
        self.thread = None
        self.recording = False

    def update(self, frame):
        # update the frames buffer
        self.frames.appendleft(frame)

        # if we are recording, update the queue as well
        if self.recording:
            self.Q.put(frame)

    def start(self, output_path, fourcc, fps):
        # indicate we are recording, start the video writer,
        # and initialize the queue of frames that need to be written
        # to the video file
        self.recording = True
        self.writer = cv2.VideoWriter(output_path, fourcc, fps,
                                      (self.frames[0].shape[1], self.frames[0].shape[0]), True)
        self.Q = Queue()

        # loop over the frames in the deque structure and add them
        # to the queue
        for i in range(len(self.frames), 0, -1):
            self.Q.put(self.frames[i - 1])

        # start a thread write frames to the video file
        self.thread = Thread(target=self.write, args=())
        self.thread.daemon = True
        self.thread.start()

    def write(self):
        while True:
            # if we are done recording, exit the thread
            if not self.recording:
                return

            if not self.Q.empty():
                # grab the next frame in the queue and write it to the video file
                frame = self.Q.get()
                self.writer.write(frame)

            # otherwise, the queue is empty, so sleep for a bit
            else:
                time.sleep(self.timeout)

    def flush(self):
        # empty the queue by flushing all remaining frames to file
        while not self.Q.empty():
            frame = self.Q.get()
            self.writer.write(frame)

    def finish(self):
        # indicate we are done recording, join the thread,
        # flush all remaining frames in the queue to file, and
        # release the writer pointer
        self.recording = False
        self.thread.join()
        self.flush()
        self.writer.release()
        print("[INFO] One Key Video Saved.")
