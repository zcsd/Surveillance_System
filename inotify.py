import pyinotify
from queue import Queue
import imutils
import cv2
import datetime
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
import os
import collections

# True for showing video GUI, change to false on server OS
SHOW_GUI = True
# Set default working directory
HOME_PATH = "/home/zichun/SurveillanceSystem"
os.chdir(HOME_PATH)

# ROI for face detection
left_offsetX = 900
right_offsetX = 1550
up_offsetY = 650
down_offsetY = 1200

# set image resize ratio for face detection
faceD_resize_ratio = 0.5

# Initialize face detector
face_detector = FaceDetector(_scale=faceD_resize_ratio)

# Initialize face recognizer, method:SVM(16.0) or KNN(0.50)
face_recognizer = FaceRecognizer(method='SVM', threshold=16.10)

q_path = Queue()
q_flag = Queue()

wm = pyinotify.WatchManager()  # Watch Manager

class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        q_path.put(event.pathname)
    def process_IN_CLOSE_WRITE(self, event):
        q_flag.put(1)

mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_CREATE

notifier = pyinotify.ThreadedNotifier(wm, EventHandler())
wdd = wm.add_watch('/home/zichun/SurveillanceSystem/videos_temp', mask)
notifier.start()

def process(file_path):
    print("[INFO] Start to process {}".format(file_path))
    file_time = file_path.replace('/home/zichun/SurveillanceSystem/videos_temp/','').replace('.avi', '')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_save_path = "{}/{}.avi".format("videos", file_time)
    out = cv2.VideoWriter(video_save_path, fourcc, 35, (1344,760))
    face_cnt = 0
    names = [] 
    stream = cv2.VideoCapture(file_path)

    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
    
        # Only interested in this ROI region(door area)
        frame_roi = frame[up_offsetY:down_offsetY, left_offsetX:right_offsetX]
        known_face_locs = face_detector.detect(frame_roi)
        if len(known_face_locs) > 0:
            face_cnt += 1
            image_save_path = "images/" + file_time + "_" +str(face_cnt) + ".jpg"
            cv2.imwrite(image_save_path, frame_roi)

            #print("[INFO] " + str(len(known_face_locs)) + " face found.")
            # Start face recognition
            #time_s = time.time()
            predictions = face_recognizer.predict(
                x_img=frame_roi, x_known_face_locs=known_face_locs)
            #time_e = time.time()
            #print("predict time: {}s".format(time_e-time_s))
            for name, (top, right, bottom, left) in predictions:
                #print("- Found {} ".format(name))
                names.append(name)
                cv2.rectangle(frame, (left+left_offsetX, top+up_offsetY),
                              (right+left_offsetX, bottom+up_offsetY), (0, 255, 0), 2)
                cv2.rectangle(frame, (left+left_offsetX, bottom+up_offsetY),
                              (right+left_offsetX, bottom+up_offsetY+15), (0, 255, 0), -1)
                cv2.putText(frame, name, (int((right-left)/4)+left+left_offsetX, bottom+up_offsetY+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        cv2.rectangle(frame, (left_offsetX, up_offsetY), (right_offsetX, down_offsetY), (0, 0, 0), 2)

        frame_to_video = imutils.resize(frame, width=1344, height=760)
        out.write(frame_to_video)

        if SHOW_GUI:
            cv2.imshow("Frame", frame_to_video)
        cv2.waitKey(1)
    
    name_counter=collections.Counter(names)

    if face_cnt == 0:
        person_id = "None"
    else:
        person_id = name_counter.most_common(1)[0][0]
    
    full_video_path = HOME_PATH + video_save_path
    print("Found {} on {}, saved in {}".format(person_id, file_time, full_video_path))
    os.remove(file_path)
    out.release()
    stream.release()    
    cv2.destroyAllWindows()

while True:
    if not q_path.empty() and not q_flag.empty():
        q_flag.get()
        process(q_path.get())

#notifier.stop()