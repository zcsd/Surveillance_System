# Office surveillance system

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

from motion_detector import MotionDetector
from face_detector import FaceDetector
from knn_classifier_train import KnnClassifierTrain
from knn_face_recognizer import KnnFaceRecognizer
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from subprocess import call
import numpy as np
import argparse
import pickle
import datetime
import imutils
import time
import cv2

# Camera resolution setting
frameWidth = 320
frameHeight = 240

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", help="train a KNN faces classifier", action="store_true")
ap.add_argument("-c", "--collect_faces", help="collect faces images for saving and training", action="store_true")
args = ap.parse_args()

if args.train:
	print("[INFO] Training KNN classifier...")
	knnclassifierTrain = KnnClassifierTrain()
	knnclassifierTrain.train(train_dir="faces/train", model_save_path="classifier/trained_knn_model.clf", n_neighbors=None)
	exit()
elif args.collect_faces:
	print("[INFO] Start to collect face images...")
	call(["python3", "face_collector.py"])
	print("[INFO] Collection Done.")
	exit()

# Start camera videostream
print("[INFO] Starting camera...")
# 0 for default webcam, 1/2/3... for external webcam
videoStream = WebcamVideoStream(src=1)
videoStream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
videoStream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
videoStream.start()
time.sleep(1.0) # for warm up camera, 1 second

# Initialize motion detector
motionDetector = MotionDetector()
noframesRead = 0 # no. of frames read

# Initialize face detector
faceDetector = FaceDetector()

# Initialize face recognizer and load trained classifier
with open("classifier/trained_knn_model.clf", 'rb') as f:
	knnClf = pickle.load(f)
knnfaceRecognizer = KnnFaceRecognizer()

# FPS calculation
fps = FPS().start()

while True:
	# grab frame
	frame = videoStream.read()
	frameShow = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	motionLocs = motionDetector.update(gray)
	# form a nice average before motion detection
	if noframesRead < 30:
		noframesRead += 1
		continue

	if len(motionLocs) > 0:
		# may consider to process in every other frame to accelerate
		# !!!!!!!!!!!!!!!!!!!!!
		faceLocs = faceDetector.detect(frame)
		if len(faceLocs) > 0:
			print("[INFO] " + str(len(faceLocs)) + " face found.")
			# Start face recognition
			predictions = knnfaceRecognizer.predict(X_img=frame, X_face_locations=faceLocs, knn_clf=knnClf)
			for name, (top, right, bottom, left) in predictions:
				print("- Found {} at ({}, {})".format(name, left, top))
				# cv2.rectangle(frameShow, (left, top), (right, bottom), (0, 255, 0), 2)
				cv2.rectangle(frameShow, (left, bottom), (right, bottom+25), (0, 255, 0), -1)
				cv2.putText(frameShow, name, (int((right-left)/3)+left,bottom+18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
			
			# Draw green bounding box on faces in frameShow
			for top, right, bottom, left in faceLocs:
				# Scale back up face locations
				top *= 1
				right *= 1
				bottom *= 1
				left *= 1
				cv2.rectangle(frameShow,(left, top), (right, bottom), (0, 255, 0), 2)
			
		# initialize the minimum and maximum (x, y)-coordinates
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		# loop over the locations of motion and accumulate the
		# minimum and maximum locations of the bounding boxes
		for l in motionLocs:
			(x, y, w, h) = cv2.boundingRect(l)
			(minX, maxX) = (min(minX, x), max(maxX, x + w))
			(minY, maxY) = (min(minY, y), max(maxY, y + h))

		# draw red bounding box on moving body
		cv2.rectangle(frameShow, (minX, minY), (maxX, maxY), (0, 0, 255), 3)

	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%d %b %Y %H:%M:%S")

	cv2.putText(frameShow, ts, (10, frameShow.shape[0] - 10), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
	cv2.imshow("Frame", frameShow)

	fps.update()

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up and release memory
cv2.destroyAllWindows()
videoStream.stop()