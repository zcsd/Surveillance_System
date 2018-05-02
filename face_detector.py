# Class FaceDetector

'''
-------------face detection Lib using Dlib---------------
Dlib: C++ toolkit containing machine learning algorithms
face recognition: face detection and recognition open source
python library based on Dlib
'''

import face_recognition as fr
import cv2

class FaceDetector:
	def __init__(self, scale=0.25):
		self.scale = scale 

	def detect(self, image):
		faceLocs = []
		# Downsampling image for faster processing
		smallImage = cv2.resize(image, (0, 0), fx=self.scale, fy=self.scale)
		# OpenCV color use BGR model; face_recognition use RGB color model
		rgbSmallImage = smallImage[:, :, ::-1]
		# Currently,use default HOG-based(Histogram of Oriented Gradients) model
		# It's a traditional object tracking method, but fairly accurate and fast
		faceLocs = fr.face_locations(rgbSmallImage)
		# CNN(convolutional neural network) model is more accurate, but too slow without GPU
		# faceLocs = fr.face_locations(smallImage, model="cnn")

		# faceLocs need to mutiply 1/scale!!!!REMEMBER TO ADDDDDDDDDDD!
		
		return faceLocs