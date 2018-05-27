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
    def __init__(self, _scale=1):
        self._scale = _scale

    def detect(self, image, known_motion_locs):
        face_locs = []
        test_locs =[]
        # Downsampling image for faster processing
        small_image = cv2.resize(image, (0, 0), fx=self._scale, fy=self._scale)
        # OpenCV color use BGR model; face_recognition use RGB color model
        rgb_small_image = cv2.cvtColor(small_image , cv2.COLOR_BGR2RGB)
        # rgb_small_image = small_image[:, :, ::-1]
        # Currently,use default HOG-based(Histogram of Oriented Gradients) model
        # It's a traditional object tracking method, but fairly accurate and fast
        # initialize the minimum and maximum (x, y)-coordinates
        '''
        # find face in motion area, need optimization
        (minX, minY) = (999999, 999999)
        (maxX, maxY) = (-999999, -999999)
        for l in known_motion_locs:
            (x, y, w, h) = cv2.boundingRect(l)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))
            zoom_face_locs = fr.face_locations(rgb_small_image[minY:maxY,minX:maxX])
            #cv2.imwrite("test.jpg", rgb_small_image[minY:maxY,minX:maxX])
            for (top, right, bottom, left) in zoom_face_locs:
                ori_face_loc = (top+y, right+x, bottom+y, left+x)
                test_locs.append(ori_face_loc)
        '''
        face_locs = fr.face_locations(rgb_small_image, number_of_times_to_upsample=1)
        # CNN(convolutional neural network) model is more accurate, but too slow without GPU
        # face_locs = fr.face_locations(small_image, model="cnn")

        return face_locs
