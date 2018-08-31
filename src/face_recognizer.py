# Class FaceRecognizer

'''
Recognizes faces in given image using a trained KNN/SVM classifier
'''

import face_recognition as fr
import pickle
import cv2

KNN_SAVE_PATH = "classifier/trained_knn_model.clf"
SVM_SAVE_PATH = "classifier/trained_svm_model.clf"


class FaceRecognizer:
    def __init__(self, method='SVM', threshold=16.0):
        # Default method is SVM, also have KNN
        self.threshold = threshold
        self.method = method
        self.clf = None

        if self.method == 'KNN':
            clf_path = KNN_SAVE_PATH
        elif self.method == 'SVM':
            clf_path = SVM_SAVE_PATH

        with open(clf_path, 'rb') as f:
            self.clf = pickle.load(f)
            self.classes_dict = {}
            class_index = 0
            for class_ in self.clf.classes_:
                # map label/name to corresponding index
                self.classes_dict[class_] = class_index
                class_index += 1

        print("[INFO] Face Recognition is working, {} is used and {} persons are in database.".format(
            self.method, class_index))

    def predict(self, x_img, x_known_face_locs):
        if self.clf is None:
            raise Exception("Must supply a classifier.")

        # If no faces are found in the image, return an empty result.
        if len(x_known_face_locs) == 0:
            return []

        # Find encodings for faces in the test iamge
        face_encodings = fr.face_encodings(
            x_img, known_face_locations=x_known_face_locs)

        # It's will be a match(True) if distance/parobability is within threshod.
        are_matches = []  # Note: it's a list, there may be muitiple faces in image
        if self.method == 'KNN':
            # Get the closet distance from all classes, it's a float number(0.xx)
            closet_distance = self.clf.kneighbors(
                face_encodings, n_neighbors=1)
            # if the closet distance <= threshold(0.5~0.6), it could be considered to be a match.
            # More tight match if threshold is set smaller
            are_matches = [closet_distance[0][i][0] <=
                           self.threshold for i in range(len(x_known_face_locs))]
        elif self.method == 'SVM':
            # Get a list of probability of all classes
            probabilities = self.clf.decision_function(face_encodings)
            # Use max method to get the largest probability, then compare with threshold
            # if the max probabilty >= threshold(~16.0), it could be considered to be a match.
            # More tight match if threshold is set larger.
            are_matches = [max(probabilities[i]) >=
                           self.threshold for i in range(len(x_known_face_locs))]

        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.clf.predict(face_encodings), x_known_face_locs, are_matches)]
