# Class KnnFaceRecognizer

'''
Recognizes faces in given image using a trained KNN classifier
'''

from sklearn import neighbors
import face_recognition as fr
import pickle

"""
:param knn_clf: a knn classifier object.
:param _distance_threshold: distance threshold for face classification. the larger it is, the more chance
       of mis-classifying an unknown person as a known one.
:return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
    For faces of unrecognized persons, the name 'unknown' will be returned.
"""

MODEL_SAVE_PATH = "classifier/trained_knn_model.clf"

class KnnFaceRecognizer:
    def __init__(self, _distance_threshold=0.6):
        self._distance_threshold = _distance_threshold
        with open(MODEL_SAVE_PATH, 'rb') as f:
            self.knn_clf = pickle.load(f)
        print("[INFO] Face Recognition is working...")

    def predict(self, x_img, x_known_face_locs):
        _distance_threshold = self._distance_threshold
        if self.knn_clf is None:
            raise Exception("Must supply knn classifier.")

        # If no faces are found in the image, return an empty result.
        if len(x_known_face_locs) == 0:
            return []

        # Find encodings for faces in the test iamge
        face_encodings = fr.face_encodings(x_img, known_face_locations=x_known_face_locs)

        # Use the KNN model to find the best matches for the test face
        closet_distance = self.knn_clf.kneighbors(face_encodings, n_neighbors=1)
        are_matches = [closet_distance[0][i][0] <= _distance_threshold for i in range(len(x_known_face_locs))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(face_encodings), x_known_face_locs, are_matches)]
