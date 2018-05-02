# Class KnnFaceRecognizer

'''
Recognizes faces in given image using a trained KNN classifier
'''

from sklearn import neighbors
import face_recognition

"""
:param knn_clf: a knn classifier object.
:param distance_threshold: distance threshold for face classification. the larger it is, the more chance
       of mis-classifying an unknown person as a known one.
:return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
    For faces of unrecognized persons, the name 'unknown' will be returned.
"""

class KnnFaceRecognizer:
    def __init__(self, distance_threshold=0.6):
        self.distance_threshold = distance_threshold

    def predict(self, X_img, X_face_locations, knn_clf=None):
        distance_threshold = self.distance_threshold
        if knn_clf is None:
            raise Exception("Must supply knn classifier.")

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        