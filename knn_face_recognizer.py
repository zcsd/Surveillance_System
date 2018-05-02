# Class KnnFaceRecognizer

'''
Recognizes faces in given image using a trained KNN classifier
'''

from sklearn import neighbors
import face_recognition as fr

"""
:param knnClf: a knn classifier object.
:param distanceThreshold: distance threshold for face classification. the larger it is, the more chance
       of mis-classifying an unknown person as a known one.
:return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
    For faces of unrecognized persons, the name 'unknown' will be returned.
"""

class KnnFaceRecognizer:
    def __init__(self, distanceThreshold=0.6):
        self.distanceThreshold = distanceThreshold

    def predict(self, xImg, xFaceLocs, knnClf):
        distanceThreshold = self.distanceThreshold
        if knnClf is None:
            raise Exception("Must supply knn classifier.")

        # If no faces are found in the image, return an empty result.
        if len(xFaceLocs) == 0:
            return []

        # Find encodings for faces in the test iamge
        facesEncodings = fr.face_encodings(xImg, known_face_locations=xFaceLocs)

        # Use the KNN model to find the best matches for the test face
        closestDistances = knnClf.kneighbors(facesEncodings, n_neighbors=1)
        areMatches = [closestDistances[0][i][0] <= distanceThreshold for i in range(len(xFaceLocs))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knnClf.predict(facesEncodings), xFaceLocs, areMatches)]
        