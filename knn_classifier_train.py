# Class KnnClassifierTrain

'''
------------------------k-Nearest-Neighbors (KNN) algorithm-------------------------------
The knn classifier is first trained on a set of labeled faces
and can then predict the person in an unknown image by finding
the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the
training set are one image of A and two images of B, The result would be 'B'.
'''

from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors
import os
import os.path
import pickle
import math
import face_recognition as fr

"""
:param train_dir: directory that contains a sub-directory for each known person, with its name.
 Structure:
    <train_dir>/
    ├── <person1>/
    │   ├── <somename1>.jpeg
    │   ├── <somename2>.jpeg
    │   ├── ...
    ├── <person2>/
    │   ├── <somename1>.jpeg
    │   └── <somename2>.jpeg
    └── ...

:param model_save_path: path to save model on disk
:param n_neighbors: number of neighbors to weigh in classification. Chosen automatically if not specified
:param _knn_algo: underlying data structure to support knn.default is ball_tree
:param _verbose: verbosity of training
"""

TRAIN_DATA_PATH = "faces/train"
MODEL_SAVE_PATH = "classifier/trained_knn_model.clf"

class KnnClassifierTrain:
    def __init__(self, _knn_algo='ball_tree', _verbose = True, _n_neighbors = 4):
        self._knn_algo = _knn_algo
        self._verbose = _verbose
        self._n_neighbors = _n_neighbors

    def train(self):
        print("[INFO] Training KNN classifier...")
        X = []
        y = []

        # Loop through each person in the training set
        for class_dir in os.listdir(TRAIN_DATA_PATH):
            if not os.path.isdir(os.path.join(TRAIN_DATA_PATH, class_dir)):
                continue

            # Loop through each training image for the current person
            for image_path in image_files_in_folder(os.path.join(TRAIN_DATA_PATH, class_dir)):
                image = fr.load_image_file(image_path)
                faces_boxes = fr.face_locations(image, number_of_times_to_upsample=2)

                if len(faces_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if self._verbose:
                        print("Image {} not suitable for training: {}".format(image_path, "Didn't find a face" if len(faces_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(fr.face_encodings(image, known_face_locations=faces_boxes)[0])
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if self._n_neighbors is None:
            self._n_neighbors = int(round(math.sqrt(len(X))))
            if self._verbose:
                print("Chose n_neighbors automatically:", self._n_neighbors)

        # Create and train the KNN classifier
        trained_knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self._n_neighbors, algorithm=self._knn_algo, weights='distance')
        trained_knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if MODEL_SAVE_PATH is not None:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pickle.dump(trained_knn_clf, f)
                print("[INFO] Training completed! Classifier saved!")
