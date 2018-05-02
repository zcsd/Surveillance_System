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
import face_recognition

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
:param knn_algo: underlying data structure to support knn.default is ball_tree
:param verbose: verbosity of training
"""

class KnnClassifierTrain:
    def __init__(self, knn_algo='ball_tree', verbose = False):
        self.knn_algo = knn_algo
        self.verbose = verbose

    def train(self, train_dir, model_save_path, n_neighbors):
        knn_algo = self.knn_algo
        verbose = self.verbose
        X = []
        y = []

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
                print("[INFO] Training complete! Classifier saved!")
