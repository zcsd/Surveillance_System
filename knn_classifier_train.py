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
:param trainDir: directory that contains a sub-directory for each known person, with its name.
 Structure:
    <trainDir>/
    ├── <person1>/
    │   ├── <somename1>.jpeg
    │   ├── <somename2>.jpeg
    │   ├── ...
    ├── <person2>/
    │   ├── <somename1>.jpeg
    │   └── <somename2>.jpeg
    └── ...

:param modelSavePath: path to save model on disk
:param nNeighbors: number of neighbors to weigh in classification. Chosen automatically if not specified
:param knnAlgo: underlying data structure to support knn.default is ball_tree
:param verbose: verbosity of training
"""

class KnnClassifierTrain:
    def __init__(self, knnAlgo='ball_tree', verbose = False):
        self.knnAlgo = knnAlgo
        self.verbose = verbose

    def train(self, trainDir, modelSavePath, nNeighbors):
        knnAlgo = self.knnAlgo
        verbose = self.verbose
        X = []
        y = []

        # Loop through each person in the training set
        for classDir in os.listdir(trainDir):
            if not os.path.isdir(os.path.join(trainDir, classDir)):
                continue

            # Loop through each training image for the current person
            for imgPath in image_files_in_folder(os.path.join(trainDir, classDir)):
                image = fr.load_image_file(imgPath)
                faceBoxes = fr.face_locations(image)

                if len(faceBoxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(imgPath, "Didn't find a face" if len(faceBoxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(fr.face_encodings(image, known_face_locations=faceBoxes)[0])
                    y.append(classDir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if nNeighbors is None:
            nNeighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose nNeighbors automatically:", nNeighbors)

        # Create and train the KNN classifier
        knnClf = neighbors.KNeighborsClassifier(n_neighbors=nNeighbors, algorithm=knnAlgo, weights='distance')
        knnClf.fit(X, y)

        # Save the trained KNN classifier
        if modelSavePath is not None:
            with open(modelSavePath, 'wb') as f:
                pickle.dump(knnClf, f)
                print("[INFO] Training completed! Classifier saved!")
