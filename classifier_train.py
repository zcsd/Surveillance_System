# Class ClassifierTrain

"""
Method 1: SVM (Support Vector Machine)
Method 2: KNN (K-Nearst-Neighbors)
"""

from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition as fr
from sklearn import neighbors
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os
import os.path
import pickle
import math
import time

"""
 Training images directory structure:
    <train>/
    ├── <person1>/
    │   ├── <somename1>.jpg
    │   ├── <somename2>.jpg
    │   ├── ...
    ├── <person2>/
    │   ├── <somename1>.jpg
    │   └── <somename2>.jpg
    └── ...

Testing images directory structure:
    <test>/
    ├── <person1>/
    │   ├── <somename1>.jpg
    │   ├── <somename2>.jpg
    │   ├── ...
    ├── <person2>/
    │   ├── <somename1>.jpg
    │   └── <somename2>.jpg
    └── ...
"""

TRAIN_DATA_PATH = "faces/train"
TEST_DATA_PATH = "faces/test"
KNN_SAVE_PATH = "classifier/trained_knn_model.clf"
LSVM_SAVE_PATH = "classifier/trained_lsvm_model.clf"


class ClassifierTrain:
    def __init__(self, method='LSVM'):
        self.method = method
        self.X_train = []
        self.y_train = []
        self.total_persons_train = 0
        self.total_images_train = 0
        self.X_test = []
        self.y_test = []
        self.total_persons_test = 0
        self.total_images_test = 0
    
    def start(self):
        print("[INFO] It's going to train a face classifer.")
        self.X_train, self.y_train = self.prepare_data(TRAIN_DATA_PATH)
        self.X_test, self.y_test = self.prepare_data(TEST_DATA_PATH)

        if self.method == 'LSVM':
            self.lsvm_train()
        elif self.method == 'KNN':
            self.knn_train(train_n_neighbors=7)
        elif self.method == "ALL":
            self.lsvm_train()
            self.knn_train(train_n_neighbors=7)
    
    def prepare_data(self, path):
        time_start = time.time()

        if "train" in path:
            mode = "training"
        elif "test" in path:
            mode = "testing"
        
        print("[INFO] Start to prepare data for {}...".format(mode))

        X = []
        y = []

        # Loop through each person in the training set
        for class_dir in os.listdir(path):
            if not os.path.isdir(os.path.join(path, class_dir)):
                continue

            # Loop through each training image for the current person
            for image_path in image_files_in_folder(os.path.join(path, class_dir)):
                image = fr.load_image_file(image_path)
                faces_boxes = fr.face_locations(
                    image, number_of_times_to_upsample=1)

                if len(faces_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    print("[WARNING] Image {} not suitable for {}: {}".format(
                        image_path, mode, "Didn't find a face" if len(faces_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(fr.face_encodings(
                        image, known_face_locations=faces_boxes)[0])
                    y.append(class_dir)
        time_end = time.time()
        time_spent = time_end - time_start
        print("[INFO] Data has been prepared well for {}. Time spent: {:.3f}s".format(mode, time_spent))
        
        return X, y

    def knn_train(self, train_n_neighbors):
        print("[INFO] Start to train KNN classifier...")
        time_start = time.time()
        # Determine how many neighbors to use for weighting in the KNN classifier
        if train_n_neighbors is None:
            train_n_neighbors = int(round(math.sqrt(len(self.X_train))))
            print("[INFO] Chose n_neighbors for KNN automatically:", train_n_neighbors)
        else:
            print("[INFO] Chose n_neighbors for KNN:", train_n_neighbors)

        # Create and train the KNN classifier
        trained_knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=train_n_neighbors, algorithm='ball_tree', weights='distance')
        trained_knn_clf.fit(self.X_train, self.y_train)
        acc_knn = accuracy_score(self.y_test, trained_knn_clf.predict(self.X_test))

        # Save the trained KNN classifier
        if KNN_SAVE_PATH is not None:
            with open(KNN_SAVE_PATH, 'wb') as f:
                pickle.dump(trained_knn_clf, f)
                time_end = time.time()
                time_spent = time_end - time_start
                print("[INFO] Training completed! KNN Classifier saved! Accuracy:{}, Time spent: {:.3f}s".format(acc_knn, time_spent))

    def lsvm_train(self):
        print("[INFO] Start to train Linear SVM classifier...")
        time_start = time.time()

        trained_lsvm_clf = LinearSVC()
        trained_lsvm_clf.fit(self.X_train, self.y_train)
        acc_svm = accuracy_score(self.y_test, trained_lsvm_clf.predict(self.X_test))

        # Save the trained SVM classifier
        if LSVM_SAVE_PATH is not None:
            with open(LSVM_SAVE_PATH, 'wb') as f:
                pickle.dump(trained_lsvm_clf, f)
                time_end = time.time()
                time_spent = time_end - time_start
                print("[INFO] Training completed! SVM Classifier saved! Accuracy:{}, Time spent: {:.3f}s".format(acc_svm, time_spent))