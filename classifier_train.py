# Class ClassifierTrain

"""
Method 1: SVM (Support Vector Machine)
Method 2: KNN (K-Nearst-Neighbors)
"""

from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition as fr
from sklearn import neighbors
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import cv2
import numpy as np
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
        self.file_name_train = []
        self.X_train = []
        self.y_train = []
        self.total_persons_train = 0
        self.total_images_train = 0
        self.file_name_test = []
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

        self.data_visualization(self.X_train, self.y_train, 'train', False)
    
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
            
            # Count how many different persons/faces in training or testing
            if mode == "training":
                self.total_persons_train += 1
            elif mode == "testing":
                self.total_persons_test += 1

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
                    file_name = image_path.strip(path+'/'+class_dir+'/')
                    # Count how many images in training or testing
                    if mode == "training":
                        self.total_images_train += 1
                        self.file_name_train.append(file_name)
                    elif mode == "testing":
                        self.total_images_test += 1
                        self.file_name_test.append(file_name)

                    # Add face encoding for current image to the training set
                    X.append(fr.face_encodings(
                        image, known_face_locations=faces_boxes)[0])
                    y.append(class_dir)

        time_end = time.time()
        time_spent = time_end - time_start
        print("[INFO] Data has been prepared well for {}. Time: {:.3f}s".format(mode, time_spent))
        
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
                print("[INFO] KNN training completed with {} classes and {} images. Time: {:.3f}s"
                      .format(self.total_persons_train, self.total_images_train, time_spent))
                print("[INFO] KNN testing/verify accuracy with {} classes and {} images: {:.0%}".format(self.total_persons_test, self.total_images_test, acc_knn))
    
    def lsvm_train(self):
        print("[INFO] Start to train Linear SVM classifier...")
        time_start = time.time()

        trained_lsvm_clf = LinearSVC()
        trained_lsvm_clf.fit(self.X_train, self.y_train)
        acc_lsvm = accuracy_score(self.y_test, trained_lsvm_clf.predict(self.X_test))

        # Save the trained SVM classifier
        if LSVM_SAVE_PATH is not None:
            with open(LSVM_SAVE_PATH, 'wb') as f:
                pickle.dump(trained_lsvm_clf, f)
                time_end = time.time()
                time_spent = time_end - time_start
                print("[INFO] LSVM training completed with {} classes and {} images. Time: {:.3f}s"
                      .format(self.total_persons_train, self.total_images_train, time_spent))
                print("[INFO] LSVM testing/verify accuracy with {} classes and {} images: {:.0%}".format(self.total_persons_test, self.total_images_test, acc_lsvm))

    def data_visualization(self, X, y, mode, show_file_name):
        # Transfer 128-Dimension face encoding to 2-D space
        X_2d = TSNE(n_components=2).fit_transform(X)
        
        fig = plt.figure()
        # Consturct a orderd set with same order in y
        set_y = {}
        for i, t in enumerate(y):
            if len(set_y) == 0:
                set_y[0] = t
            else:
                if t in set_y.values():
                    pass
                else:
                    set_y[len(set_y)] = t

        # Display the corresponding file name in scatter position
        if show_file_name:
            file_name_list = []
            if mode == 'train':
                file_name_list = self.file_name_train
            elif mode == 'test':
                file_name_list = self.file_name_test
            
            for i, p in enumerate(X_2d):
                plt.text(p[0], p[1], file_name_list[i] , horizontalalignment='center', 
                        verticalalignment='center', fontsize=5, color='gray')
        '''
        # Start and end number for each person's face images
        n_start = 0
        n_end = 0

        for i in range(len(set_y)):
            n_end = n_start + y.count(set_y[i])
            # Plot scatter for each person
            plt.scatter(X_2d[n_start:n_end , 0], X_2d[n_start:n_end, 1], label=set_y[i])
            # Calculate center point for each person
            p_x = [p for p in X_2d[n_start:n_end, 0]]
            p_y = [p for p in X_2d[n_start:n_end, 1]]
            center_x = sum(p_x) / y.count(set_y[i])
            center_y = sum(p_y) / y.count(set_y[i])
            # Draw name text for each person on each cluster center
            plt.text(center_x, center_y, set_y[i], horizontalalignment='center', 
                     verticalalignment='center', fontsize=15, color='black')
            
            n_start = n_end
        '''
        # To display classifier decision region
        encoder = LabelEncoder()
        encoder.fit(y)
        # Numerical encoding of identities
        y_1d = encoder.transform(y)
        show_lsvm_clf = svm.SVC(kernel='poly', degree=3)
        show_lsvm_clf.fit(X_2d, y_1d)    
        fig = plot_decision_regions(X=X_2d, y=y_1d, clf=show_lsvm_clf, legend=2)

        # plt.legend(bbox_to_anchor=(1, 1));
        
        #plt.ylim(-25, 25)
        #plt.xlim(-25, 25)
        plt.show()
