#!/usr/bin/python3
# Python 3.5+
# Training system for face recognition
# Author: @zichun

"""
How to use: python3 training.py
"""

from src.classifier_train import ClassifierTrain


# method: SVM, KNN, ALL
classifier_train = ClassifierTrain(method='ALL')
classifier_train.start()
