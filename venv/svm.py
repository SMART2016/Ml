import numpy as np
import pandas as ps
from sklearn import preprocessing, cross_validation, neighbors,svm
import warnings
from math import sqrt
from collections import Counter
import random


def inBuiltImplSvm():
    # dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    # new_features = [5, 7]

    #result = k_nearest_neighbors(dataset, new_features)

    df = ps.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?', -99999, inplace=True)
    df.drop(['Id'], 1, inplace=True)

    X = np.array(df.drop(['Class'], 1))
    Y = np.array(df['Class'])

    x_training, x_test, y_training, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = svm.SVC()
    clf.fit(x_training, y_training)
    accuracy = clf.score(x_test, y_test)
    print("Accuracy with inBuilt Implementation",accuracy)

    example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
    example_measures = example_measures.reshape(1, -1)
    prediction = clf.predict(example_measures)
    print("Prediction with In Built Algo",prediction)


inBuiltImplSvm()