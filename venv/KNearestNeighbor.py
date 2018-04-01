import numpy as np
import pandas as ps
from sklearn import preprocessing,cross_validation,neighbors
import warnings
from math import sqrt
from collections import Counter
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            #euclidean distance between single feature and input point  {[(x(1) - x'(1))^2] + [(x(2) - x'(2))^2]}^1/2
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    #print(distances)
    #sorted the distances list based on the eclidean distances
    #gets the top k records , where each record is in turn an array
    #get the second element or first index element from each array which is the class
    votes = [i[1] for i in sorted(distances)[:k]]

    #Eg: if array is X=['r','r','r','r','r','k','k','k'] then
    # Counter(X) Aggregates the X's based on there occurence and will give a dictionary Counter({'r': 5, 'k': 3}) and
    # Counter(X).most_common(1) will give an array containing the single element that occured most as in the above r has occured 5 times [('r', 5)]
    # Counter(X).most_common(2) will give an array containing 2 elements that occured the most in order in above its r =5 and k =3 [('r', 5),('k',3)]
    # Counter(X).most_common(1)[0] returns the first element in the array from above which is ('r',5)
    # Counter(votes).most_common(1)[0][0] return the first element of the above ('r',5) which is r
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

def localImplKNearestNeighbor():
    df = ps.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['Id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    print("test_set is :", test_set)
    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print('Accuracy with local Implementation:', correct / total)



def inBuiltImplKnearestNeighbor():
    # dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    # new_features = [5, 7]

    #result = k_nearest_neighbors(dataset, new_features)

    df = ps.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?', -99999, inplace=True)
    df.drop(['Id'], 1, inplace=True)

    X = np.array(df.drop(['Class'], 1))
    Y = np.array(df['Class'])

    x_training, x_test, y_training, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_training, y_training)
    accuracy = clf.score(x_test, y_test)
    print("Accuracy with inBuilt Implementation",accuracy)

    example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
    example_measures = example_measures.reshape(1, -1)
    prediction = clf.predict(example_measures)
    print("Prediction with In Built Algo",prediction)

localImplKNearestNeighbor()
#inBuiltImplKnearestNeighbor()