from utils import *
import preproc
import random
import numpy as np
from dtree import dtree_bagging, dtree_ada_boost
from svm import svm_bagging, svm_ada_boost
from knn import knn_bagging, knn_ada_boost
from naivebayes import nb_bagging, nb_ada_boost

bagging_map = {"dtree": dtree_bagging, "svm": svm_bagging, "knn": knn_bagging, "nb": nb_bagging}
ada_boost_map = {"dtree": dtree_ada_boost, "svm": svm_ada_boost, "knn": knn_ada_boost, "nb": nb_ada_boost}


def train_bagging(method):
    classifier = bagging_map[method]
    train_data, train_y, vali_data, vali_y = preproc.load_data()
    test_data = preproc.load_test()
    # use this while using tfidf
    train_data, vali_data, test_data = preproc.load_text()
    for i in range(bagging_times):
        print("bagging: ", i)
        bagging_x = []
        bagging_y = []
        for j in range(len(train_y)):
            id = int(random.random() * len(train_y))
            bagging_x.append(train_data[id])
            bagging_y.append(train_y[id])
        classifier(bagging_x, bagging_y, vali_data, vali_y, test_data, str(i))


def train_ada_boost(method):
    classifier = ada_boost_map[method]
    train_data, train_y, vali_data, vali_y = preproc.load_data()
    test_set = preproc.load_test()
    weights = np.array([1.0 / len(train_data)] * len(train_data))
    for i in range(ada_times):
        indices = np.random.choice(len(train_data), size=len(train_data), p=weights)
        data = np.array(train_data)[indices]
        y = np.array(train_y)[indices]
        weights, sigma = classifier(data, y, vali_data, vali_y, test_set, id=str(i), weights=weights,
                                               raw_data=train_data, raw_y=train_y)
        print("Ada_Boost: ", i, sigma)
        if sigma > 0.5:
            break


if __name__ == "__main__":
    train_bagging("svm")
    # train_ada_boost("dtree")
