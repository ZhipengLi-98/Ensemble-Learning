from utils import *
import preproc
import dtree
import svm
import random
import numpy as np


def dtree_train_bagging():
    train_data, train_y, vali_data, vali_y = preproc.load_data()
    test_set = preproc.load_test()
    for i in range(bagging_times):
        print("bagging: ", i)
        bagging_x = []
        bagging_y = []
        for j in range(len(train_y)):
            id = int(random.random() * len(train_y))
            bagging_x.append(train_data[id])
            bagging_y.append(train_y[id])
        dtree.dtree_bagging(bagging_x, bagging_y, vali_data, vali_y, test_set, str(i))


def svm_train_bagging():
    train_data, train_y, vali_data, vali_y = preproc.load_data()
    test_set = preproc.load_test()
    for i in range(bagging_times):
        print("bagging: ", i)
        bagging_x = []
        bagging_y = []
        for j in range(len(train_y)):
            id = int(random.random() * len(train_y))
            bagging_x.append(train_data[id])
            bagging_y.append(train_y[id])
        svm.svm_bagging(bagging_x, bagging_y, vali_data, vali_y, test_set, str(i))


def dtree_train_ada():
    train_data, train_y, vali_data, vali_y = preproc.load_data()
    test_set = preproc.load_test()
    weights = np.array([1.0 / len(train_data)] * len(train_data))
    for i in range(ada_times):
        indices = np.random.choice(len(train_data), size=len(train_data), p=weights)
        data = np.array(train_data)[indices]
        y = np.array(train_y)[indices]
        weights, sigma = dtree.dtree_ada_boost(data, y, vali_data, vali_y, test_set, id=str(i), weights=weights, raw_data=train_data, raw_y=train_y)
        print("Ada_Boost: ", i, sigma)
        if sigma > 0.5:
            break


if __name__ == "__main__":
    # dtree_train_bagging()
    # svm_train_bagging()
    dtree_train_ada()
