from utils import *
import preproc
import dtree
import random


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


if __name__ == "__main__":
    dtree_train_bagging()