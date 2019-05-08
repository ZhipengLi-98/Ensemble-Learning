from sklearn import tree
from sklearn.externals import joblib
import os
import numpy as np


def dtree_train_bagging(train_data, train_y, test_data, test_y, id=""):
    dtree = tree.DecisionTreeClassifier(class_weight="balanced")
    dtree.fit(train_data, train_y)

    if not os.path.exists("model/bagging"):
        os.mkdir("model/bagging")
    joblib.dump(dtree, "model/bagging/dtree" + str(id) + ".pkl")
    print("DTree Bagging " + str(id) + " Test: ", dtree.score(test_data, test_y))


def dtree_train_ada_boost(train_data, train_y, test_data, test_y, id, weights, raw_data, raw_y):
    dtree = tree.DecisionTreeClassifier(class_weight="balanced")
    dtree.fit(train_data, train_y)

    result = dtree.predict(raw_data)
    sigma = float(np.dot(np.array(result) != np.array(raw_y), weights))
    if not sigma > 0.5:
        beta = sigma / (1.0 - sigma)
        update = []
        for i in range(len(raw_y)):
            if raw_y[i] != result[i]:
                update.append(1)
            else:
                update.append(beta)
        weights = np.multiply(update, weights)
        weights = weights / np.sum(weights)
        if not os.path.exists("model/ada_boost"):
            os.mkdir("model/ada_boost")
        joblib.dump(dtree, "model/ada_boost/dtree" + str(id) + ".pkl")
        with open("model/ada_boost/beta" + str(id) + ".txt", "w") as f:
            f.write(str(beta))
    return weights, sigma


def dtree_classify(data, mode, id):
    model = joblib.load("model/" + mode + "/dtree" + str(id) + ".pkl")
    return model.predict(data)
