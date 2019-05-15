from sklearn import tree
from sklearn.externals import joblib
import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import json


def dtree_bagging(train_data, train_y, vali_data, vali_y, test_data, id=""):
    dtree_bagging_vec = DictVectorizer()
    train_x = dtree_bagging_vec.fit_transform(train_data)
    vali_x = dtree_bagging_vec.transform(vali_data)
    test_x = dtree_bagging_vec.transform(test_data)

    dtree = tree.DecisionTreeClassifier(class_weight="balanced", max_depth=1000)
    dtree.fit(train_x, train_y)

    if not os.path.exists("model/bagging"):
        os.mkdir("model/bagging")
    print("DTree Bagging " + str(id) + " Test: ", dtree.score(vali_x, vali_y))

    result = list(dtree.predict(test_x))
    with open("model/bagging/dtree_result_" + str(id) + ".json", "w") as f:
        json.dump(result, f)


def dtree_ada_boost(train_data, train_y, vali_data, vali_y, test_data, id, weights, raw_data, raw_y):
    dtree_ada_boost_vec = DictVectorizer()
    train_x = dtree_ada_boost_vec.fit_transform(train_data)
    raw_x = dtree_ada_boost_vec.transform(raw_data)
    vali_x = dtree_ada_boost_vec.transform(vali_data)
    test_x = dtree_ada_boost_vec.transform(test_data)

    dtree = tree.DecisionTreeClassifier(class_weight="balanced", max_depth=500)
    dtree.fit(train_x, train_y)

    result = dtree.predict(raw_x)
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
        with open("model/ada_boost/dtree_beta" + str(id) + ".txt", "w") as f:
            f.write(str(beta))

        print("DTree Ada_Boost " + str(id) + " Test: ", dtree.score(vali_x, vali_y))
        result = list(dtree.predict(test_x))
        with open("model/ada_boost/dtree_result_" + str(id) + ".json", "w") as f:
            json.dump(result, f)

    return weights, sigma

