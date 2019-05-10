from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import os
import json
import numpy as np


def knn_bagging(train_data, train_y, vali_data, vali_y, test_data, id=""):
    knn_bagging_vec = DictVectorizer()
    train_x = knn_bagging_vec.fit_transform(train_data)
    vali_x = knn_bagging_vec.transform(vali_data)
    test_x = knn_bagging_vec.transform(test_data)

    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    if not os.path.exists("model/bagging"):
        os.mkdir("model/bagging")
    joblib.dump(knn, "model/bagging/knn" + str(id) + ".pkl")
    print("KNN Bagging " + str(id) + " Test: ", knn.score(vali_x, vali_y))

    result = list(knn.predict(test_x))
    with open("model/bagging/knn_result_" + str(id) + ".json", "w") as f:
        json.dump(result, f)


def knn_ada_boost(train_data, train_y, vali_data, vali_y, test_data, id, weights, raw_data, raw_y):
    knn_ada_boost_vec = DictVectorizer()
    train_x = knn_ada_boost_vec.fit_transform(train_data)
    raw_x = knn_ada_boost_vec.transform(raw_data)
    vali_x = knn_ada_boost_vec.transform(vali_data)
    test_x = knn_ada_boost_vec.transform(test_data)

    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    result = knn.predict(raw_x)
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
        joblib.dump(knn, "model/ada_boost/knn" + str(id) + ".pkl")
        with open("model/ada_boost/beta" + str(id) + ".txt", "w") as f:
            f.write(str(beta))

        print("KNN Ada_Boost " + str(id) + " Test: ", knn.score(vali_x, vali_y))
        result = list(knn.predict(test_x))
        with open("model/ada_boost/knn_result_" + str(id) + ".json", "w") as f:
            json.dump(result, f)

    return weights, sigma

