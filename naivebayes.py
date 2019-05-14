from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.externals import joblib
import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import json


def nb_bagging(train_data, train_y, vali_data, vali_y, test_data, id=""):
    nb_bagging_vec = DictVectorizer()
    train_x = nb_bagging_vec.fit_transform(train_data)
    vali_x = nb_bagging_vec.transform(vali_data)
    test_x = nb_bagging_vec.transform(test_data)

    # nb = BernoulliNB()
    nb = MultinomialNB()
    nb.fit(train_x, train_y)

    if not os.path.exists("model/bagging"):
        os.mkdir("model/bagging")
    joblib.dump(nb, "model/bagging/nb" + str(id) + ".pkl")
    print("NB Bagging " + str(id) + " Test: ", nb.score(vali_x, vali_y))

    result = list(nb.predict(test_x))
    with open("model/bagging/nb_result_" + str(id) + ".json", "w") as f:
        json.dump(result, f)


def nb_ada_boost(train_data, train_y, vali_data, vali_y, test_data, id, weights, raw_data, raw_y):
    nb_ada_boost_vec = DictVectorizer()
    train_x = nb_ada_boost_vec.fit_transform(train_data)
    raw_x = nb_ada_boost_vec.transform(raw_data)
    vali_x = nb_ada_boost_vec.transform(vali_data)
    test_x = nb_ada_boost_vec.transform(test_data)

    nb = BernoulliNB()
    # nb = MultinomialNB()
    nb.fit(train_x, train_y)

    result = nb.predict(raw_x)
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
        joblib.dump(nb, "model/ada_boost/nb" + str(id) + ".pkl")
        with open("model/ada_boost/nb_beta" + str(id) + ".txt", "w") as f:
            f.write(str(beta))

        print("NB Ada_Boost " + str(id) + " Test: ", nb.score(vali_x, vali_y))
        result = list(nb.predict(test_x))
        with open("model/ada_boost/nb_result_" + str(id) + ".json", "w") as f:
            json.dump(result, f)

    return weights, sigma

