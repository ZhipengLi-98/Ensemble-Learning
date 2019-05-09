from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import os
from sklearn.externals import joblib
import json


def svm_bagging(train_data, train_y, vali_data, vali_y, test_data, id=""):
    svm_bagging_vec = DictVectorizer()
    train_x = svm_bagging_vec.fit_transform(train_data)
    vali_x = svm_bagging_vec.transform(vali_data)
    test_x = svm_bagging_vec.transform(test_data)

    svm = LinearSVC(multi_class="ovr", class_weight="balanced", verbose=True)
    svm.fit(train_x, train_y)

    if not os.path.exists("model/bagging"):
        os.mkdir("model/bagging")
    joblib.dump(svm, "model/bagging/svm" + str(id) + ".pkl")
    print("SVM Bagging " + str(id) + " Test: ", svm.score(vali_x, vali_y))

    result = list(svm.predict(test_x))
    with open("model/bagging/svm_result_" + str(id) + ".json", "w") as f:
        json.dump(result, f)
