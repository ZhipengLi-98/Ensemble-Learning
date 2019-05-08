from utils import *
import math

method = {"DTree": dtree_classify,
          "SVM": svm_classify}
choice = "DTree"


def bagging():
    classify = method[choice]
    ans = []
    test_set = []
    for data in test_set:
        label = {0: 0, 1: 0}
        for i in range(bagging_times):
            result = classify()
            label[result[0]] += 1
        ans.append(max(label[0], label[1]))


def ada_boost():
    classify = method[choice]
    ans = []
    test_set = []
    betas = {}
    for data in test_set:
        label = {0: 0, 1: 0}
        for i in range(len(betas)):
            result = classify()
            label[result[0]] += math.log(1.0 / betas[i])
        ans.append(max(label[0], label[1]))


if __name__ == "__main__":
    bagging()