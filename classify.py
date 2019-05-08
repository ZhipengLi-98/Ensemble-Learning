from utils import *
import math
import preproc
import json


def bagging():
    ans = []
    label = {}
    temp = {}
    for j in range(bagging_times):
        with open("model/bagging/dtree_result_" + str(j) +".json") as f:
            result = list(json.load(f))
            for i in range(len(result)):
                if i not in label.keys():
                    label[i] = int(result[i])
                else:
                    label[i] += int(result[i])
            temp[j] = result
    for i in label.keys():
        ans.append(float(label[i] / bagging_times))
    return ans


def ada_boost():
    ans = []
    test_set = []
    betas = {}
    for data in test_set:
        label = {0: 0, 1: 0}
        for i in range(len(betas)):
            result = classify()
            label[result[0]] += math.log(1.0 / betas[i])
        ans.append(max(label[0], label[1]))


def write(result):
    ans = [str(i) + ',' + str(result[i]) + '\n' for i in range(len(result))]
    ans = ['Id,Predicted\n'] + ans
    with open('result.csv', 'w') as f:
        f.writelines(ans)


if __name__ == "__main__":
    write(bagging())
