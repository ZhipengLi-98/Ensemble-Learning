from utils import *
import math
import json
import os


def bagging(method):
    ans = []
    label = {}
    temp = {}
    for j in range(bagging_times):
        with open("model/bagging/" + method + "_result_" + str(j) +".json") as f:
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


def ada_boost(method):
    ans = []
    temp = []
    label = {}
    betas = {}
    dir = os.listdir("model/ada_boost")
    for file in dir:
        if file.split(".")[-1] == "txt":
            id = file.split(".")[0][-1]
            with open("model/ada_boost/" + file) as f:
                betas[int(id)] = float(f.read())
            with open("model/ada_boost/dtree_result_" + id + ".json") as f:
                label[int(id)] = json.load(f)
    ans = [0 for i in range(len(label[0]))]
    temp = [0 for i in range(len(label[0]))]
    for i in betas.keys():
        beta = betas[i]
        result = label[i]
        for j in range(len(result)):
            ans[j] += float(result[j]) * math.log(1.0 / beta)
            temp[j] += math.log(1.0 / beta)
    for i in range(len(ans)):
        ans[i] = ans[i] / temp[i]
    return ans


def write(result):
    ans = [str(i) + ',' + str(result[i]) + '\n' for i in range(len(result))]
    ans = ['Id,Predicted\n'] + ans
    with open('result.csv', 'w') as f:
        f.writelines(ans)


if __name__ == "__main__":
    # write(bagging("dtree"))
    write(ada_boost("dtree"))
