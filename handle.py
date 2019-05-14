import os

if __name__ == "__main__":
    path = "./model/bagging/"
    files = os.listdir(path)
    for i in files:
        if i.find("svm_result") > -1:
            temp = i[:11]
            id = int(i[11: -5])
            os.rename(path + i, path + temp + str(id + 10000) + ".json")
