from sklearn.model_selection import train_test_split
import csv
import json


def read_csv(name):
    with open(name, "r") as f:
        table = list(csv.reader(f))
        head = table[0][0].split("\t")
        d = table[1:]
        print(len(d))
        data = []
        for i in d:
            temp = ""
            for j in i:
                temp += j
            temp = temp.split("\n")
            for j in temp:
                data.append(j)

        labels = []
        data_set = []
        for line in data:
            temp = ""
            for i in line:
                temp += i
            temp = temp.split("\t")
            if len(temp) == len(head):
                mapping = {}
                for index in range(len(temp) - 1):
                    mapping[head[index]] = temp[index]
                data_set.append(mapping)
                labels.append(temp[-1])
        print(len(data_set))
        print(len(labels))
        x_train, x_vali, y_train, y_vali = train_test_split(data_set, labels, test_size=0.1)
        with open("x_train.json", "w") as f:
            json.dump(x_train, f)
        with open("x_vali.json", "w") as f:
            json.dump(x_vali, f)
        with open("y_train.json", "w") as f:
            json.dump(y_train, f)
        with open("y_vali.json", "w") as f:
            json.dump(y_vali, f)


def load_data():
    with open("x_train.json", "r") as f:
        x_train = json.load(f)
    with open("x_vali.json", "r") as f:
        x_vali = json.load(f)
    with open("y_train.json", "r") as f:
        y_train = json.load(f)
    with open("y_vali.json", "r") as f:
        y_vali = json.load(f)
    return x_train, y_train, x_vali, y_vali


if __name__ == "__main__":
    read_csv("train.csv")


