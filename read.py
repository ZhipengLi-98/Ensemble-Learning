import pandas as pd

if __name__ == "__main__":
    train_set = pd.read_csv("train.csv", sep="\t", nrows=10)
    # print(train_set.keys())
    with open("train.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            ans = line.split("\t")[0]
