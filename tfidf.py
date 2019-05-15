from preproc import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def gettfidf(train_list, vali_list, test_list):
    text = train_list + test_list

    count = CountVectorizer()
    counts = count.fit_transform(text)

    count_v1 = CountVectorizer(vocabulary=count.vocabulary_)
    counts_train = count_v1.fit_transform(train_list)
    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit_transform(counts_train)

    count_v2 = CountVectorizer(vocabulary=count.vocabulary_)
    counts_vali = count_v2.fit_transform(vali_list)
    tfidftransformer = TfidfTransformer()
    vali_data = tfidftransformer.fit_transform(counts_vali)

    count_v3 = CountVectorizer(vocabulary=count.vocabulary_)
    counts_test = count_v3.fit_transform(test_list)
    tfidftransformer = TfidfTransformer()
    test_data = tfidftransformer.fit_transform(counts_test)

    return train_data, vali_data, test_data


if __name__ == "__main__":
    x_train, y_train, x_vali, y_vali = load_data()
    text = []
    for i in x_train:
        text.append(i["reviewText"])

    count = CountVectorizer()
    counts = count.fit_transform(text)

    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit_transform(counts)


