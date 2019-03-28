# encoding: utf-8
import re
import os
import math
import numpy as np
import pandas as pd
from schemes import *
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy.sparse import csr_matrix,lil_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def createdic(train, test):
    """
    :param train: train array [n_samples*[question,label]]
    :param test: test array [n_samples*[question,label]]
    :return: cat_train_float: ndarry[n_samples*label], fix string like category name to float like category id.
              cat_test_float: ndarry[n_samples*label], fix string like category name to float like category id.
              term_cat_train_dict: dictionary, {term-category-number}, each term link to all categories, each category
                                   link to the number this term occurring this category. train file.
              term_cat_test_dict:  same as above. test file.
              cat_que_train_dict: dictionary, {category-question_list}, each category link to a list that all the
                                  question ids belonging to this category. train file.
              cat_que_test_dict: same as above. test file.
              terms_train: ndarry[n_samples*terms_number], Correspondence of terms that appear in each question.
              terms_test: ndarry[n_samples*terms_number], Correspondence of terms that appear in each question.
    """
    subject_train = train[:, 1]
    cat_train = train[:, 0]
    subject_test = test[:, 1]
    cat_test = test[:, 0]
    del train
    del test
    term_dict = {}
    cat_dict = {}
    term_cat_train_dict = {}
    term_cat_test_dict = {}
    cat_que_train_dict = {}
    cat_que_test_dict = {}
    cat_num = 0
    term_num = 0
    for i in range(len(subject_train)):
        line = str((subject_train[i]))
        word = line.split(" ")
        for w in word:
            if w not in term_dict:
                term_dict[w] = term_num
                term_num += 1
        if cat_train[i] not in cat_dict:
            cat_dict[cat_train[i]] = cat_num
            cat_num += 1
            cat_que_train_dict.setdefault(cat_train[i], [])
            cat_que_test_dict.setdefault(cat_test[i], [])
        if cat_train[i] in cat_dict:
            cat_que_train_dict.setdefault(cat_train[i], []).append(i)
    for i in range(len(subject_test)):
        if cat_test[i] in cat_dict:
            cat_que_test_dict.setdefault(cat_test[i], []).append(i)

    cat_train_float = np.zeros(cat_train.shape, dtype=float)
    for i in range(cat_train.shape[0]):
        cat_train_float[i] = cat_dict[cat_train[i]]
    cat_test_float = np.zeros(cat_test.shape, dtype=float)
    for i in range(cat_test.shape[0]):
        cat_test_float[i] = cat_dict[cat_test[i]]

    vectorizer = CountVectorizer()
    terms_train = vectorizer.fit_transform(subject_train).toarray()
    term_dict = vectorizer.vocabulary_
    terms_test = vectorizer.transform(subject_test).toarray()

    # print("Processing train terms")
    # terms_train = np.zeros((que_num, term_num))
    # for i in range(len(subject_train)):
    #     line = str((subject_train[i]))
    #     word = line.split(" ")
    #     for w in word:
    #         if w in term_dict:
    #             terms_train[i, term_dict[w]] += 1
    #
    # print("Processing test terms")
    # terms_test = np.zeros((len(subject_test), term_num))
    # for i in range(len(subject_test)):
    #     line = str((subject_test[i]))
    #     word = line.split(" ")
    #     for w in word:
    #         if w in term_dict:
    #             terms_test[i, term_dict[w]] += 1

    print("Processing term_cat_train_dict")
    for key in term_dict:
        for cat in cat_dict:
            term_cat_train_dict.setdefault(term_dict[key], {})[cat] = 0
    for cat in cat_que_train_dict:
        quelist = cat_que_train_dict[cat]
        for que in quelist:
            line = str((subject_train[que]))
            word = line.split(" ")
            word = list(set(word))
            for w in word:
                if w in term_dict:
                    term_cat_train_dict[term_dict[w]][cat] += 1

    print("Processing term_cat_test_dict")
    for key in term_dict:
        for cat in cat_dict:
            term_cat_test_dict.setdefault(term_dict[key], {})[cat] = 0
    for cat in cat_que_test_dict:
        quelist = cat_que_test_dict[cat]
        for que in quelist:
            line = str((subject_test[que]))
            word = line.split(" ")
            word = list(set(word))
            for w in word:
                if w in term_dict:
                    term_cat_test_dict[term_dict[w]][cat] += 1

    print("dictionaries created")
    return cat_train, cat_test, terms_train, terms_test, term_cat_train_dict, term_cat_test_dict,cat_que_train_dict,cat_que_test_dict


def loaddata(trainfile, testfile):
    """
    :param trainfile: training data txt format file
    :param testfile: test data txt format file
    :return: train array [n_samples*[question,label]]
             test array [n_samples*[question,label]]
    """
    # train = np.array(pd.read_csv(trainfile, encoding="utf-8", header=None))
    # test = np.array(pd.read_csv(testfile, encoding="utf-8", header=None))
    trainlist = [w.strip().split('\t') for w in open(trainfile, 'r', encoding='utf-8').readlines()]
    testlist = [w.strip().split('\t') for w in open(testfile, 'r', encoding='utf-8').readlines()]
    trainarray = np.array(trainlist)
    testarray = np.array(testlist)
    return trainarray, testarray


if __name__ == "__main__":
    import time
    now = str(time.strftime('%m%d_%H%M',time.localtime(time.time())))
    print(now)


    train, test = loaddata("./reuters/r8-train-all-terms.txt", "./reuters/r8-test-all-terms.txt")
    train_corpus = train[:, 1]
    train_label = train[:, 0]
    test_corpus = test[:, 1]
    test_label = test[:, 0]
    

    # corpus = [
    #     'This is the first document.',
    #     'This is the second second document.',
    #     'And the third one.',
    #     'Is this the first document?',
    # ]

    vectorizer = TfidfVectorizer(use_idf=False,norm="l2")
    # vector = vectorizer.fit_transform(corpus).toarray()
    train_x = vectorizer.fit_transform(train_corpus).toarray()
    test_x = vectorizer.transform(test_corpus).toarray()
    
    train_x = lil_matrix(train_x)
    test_x = lil_matrix(test_x)
    # print(vector)
    # print()

    # countvectorizer = CountVectorizer()
    # terms_train = countvectorizer.fit_transform(corpus).toarray()
    # print(terms_train)
    # print()
    # weights = np.zeros(terms_train.shape)
    # for que in range(terms_train.shape[0]):
    #     line = terms_train[que, :].tolist()
    #     for term in range(len(line)):
    #         if line[term] != 0:
    #             term_frequncy = sum(line)
    #             tf = line[term] / term_frequncy
    #             weights[que, term] = tf

    # print(weights)
    # print()

    # std = MinMaxScaler()
    # weights_std = std.fit_transform(weights)
    # print(weights_std)


    # package=()
    # package = createdic(train, test)

    # train_label = package[0]
    # test_label = package[1]
    # terms_train = package[2]
    # terms_test = package[3]
    # term_cat_train_dict = package[4]
    # term_cat_test_dict = package[5]
    # cat_que_train_dict = package[6]
    # cat_que_test_dict = package[7]

    # #读取数据
    # print("Processing Training Set... ")
    # y_trian = train_label
    # x_train = tf(terms_train, term_cat_train_dict, cat_que_train_dict)
    # # print("Finish! ")
    # print("Processing Test Set... ")
    # y_test = test_label
    # x_test = tf(terms_test, term_cat_test_dict, cat_que_test_dict)


    # print((x_train==train_x).all())
    # print((x_test==test_x).all())

    clf = SVC(kernel='linear')
    clf.fit(train_x, train_label)
    y_pre = clf.predict(test_x)
    macro_f1 = f1_score(test_label, y_pre, average='macro')
    micro_f1 = f1_score(test_label, y_pre, average='micro')
    print(macro_f1)
    print(micro_f1)
    X = [i for i in range(0, 30, 5)]
    X[0] = 1
    for i in X:
        clf = KNN(n_neighbors=i, weights='uniform')
        clf.fit(train_x, train_label)
        y_pre = clf.predict(test_x)
        macro_f1 = f1_score(test_label, y_pre, average='macro')
        micro_f1 = f1_score(test_label, y_pre, average='micro')
        print(i)
        print(macro_f1)
        print(micro_f1)