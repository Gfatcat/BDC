# encoding: utf-8
import math
import os
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.exceptions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC

from schemes import *

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    trainarray = np.array(trainlist, dtype = str)
    testarray = np.array(testlist, dtype = str)
    return trainarray, testarray


# 创建各种字典和矩阵
def createdic(train, test):
    """
    :param train: train array [n_samples*[question,label]]
    :param test: test array [n_samples*[question,label]]
    :return: cat_train_float: ndarry[n_samples*label], fix string like category name to float like category id.
              cat_test_float: ndarry[n_samples*label], fix string like category name to float like category id.
              term_cat_text_train: dictionary, {term-category-number}, each term link to all categories, each category
                                   link to the number this term occurring this category. train file.
              term_cat_text_test:  same as above. test file.
              cat_text_train_dict: dictionary, {category-question_list}, each category link to a list that all the
                                  question ids belonging to this category. train file.
              cat_text_test_dict: same as above. test file.
              terms_train: ndarry[n_samples*terms_number], Correspondence of terms that appear in each question.
              terms_test: ndarry[n_samples*terms_number], Correspondence of terms that appear in each question.
    """
    corpus_train = train[:, 1]
    cat_train = train[:, 0]
    corpus_test = test[:, 1]
    cat_test = test[:, 0]
    del train
    del test
    term_dict = {}
    cat_dict = {}
    term_cat_text_train = {}
    term_cat_frequency_train = {}
    cat_text_train_dict = {}
    cat_num = 0
    for i in range(len(corpus_train)):
        line = str((corpus_train[i]))
        word = line.split(" ")
        if cat_train[i] not in cat_dict:
            cat_dict[cat_train[i]] = cat_num
            cat_num += 1
            cat_text_train_dict.setdefault(cat_train[i], [])
        if cat_train[i] in cat_dict:
            cat_text_train_dict.setdefault(cat_train[i], []).append(i)

    cat_train_float = np.zeros(cat_train.shape, dtype=float)
    for i in range(cat_train.shape[0]):
        cat_train_float[i] = cat_dict[cat_train[i]]

    vectorizer = CountVectorizer()
    terms_train = vectorizer.fit_transform(corpus_train).toarray()
    term_dict = vectorizer.vocabulary_
    terms_test = vectorizer.transform(corpus_test).toarray()

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

    print("Processing term_cat_text_train")
    for key in term_dict:
        for cat in cat_dict:
            term_cat_text_train.setdefault(term_dict[key], {})[cat] = 0.
            term_cat_frequency_train.setdefault(term_dict[key], {})[cat] = 0.
    for cat in cat_text_train_dict:
        quelist = cat_text_train_dict[cat]
        for que in quelist:
            line = str((corpus_train[que]))
            words = line.split(" ")
            word = list(set(words))  # set 结构去重
            for w in word:
                if w in term_dict:
                    term_cat_text_train[term_dict[w]][cat] += 1.
            for w in words:
                if w in term_dict:
                    term_cat_frequency_train[term_dict[w]][cat] += 1.

    print("dictionaries created")
    return cat_train, cat_test, terms_train, terms_test, term_cat_text_train, term_cat_frequency_train, cat_text_train_dict


# 把词向量模型根据scheme转换为对应的权重
def getdata(train, test , scheme, term_cat_text, cat_text_dict, term_cat_frequency):
    if scheme == "tf":
        train_weights, test_weights = tf(train, test)
    if scheme == "tf_idf":
        train_weights, test_weights = tf_idf(train, test)
    elif scheme == "dc":
        train_weights, test_weights = dc(train, test, term_cat_frequency, cat_text_dict)
    elif scheme == "tf_dc":
        train_weights, test_weights = tf_dc(train, test, term_cat_frequency, cat_text_dict)
    elif scheme == "iqf_qf_icf":
        train_weights, test_weights = iqf_qf_icf(train, test, term_cat_text, cat_text_dict)
    elif scheme == "bdc":
        train_weights, test_weights = bdc(train, test, term_cat_frequency, cat_text_dict)
    elif scheme == "tf_bdc":
        train_weights, test_weights = tf_bdc(train, test, term_cat_frequency, cat_text_dict)
    elif scheme == "tf_ig":
        train_weights, test_weights = tf_ig(train, test, term_cat_text, cat_text_dict)
    elif scheme == "tf_rf":
        train_weights, test_weights = tf_rf(train, test, term_cat_text, cat_text_dict)
    elif scheme == "tf_chi":
        train_weights, test_weights = tf_chi(train, test, term_cat_text, cat_text_dict)
    elif scheme == "tf_eccd":
        train_weights, test_weights = tf_eccd(train, test, term_cat_text, cat_text_dict, term_cat_frequency)
    print(scheme + " data got!")
    return train_weights, test_weights


def experiment(package, scheme="tf_idf", model="knn", knn_neighbour=0, result_dir='./'):
    """
    scheme: 可选 tf_idf, tf_OR, tf_ig, tf_chi, tf_rf, qf_icf, iqf_qf_icf, vrf
    model: 可选 svm, knn
    knn_neighbour:
        为0：测试模式，选用 [1,5,10,15,20,25,30,35] 作为邻居数分别进行训练，文件输出正确率（可plot或导入evaluate程序）
        为n：使用邻居数为n进行训练，文件输出所有预测标签和正确标签，中间以\t分隔
    """
    train_label = package[0]
    test_label = package[1]
    terms_train = package[2]
    terms_test = package[3]
    term_cat_text_train = package[4]
    term_cat_frequency_train = package[5]
    cat_text_train_dict = package[6]
    print("Getting " + scheme + " data ")
    # 读取数据
    print("Processing data Set... ")
    y_trian = train_label
    y_test = test_label
    x_train, x_test = getdata(terms_train,terms_test, scheme, term_cat_text_train, cat_text_train_dict, term_cat_frequency_train)

    # svm训练
    if model.lower() == "svm" or "all":
        print("Training " + "svm" + " with " + scheme)
        svmclf = SVC(kernel='linear')
        svmclf.fit(x_train, y_trian)
        y_pre = svmclf.predict(x_test)
        svmresultfile = open(result_dir + "svm" + "_" + scheme + ".txt", "w")
        macro_f1 = f1_score(y_test, y_pre, average='macro')
        micro_f1 = f1_score(y_test, y_pre, average='micro')
        # for i in range(len(y_pre)):
        #     resultfile.write(str(y_test[i]) + " " + str(y_pre[i]) + "\n")
        svmresultfile.write(str(micro_f1) + "\t" + str(macro_f1) + "\n")
        print(str(micro_f1) + "\t" + str(macro_f1) + "\n")
        svmresultfile.close()
        print("svm" + " with " + scheme + " finish!")

    if model.lower() == "knn" or "all":
        print("Training " + "knn" + " with " + scheme)
        if knn_neighbour != 0:
            knnclf = KNN(n_neighbors=knn_neighbour, weights='uniform')
            knnclf.fit(x_train, y_trian)
            y_pre = knnclf.predict(x_test)
            knnresultfile = open(result_dir + "knn" + "_" + scheme + ".txt", "w")
            macro_f1 = f1_score(y_test, y_pre, average='macro')
            micro_f1 = f1_score(y_test, y_pre, average='micro')
            for i in range(len(y_pre)):
                knnresultfile.write(str(y_test[i]) + " " + str(y_pre[i]) + "\n")
            knnresultfile.write(str(micro_f1) + "\t" + str(macro_f1) + "\n")
            print(str(micro_f1) + "\t" + str(macro_f1) + "\n")
            knnresultfile.close()
        else:
            X = [i for i in range(0, 76, 5)]
            X[0] = 1
            knnresultfile = open(result_dir + "knn" + "_" + scheme + ".txt", "w")
            for i in X:
                knnclf = KNN(n_neighbors=i, weights='uniform')
                knnclf.fit(x_train, y_trian)
                y_pre = knnclf.predict(x_test)
                macro_f1 = f1_score(y_test, y_pre, average='macro')
                micro_f1 = f1_score(y_test, y_pre, average='micro')
                str_nei = str(i)
                knnresultfile.write(str_nei + "\t" + str(micro_f1) + "\t" + str(macro_f1) + "\n")
                print(str_nei + "\t" + str(micro_f1) + "\t" + str(macro_f1) + "\n")
            knnresultfile.close()
        print("knn" + " with " + scheme + " finish!")
    # print(model + " with " + scheme + " finish!")


if __name__ == "__main__":
    train, test = loaddata("./reuters/r8-train-all-terms.txt", "./reuters/r8-test-all-terms.txt")
    # train, test = loaddata("./20news/20ng-train-no-stop.txt", "./20news/20ng-test-no-stop.txt")
    package = ()
    package = createdic(train, test)

    schemes = ["tf", "tf_idf", "tf_ig", "tf_chi", "tf_rf", "iqf_qf_icf", "tf_eccd", "dc", "bdc", "tf_dc", "tf_bdc"]

    # import time

    # now = str(time.strftime('_%m%d_%H%M', time.localtime(time.time())))
    # path = ("./results" + now + "/")

    # if not os.path.exists(path):
    #     os.mkdir(path)

    # for sche in schemes:
    #     experiment(package, scheme=sche, model="all", result_dir=path)

    experiment(package, scheme="tf_chi", model="all", result_dir="./test/")
    # experiment(package, scheme="tf_idf", model="all", result_dir="./test/")
