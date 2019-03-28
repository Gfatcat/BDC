from math import log

import numpy as np
from scipy.sparse import lil_matrix
from sklearn import preprocessing


"""
n:总问题数
tf: term frequency
cf: category frequency  包含t的分类数
df: document frequency 包含t的文档数
tp: 正类中包含t的文档数(A)
tn: 负类中不包含t的文档数(D)
fp: 正类中不包含t的文档数(C)
fn: 负类中包含t的文档数(B)
schemes: 公式详见论文

Positive category: To obtain the most valuable terms in the category under consideration, we temporarily treat this
                   category as the positive category.
Negative category: The categories beyond the positive category.
N:  The number of documents in the whole collection.
tp: The number of documents that contain w in the positive category.
fp: The number of documents that do not contain w in the positive category.
fn: The number of documents that contain w in the negative category.
tn: The number of documents that do not contain w in the negative category.
"""


def tf(train, test):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)
    for text in range(train.shape[0]):
        line = train[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:                
                tf = line[term]
                train_weights[text, term] = tf

    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:                
                tf = line[term]
                test_weights[text, term] = tf

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))

    return train_weights, test_weights


def tf_idf(train, test):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)

    train_text_num = train.shape[0]  # total number of texts in the corpus
    test_text_num = test.shape[0]

    for text in range(train.shape[0]):
        line = train[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term]
                df = np.nonzero(train[:, term])[0].size  # The total number of texts in the corpus containing the word
                # 'term'
                idf = log(train_text_num / df)
                train_weights[text, term] = tf * idf

    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term] 
                df = np.nonzero(test[:, term])[0].size  # The total number of texts in the corpus containing the word 'term'
                idf = log(test_text_num / df)
                test_weights[text, term] = tf * idf

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))

    return train_weights, test_weights


def tf_chi(train, test, term_cat_dict, cat_text_dict):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)

    train_text_num = train.shape[0]

    term_chi = {}
    for term in term_cat_dict:
        term_chi.setdefault(term, {})

    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for term in term_cat_dict:
            tp = float(term_cat_dict[term][cat])
            fn = sum(term_cat_dict[term].values()) - tp
            fp = len(textlist) - tp
            tn = train_text_num - tp - fn - fp
            chi = ((tp * tn - fp * fn) ** 2) / (((tp + fp) * (tn + fn) * (tp + fn) * (fp + tn)))
            term_chi[term][cat] = chi

    # process train corpus
    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for text in textlist:
            line = train[text, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    # tf = line[term]
                    train_weights[text, term] =  term_chi[term][cat]

    # process test corpus
    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                # tf = line[term]
                cat = max(term_chi[term], key=term_chi[term].get)
                test_weights[text, term] =  term_chi[term][cat]

    # train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    # test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))

    train_weights = lil_matrix(train_weights)
    test_weights = lil_matrix(test_weights)

    return train_weights, test_weights


def tf_ig(train, test, term_cat_dict, cat_text_dict):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)

    train_text_num = train.shape[0]

    term_ig = {}
    for term in term_cat_dict:
        term_ig.setdefault(term, {})

    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for term in term_cat_dict:
            tp = float(term_cat_dict[term][cat])
            fn = sum(term_cat_dict[term].values()) - tp
            fp = len(textlist) - tp
            tn = train_text_num - tp - fn - fp
            ig = tp * log((tp) / (tp + fn)) + fn * log((fn) / (tp + fn)) + fp * log(
                (fp) / (fp + tn)) + tn * log((tn) / (fp + tn)) - (tp + fp) * log(
                (tp + fp) / (train_text_num)) - (fn + tn) * log((tn + fn) / (train_text_num))
            term_ig[term][cat] = ig

    # process train corpus
    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for text in textlist:
            line = train[text, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    tf = line[term] 
                    train_weights[text, term] = tf * term_ig[term][cat]

    # process test corpus
    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term]
                cat = max(term_ig[term], key=term_ig[term].get)
                test_weights[text, term] = tf * term_ig[term][cat]

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))

    return train_weights, test_weights


def tf_rf(train, test, term_cat_dict, cat_text_dict):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)

    term_rf = {}
    for term in term_cat_dict:
        term_rf.setdefault(term, {})
    
    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for term in term_cat_dict:
            tp = float(term_cat_dict[term][cat])
            fn = sum(term_cat_dict[term].values()) - tp
            rf = log(2 + (tp + 1) / (fn + 1))
            term_rf[term][cat] = rf

    # process train corpus
    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for text in textlist:
            line = train[text, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    tf = line[term]
                    train_weights[text, term] = tf * term_rf[term][cat]

    # process test corpus
    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term]
                cat = max(term_rf[term], key=term_rf[term].get)
                test_weights[text, term] = tf * term_rf[term][cat]

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))

    return train_weights, test_weights


def tf_OR(train, test, term_cat_dict, cat_que_dict):
    weights = np.zeros(train.shape)
    n = train.shape[0]
    for cat in cat_que_dict:
        quelist = cat_que_dict[cat]
        for que in quelist:
            line = train[que, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    term_frequncy = sum(line)
                    tf = line[term] / term_frequncy
                    tp = float(term_cat_dict[term][cat])
                    fn = sum(term_cat_dict[term].values()) - tp
                    fp = len(quelist) - tp
                    tn = n - tp - fn - fp
                    OR = log((tp * fn + 1) / (fp * tn + 1))
                    tf_OR = tf * OR
                    weights[que, term] = tf_OR
    l2_weights = preprocessing.normalize(weights, norm='l2')
    return_weights = lil_matrix(l2_weights)
    return return_weights


def qf_icf(train, test, term_cat_dict, cat_que_dict):
    c = len(cat_que_dict)
    weights = np.zeros(train.shape)
    for cat in cat_que_dict:
        quelist = cat_que_dict[cat]
        for que in quelist:
            line = train[que, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    tp = term_cat_dict[term][cat]
                    cf = 0
                    for cats in term_cat_dict[term]:
                        if term_cat_dict[term][cats] > 0:
                            cf += 1
                    qf = log(tp + 1)
                    icf = log(c / cf + 1)
                    qf_icf = qf * icf
                    weights[que, term] = qf_icf
    l2_weights = preprocessing.normalize(weights, norm='l2')
    return_weights = lil_matrix(l2_weights)
    return return_weights


def iqf_qf_icf(train, test, term_cat_dict, cat_text_dict):

    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)

    c = len(cat_text_dict)
    n = train.shape[0]

    term_iqf_qf_icf = {}
    for term in term_cat_dict:
        term_iqf_qf_icf.setdefault(term, {})
    
    for cat in cat_text_dict:
        for term in term_cat_dict:
            tp = term_cat_dict[term][cat]
            cf = 0
            for cats in term_cat_dict[term]:
                if term_cat_dict[term][cats] > 0:
                    cf += 1
            fn = sum(term_cat_dict[term].values()) - tp
            iqf = log(n / (tp + fn))
            qf = log(tp + 1)
            icf = log(c / cf + 1)
            iqf_qf_icf = iqf * qf * icf
            term_iqf_qf_icf[term][cat] = iqf_qf_icf

    # process train corpus
    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for text in textlist:
            line = train[text, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    tf = line[term]
                    train_weights[text, term] = tf * term_iqf_qf_icf[term][cat]

    # process test corpus
    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term]
                cat = max(term_iqf_qf_icf[term], key=term_iqf_qf_icf[term].get)
                test_weights[text, term] = tf * term_iqf_qf_icf[term][cat]

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))

    return train_weights, test_weights


def vrf(train, test, term_cat_dict, cat_que_dict):
    weights = np.zeros(train.shape)
    n = train.shape[0]
    for cat in cat_que_dict:
        quelist = cat_que_dict[cat]
        for que in quelist:
            line = train[que, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    tp = float(term_cat_dict[term][cat])
                    fn = sum(term_cat_dict[term].values()) - tp
                    fp = len(quelist) - tp
                    tn = n - tp - fn - fp
                    vrf = log(tp + 1) / log(tn + 1)
                    weights[que, term] = vrf
    l2_weights = preprocessing.normalize(weights, norm='l2')
    return_weights = lil_matrix(l2_weights)
    return return_weights


def tf_eccd(train, test, term_cat_dict, cat_text_dict, term_cat_frequency):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)
    n = train.shape[0]
    cat_num = len(cat_text_dict)
    max_entropy = -cat_num * 1 / cat_num * log((1 / cat_num))

    term_entropy = {}
    term_eccd = {}
    for term in term_cat_dict:
        term_eccd.setdefault(term, {})

    for term in term_cat_frequency:
        term_all_frenquency = sum(term_cat_frequency[term].values())
        if term_all_frenquency == 0:
            term_entropy[term] = max_entropy
            continue
        entropy = 0.0
        for cat in term_cat_frequency[term]:
            tcf = term_cat_frequency[term][cat] / term_all_frenquency  # frequency of term in category ci
            if tcf != 0:
                entropy -= tcf * log(tcf)
        term_entropy[term] = entropy

    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for term in term_cat_dict:
            tp = term_cat_dict[term][cat]
            fn = sum(term_cat_dict[term].values()) - tp
            fp = len(textlist) - tp
            tn = n - tp - fn - fp
            eccd = (tp * tn - fn * fp) / (tp + fp) / (tn + fn) * (
                    max_entropy - term_entropy[term]) / max_entropy
            term_eccd[term][cat]= eccd

    # process train corpus
    for cat in cat_text_dict:
        textlist = cat_text_dict[cat]
        for text in textlist:
            line = train[text, :].tolist()
            for term in range(len(line)):
                if line[term] != 0:
                    tf = line[term] 
                    train_weights[text, term] = tf * term_eccd[term][cat]

    # process test corpus
    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term]
                cat = max(term_eccd[term], key=term_eccd[term].get)
                test_weights[text, term] = tf * term_eccd[term][cat]

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))
    return train_weights, test_weights


def dc(train, test, term_cat_frequency, cat_que_dict):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)
    cat_num = len(cat_que_dict)

    term_entropy = {}
    for term in term_cat_frequency:
        ft = sum(term_cat_frequency[term].values())
        # which doesn't exist in test corpus
        if ft == 0:
            term_entropy[term] = 0.0
            continue
        entropy = 0.0
        for cat in term_cat_frequency[term]:
            tfc = term_cat_frequency[term][cat] / ft  # frequency of term in category ci
            if tfc != 0:
                entropy += tfc * log(tfc)
        term_entropy[term] = entropy  # H(t)

    for text in range(train.shape[0]):
        line = train[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                dc = 1. + (term_entropy[term] / log(cat_num))
                train_weights[text, term] = dc

    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                dc = 1. + (term_entropy[term] / log(cat_num))
                test_weights[text, term] = dc

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))
    return train_weights, test_weights


def tf_dc(train, test, term_cat_frequency, cat_que_dict):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)
    cat_num = len(cat_que_dict)

    term_entropy = {}
    for term in term_cat_frequency:
        ft = sum(term_cat_frequency[term].values())
        # which doesn't exist in test corpus
        if ft == 0:
            term_entropy[term] = 0.0
            continue
        entropy = 0.0
        for cat in term_cat_frequency[term]:
            tfc = term_cat_frequency[term][cat] / ft  # frequency of term in category ci
            if tfc != 0:
                entropy += tfc * log(tfc)
        term_entropy[term] = entropy  # H(t)

    for text in range(train.shape[0]):
        line = train[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term] 
                dc = 1. + (term_entropy[term] / log(cat_num))
                train_weights[text, term] = tf * dc

    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term] 
                dc = 1. + (term_entropy[term] / log(cat_num))
                test_weights[text, term] = tf * dc

    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))
    return train_weights, test_weights


def bdc(train, test, term_cat_frequency, cat_que_dict):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)
    cat_num = len(cat_que_dict)

    term_entropy = {}
    fc = {}
    for cat in cat_que_dict:
        quelist = cat_que_dict[cat]
        fc[cat] = sum(sum(train[quelist, :]))

    for term in term_cat_frequency:
        ft = sum(term_cat_frequency[term].values())  # frequency of term in category ci
        # which doesn't exist in test corpus
        if ft == 0:
            term_entropy[term] = 0.0
            continue
        entropy = 0.0
        ptc_sum = 0.
        for cat in term_cat_frequency[term]:
            ptc = term_cat_frequency[term][cat] / fc[cat]
            ptc_sum += ptc
        for cat in term_cat_frequency[term]:
            ptc = term_cat_frequency[term][cat] / fc[cat]
            if ptc != 0:
                entropy += (ptc / ptc_sum) * log(ptc / ptc_sum)
        term_entropy[term] = entropy  # BH(t)

    for text in range(train.shape[0]):
        line = train[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                bdc = 1. + (term_entropy[term] / log(cat_num))
                train_weights[text, term] = bdc

    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                bdc = 1. + (term_entropy[term] / log(cat_num))
                test_weights[text, term] = bdc


    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))
    return train_weights, test_weights


def tf_bdc(train, test, term_cat_frequency, cat_que_dict):
    train_weights = np.zeros(train.shape)
    test_weights = np.zeros(test.shape)
    cat_num = len(cat_que_dict)

    term_entropy = {}
    fc = {}
    for cat in cat_que_dict:
        quelist = cat_que_dict[cat]
        fc[cat] = sum(sum(train[quelist, :]))

    for term in term_cat_frequency:
        ft = sum(term_cat_frequency[term].values())  # frequency of term in category ci
        # which doesn't exist in test corpus
        if ft == 0:
            term_entropy[term] = 0.0
            continue
        entropy = 0.0
        ptc_sum = 0.
        for cat in term_cat_frequency[term]:
            ptc = term_cat_frequency[term][cat] / fc[cat]
            ptc_sum += ptc
        for cat in term_cat_frequency[term]:
            ptc = term_cat_frequency[term][cat] / fc[cat]
            if ptc != 0:
                entropy += (ptc / ptc_sum) * log(ptc / ptc_sum)
        term_entropy[term] = entropy  # BH(t)

    for text in range(train.shape[0]):
        line = train[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term]
                bdc = 1. + (term_entropy[term] / log(cat_num))
                train_weights[text, term] = tf * bdc

    for text in range(test.shape[0]):
        line = test[text, :].tolist()
        for term in range(len(line)):
            if line[term] != 0:
                tf = line[term]
                bdc = 1. + (term_entropy[term] / log(cat_num))
                test_weights[text, term] = tf * bdc


    train_weights = lil_matrix(preprocessing.normalize(train_weights, norm='l2'))
    test_weights = lil_matrix(preprocessing.normalize(test_weights, norm='l2'))
    return train_weights, test_weights
