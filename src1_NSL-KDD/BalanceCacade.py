
from imblearn.combine import SMOTEENN
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

import math
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
import sklearn.metrics as metrics

def BalanceCascade(X_train, y_train, X_test, num):
    negnum = y_train[y_train == 0].shape[0]
    posnum = y_train[y_train == 1].shape[0]
    neg_index = np.argwhere(y_train == 0).reshape(negnum, )
    pos_index = np.argwhere(y_train == 1).reshape(posnum, )
    pos_train = X_train[pos_index, :]

    FP = pow(posnum / negnum, 1 / (num - 1))
    classifiers = {};
    thresholds = {};
    test_prob = np.empty((X_test.shape[0], num))
    for i in range(num):
        classifiers[i] = AdaBoostClassifier()
        neg_train_index = np.random.permutation(neg_index)[:posnum]
        neg_train = X_train[neg_train_index, :]
        cur_X_train = np.r_[pos_train, neg_train]
        cur_y_train = np.r_[y_train[pos_index], y_train[neg_train_index]]
        classifiers[i].fit(cur_X_train, cur_y_train)
        predict_result = classifiers[i].predict_proba(X_train[neg_index, :])[:, -1]
        thresholds[i] = np.sort(predict_result)[int(neg_index.shape[0] * (1 - FP))] - 0.5
        neg_index = np.argwhere(predict_result >= (thresholds[i] + 0.5)).reshape(-1, )
        test_prob[:, i] = classifiers[i].predict_proba(X_test)[:, -1] + thresholds[i]
        print("No.{} Classifier Training Finished".format(i))
    test_prob_result = np.average(test_prob, axis=1)
    return test_prob_result

df_train = pd.read_csv("./data/df_train.csv",index_col=0)
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['attack_type'],axis=1), df_train['attack_type'], test_size=0.2, random_state=42)

