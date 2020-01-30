import numpy as np
import pandas as pd
from time import time
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, hinge_loss
from sklearn.model_selection import train_test_split
from featureExtraction import feature_extraction
# from models import Classifiers
from scipy.sparse import hstack
#
# import os
# os.chdir('/content/gdrive/My Drive/EmotionDetection')

# Read cleaned training data
train_label = pd.read_csv("data/train_train_labels.csv")
test_label = pd.read_csv("data/train_test_labels.csv")
tr_tr_y = train_label['label']
tr_tst_y = test_label['label']

def change_label(y):
    y_happy = ['happy' if x == 'happy' else 'not' for x in y]
    y_sad = ['sad' if x == 'sad' else 'not' for x in y]
    y_angry = ['angry' if x == 'angry' else 'not' for x in y]
    y_other = ['others' if x == 'others' else 'not' for x in y]
    return y_happy, y_sad, y_angry, y_other

y_happy_train_train, y_sad_train_train, y_angry_train_train, y_other_train_train = change_label(tr_tr_y)
y_happy_train_test, y_sad_train_test, y_angry_train_test, y_other_train_test = change_label(tr_tst_y)

import joblib
train_vecs = joblib.load('data/train_train_vecs.pkl')
test_vecs = joblib.load('data/train_test_vecs.pkl')


feature_list = ['unigram', 'bigram', 'trigram', 'emoji', 'emotic', 'special_char', 'uppercases', 'marks', 'wordcount']
train_vecs = list(train_vecs)
test_vecs = list(test_vecs)

def feature_selection_helper(features, test_feature, y, y_test):
    #  To do: grid search for C maybe
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    feature_order = []
    n = len(features)
    loss = 100000
    lossnew = 10000
    while lossnew < loss:
        min_loss = float('inf')
        min_var = -1
        for j in range(n):
            if feature_list[j] in feature_order:
                continue
            if len(feature_order) == 0:
                vec_temp = features[j]
                vect_temp = test_feature[j]
            else:
                vec_temp = hstack((vec, features[j]))
                vect_temp = hstack((vect, test_feature[j]))
            clf = clf.fit(vec_temp, y)
            # y_predict = clf.predict(vect_temp)
            pred_decision = clf.decision_function(vect_temp)
            loss = hinge_loss(y_test, pred_decision)

            if loss < min_loss:
                # print(score)
                min_loss = loss
                min_var = j
        if len(feature_order) == 0:
            vec = features[min_var]
            vect = test_feature[min_var]
        else:
            vec = hstack((vec, features[min_var]))
            vect = hstack((vect, test_feature[min_var]))
        loss = lossnew
        lossnew = min_loss
        feature_order.append(min_var)

    return feature_order


h_order = feature_selection_helper(train_vecs, test_vecs, y_happy_train_train, y_happy_train_test)

print("happy:", h_order)

s_order = feature_selection_helper(train_vecs, test_vecs, y_sad_train_train, y_sad_train_test)

print("sad:", s_order)

a_order = feature_selection_helper(train_vecs, test_vecs, y_angry_train_train, y_angry_train_test)

print("angry:", a_order)

o_order = feature_selection_helper(train_vecs, test_vecs, y_other_train_train, y_other_train_test)

print("other:", o_order)

all_order = feature_selection_helper(train_vecs, test_vecs, tr_tr_y, tr_tst_y)

print("all:",all_order)




# save the selected orders to files
joblib.dump(h_order, 'data/h_order.pkl')
joblib.dump(a_order, 'data/a_order.pkl')
joblib.dump(o_order, 'data/o_order.pkl')
joblib.dump(all_order, 'data/all_order.pkl')

