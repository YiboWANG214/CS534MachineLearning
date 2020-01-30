import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn import svm
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, hinge_loss

fileH = "modelH1"
fileS = "modelS1"
fileA = "modelA1"
fileO = "modelO1"
fileAll = "modelAll1"

def save_model(classifier, feature, y, filename):
    classifier = classifier.fit(feature, y)
    pickle.dump(classifier, open(filename, 'wb'))


label_map = {0:'happy', 1:'sad', 2:'angry', 3:'others'}

def change_label(y):
    y_happy = ['happy' if x == 'happy' else 'not' for x in y]
    y_sad = ['sad' if x == 'sad' else 'not' for x in y]
    y_angry = ['angry' if x == 'angry' else 'not' for x in y]
    y_other = ['others' if x == 'others' else 'not' for x in y]
    return y_happy, y_sad, y_angry, y_other



#  Read feature vectors
train_vecs = joblib.load('data/train_train_vecs.pkl')
test_vecs = joblib.load('data/train_test_vecs.pkl')

train_label = pd.read_csv("data/train_train_labels.csv")
test_label = pd.read_csv("data/train_test_labels.csv")
tr_tr_y = train_label['label']
tr_tst_y = test_label['label']


y_happy_train, y_sad_train, y_angry_train, y_other_train = change_label(tr_tr_y)

#features = [ '0: unigram', '1:bigram', '2:traigram', '3:emoji', '4:emotic',
# '5:special_char', '6:uppercases', '7:marks', '8:wordcount']

train_vecs = list(train_vecs)
test_vecs = list(test_vecs)


# Read selected feature orders
h_order = joblib.load('data/h_order.pkl')
s_order = joblib.load('data/s_order.pkl')
a_order = joblib.load('data/a_order.pkl')
o_order = joblib.load('data/o_order.pkl')

# h_order = [0,1,3,4,7,8]
# s_order = [0,1,3,4,6]
# a_order = [0,1,2,3,6,7]
# o_order = [0,1,3,4,6]
print("building models")
feature_train_H = None
feature_test_H = None
for i in h_order:
    if feature_train_H is None:
        feature_train_H = train_vecs[i]
        feature_test_H = test_vecs[i]
    else:
        feature_train_H = hstack((feature_train_H, train_vecs[i]))
        feature_test_H = hstack((feature_test_H, test_vecs[i]))

classifierH = svm.SVC(kernel='linear', C=1, probability=True)
classifierH = classifierH.fit(feature_train_H, y_happy_train)
# save_model(classifierH, feature_train_H, y_happy_train, fileH)


feature_train_S = None
feature_test_S = None
for i in s_order:
    if feature_train_S is None:
        feature_train_S = train_vecs[i]
        feature_test_S = test_vecs[i]
    else:
        feature_train_S = hstack((feature_train_S, train_vecs[i]))
        feature_test_S = hstack((feature_test_S, test_vecs[i]))

classifierS = svm.SVC(kernel='linear', C=1, probability=True)
classifierS = classifierS.fit(feature_train_S, y_sad_train)

feature_train_A = None
feature_test_A = None
for i in a_order:
    if feature_train_A is None:
        feature_train_A = train_vecs[i]
        feature_test_A = test_vecs[i]
    else:
        feature_train_A = hstack((feature_train_A, train_vecs[i]))
        feature_test_A = hstack((feature_test_A, test_vecs[i]))
classifierA = svm.SVC(kernel='linear', C=1, probability=True)
classifierA = classifierA.fit(feature_train_A, y_angry_train)

feature_train_O = None
feature_test_O = None
for i in o_order:
    if feature_train_O is None:
        feature_train_O = train_vecs[i]
        feature_test_O = test_vecs[i]
    else:
        feature_train_O = hstack((feature_train_O, train_vecs[i]))
        feature_test_O = hstack((feature_test_O, test_vecs[i]))
classifierO = svm.SVC(kernel='linear', C=1, probability=True)
classifierO = classifierO.fit(feature_train_O, y_other_train)

print("finished build models")


counts = tr_tr_y.value_counts()
h_ratio = counts['happy'] / tr_tr_y.shape[0]
s_ratio = counts['sad'] / tr_tr_y.shape[0]
a_ratio = counts['angry'] / tr_tr_y.shape[0]
o_ratio = counts['others'] / tr_tr_y.shape[0]

label_map = {0:'happy', 1:'sad', 2:'angry', 3:'others'}

def predict(classifierH, classifierS, classifierA, classifierO, featureH, featureS, featureA, featureO):
    y_predictH = classifierH.predict_proba(featureH)[:,0]
    y_predictS = classifierS.predict_proba(featureS)[:,1]
    y_predictA = classifierA.predict_proba(featureA)[:,0]
    y_predictO = classifierO.predict_proba(featureO)[:,1]
    y_predict = []
    for i in range(len(y_predictA)):
        list = [y_predictH[i],y_predictS[i],y_predictA[i],y_predictO[i]]
        label = label_map[list.index(max(list))]
        y_predict.append(label)

    df = pd.DataFrame({'happyP': y_predictH, 'sadP': y_predictS, 'angryP': y_predictA, 'ohtersP': y_predictO, 'true_label':tr_tst_y, 'predict_label':y_predict})
    df.to_csv("predictP.csv")
    return y_predict


y_predict = predict(classifierH, classifierS, classifierA, classifierO, feature_test_H, feature_test_S, feature_test_A, feature_test_O)

report = classification_report(tr_tst_y, y_predict)
print(report)
print(confusion_matrix(tr_tst_y, y_predict))



