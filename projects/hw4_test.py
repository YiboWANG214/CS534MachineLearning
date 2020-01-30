import numpy as np
import matplotlib.pyplot as plt

from hw4 import GreedyKNN

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score, roc_curve, auc


data = load_breast_cancer()
X = data.data
y = data.target
x = int(np.round(len(X) * 0.7))
TrainX = X[:x, :]
TestX = X[x:, :]
Trainy = y[:x]
Testy = y[x:]


Gknn = GreedyKNN()
# Test your code with the breast cancner dataset with/without standardizing the features
S_TrainX = scale(TrainX)
mean = np.mean(TrainX, axis=0)
std = np.std(TrainX, axis=0)
S_TestX = (TestX - mean) / std

# Split the data into training and test sets, and run the get_feature_order on the training set
feature_order1 = Gknn.get_feature_order(TrainX, Trainy)
print("feature order without standardization:", feature_order1)
feature_order2 = Gknn.get_feature_order(S_TrainX, Trainy)
print("feature order with standardization", feature_order2)

# Plot the test set AUROC performance across various kNN models built based on the first t features of feature_lst
k1 = len(feature_order1)
auroc1 = []
for t in range(k1):
    X1 = TestX[:, feature_order1[0:t+1]]
    TrainX1 = TrainX[:, feature_order1[0:t+1]]
    y_hat = Gknn.kNNpredict(X1, TrainX1, Trainy, k=5)
    auroc1.append(roc_auc_score(Testy, y_hat))
    fpr, tpr, threshold = roc_curve(Testy, y_hat)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

k2 = len(feature_order2)
auroc2 = []
for t in range(k2):
    X2 = S_TestX[:, feature_order2[0:t+1]]
    TrainX2 = S_TrainX[:, feature_order2[0:t+1]]
    y_hat = Gknn.kNNpredict(X2, TrainX2, Trainy, k=5)
    auroc2.append(roc_auc_score(Testy, y_hat))
    fpr2, tpr2, threshold2 = roc_curve(Testy, y_hat)
    roc_auc2 = auc(fpr2, tpr2)

    plt.figure()
    plt.plot(fpr2, tpr2, color='orange', label='ROC curve (area = %0.2f)' % roc_auc2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
print("auroc of test set is: ", auroc1)
print("auroc of standardized test set is: ", auroc2)

# Change k to different values and interpret the results in terms of the bias-variance tradeoff
auroc1 = []
auroc2 = []
gap1 = []
for k in range(1, 11):
    y_hat3 = Gknn.kNNpredict(TrainX, TrainX, Trainy, k)
    y_hat4 = Gknn.kNNpredict(TestX, TrainX, Trainy, k)
    a = roc_auc_score(Trainy, y_hat3)
    b = roc_auc_score(Testy, y_hat4)
    auroc1.append(a)
    auroc2.append(b)
    gap1.append(a-b)
# print(auroc1)
# print(auroc2)
print("the gap of auroc between training set and test set:", gap1)

auroc3 = []
auroc4 = []
gap2 = []
for k in range(1, 11):
    y_hat3 = Gknn.kNNpredict(S_TrainX, S_TrainX, Trainy, k)
    y_hat4 = Gknn.kNNpredict(S_TestX, S_TrainX, Trainy, k)
    c = roc_auc_score(Trainy, y_hat3)
    d = roc_auc_score(Testy, y_hat4)
    auroc3.append(c)
    auroc4.append(d)
    gap2.append(c-d)
print("the gap of auroc between standardized training set and standardized test set:", gap2)

# interpret: The gap gets smaller first because when k gets larger, variance gets smaller.
