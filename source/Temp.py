# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
from sklearn import svm

import pandas as pd


X_train_data = {'NegativeScore': [0.40,0.00,0.00,0.00], \
             'NeutralScore':[0.00,0.12,0.00,0.00], \
             'PositiveScore': [0.00,0.00,0.19,0.43]}

X_train = pd.DataFrame(X_train_data, columns = \
            ['NegativeScore', 'NeutralScore', 'PositiveScore'])

y_train_data = {'Y': ["Negative", "Others", "Others", "Others"]}
y_train = pd.DataFrame(y_train_data, columns = ['Y'])
y_train = y_train.values.ravel()

X_test_data = {'NegativeScore': [0.25, 0.00, 0.00, 0.00, 0.00], \
               'NeutralScore':  [0.00, 0.28, 0.00, 0.00, 0.00], \
               'PositiveScore': [0.00, 0.00, 0.20, 0.30, 0.11]}
X_test = pd.DataFrame(X_test_data, columns = ['NegativeScore', 'NeutralScore', 'PositiveScore'])

y_test_data = {'Y': ["Negative", "Others", "Others", "Others", "Others"]}
y_test = pd.DataFrame(y_test_data, columns = ['Y'])
y_test = y_test.values.ravel()

logreg = LogisticRegression()
# train the model using X_train_dtm
print("X_train")
print(X_train)
print("y_train")
print(y_train)

print("X_test")
print(X_test)

logreg.fit(X_train, y_train)
# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test)
print("y_pred_class")
print(y_pred_class)
print("y_test")
print(y_test)
# calculate accuracy
print("print the confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred_class))


nb = MultinomialNB()
# train the model using X_train_dtm
nb.fit(X_train, y_train)
print(" Predicting MultinomialNB using model ")
# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)
print(y_pred_class)
print("print the confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred_class))
print("calculate accuracy of class predictions")
print(metrics.accuracy_score(y_test, y_pred_class))


print(" Predicting SVC=linear using model ")
C = 5.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
y_pred_class = svc.predict(X_test)
print(y_pred_class)
print("print the confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred_class))


print(" Predicting SVC=rbf using model ")
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
y_pred_class = rbf_svc.predict(X_test)
print(y_pred_class)
print("print the confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred_class))

print(" Predicting SVC=poly using model ")
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
y_pred_class = poly_svc.predict(X_test)
print(y_pred_class)
print("print the confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred_class))

print(" Predicting svm.LinearSVC using model ")
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
y_pred_class = lin_svc.predict(X_test)
print(y_pred_class)
print("print the confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred_class))
