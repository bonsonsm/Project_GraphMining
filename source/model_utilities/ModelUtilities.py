# -*- coding: utf-8 -*-
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm, datasets

import pandas as pd

from nltk.corpus import stopwords

class ModelFunctions:
    
    def compute_MultiNomial_NB(self, X_train_dtm, y_train, X_test_dtm, y_test, X_test):
        print("##########Computing MultiNomial NB##########")
        nb = MultinomialNB()
        # train the model using X_train_dtm
        nb.fit(X_train_dtm, y_train)
        print(" Predicting using model ")
        # make class predictions for X_test_dtm
        y_pred_class = nb.predict(X_test_dtm)
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        print("calculate accuracy of class predictions")
        print(metrics.accuracy_score(y_test, y_pred_class))
        # calculate predicted probabilities for X_test_dtm (poorly calibrated)
        y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
        #print(y_pred_prob)
        print("roc auc")
        #calculate AUC
        print(metrics.roc_auc_score(y_test, y_pred_prob))
        return nb

    def compute_LogisticRegression(self, X_train_dtm, y_train, X_test_dtm, y_test, X_test):
        print("##########Computing Logistic regression##########")
        logreg = LogisticRegression()
        # train the model using X_train_dtm
        logreg.fit(X_train_dtm, y_train)
        # make class predictions for X_test_dtm
        y_pred_class = logreg.predict(X_test_dtm)
        # calculate predicted probabilities for X_test_dtm (well calibrated)
        y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
        #print(y_pred_prob)
        # calculate accuracy
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        print("calculate accuracy of class predictions")
        print(metrics.accuracy_score(y_test, y_pred_class))
        print("roc auc")
        # calculate AUC
        print(metrics.roc_auc_score(y_test, y_pred_prob))
        return logreg
        
    def compute_SVM(self, X_train_dtm, y_train, X_test_dtm, y_test, X_test):
        print("########## Computing SVM ##########")
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 5.0  # SVM regularization parameter
        svc = svm.SVC(kernel='linear', C=C).fit(X_train_dtm, y_train)
        y_pred_class = svc.predict(X_test_dtm)
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        print("calculate accuracy of class predictions")
        print(metrics.accuracy_score(y_test, y_pred_class))
        
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train_dtm, y_train)
        y_pred_class = rbf_svc.predict(X_test_dtm)
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        print("calculate accuracy of class predictions")
        print(metrics.accuracy_score(y_test, y_pred_class))
        
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train_dtm, y_train)
        y_pred_class = poly_svc.predict(X_test_dtm)
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        print("calculate accuracy of class predictions")
        print(metrics.accuracy_score(y_test, y_pred_class))
        
        lin_svc = svm.LinearSVC(C=C).fit(X_train_dtm, y_train)
        y_pred_class = lin_svc.predict(X_test_dtm)
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        print("calculate accuracy of class predictions")
        print(metrics.accuracy_score(y_test, y_pred_class))
        
        # calculate predicted probabilities for X_test_dtm (well calibrated)
        #y_pred_prob = svc.predict_proba(X_test_dtm)[:, 1]
        #print(y_pred_prob)
        # calculate accuracy
    
        #print("roc auc")
        # calculate AUC
        #print(metrics.roc_auc_score(y_test, y_pred_prob))            
        
    def createDTM_TFIDF(self, pd_negative_train, pd_neutral_train, pd_positive_train, \
            pd_test_data):
            
        print("**** createDTM ****")
        # merge all the negative, neutral and positive records
        pd_train_df = pd.concat([pd_negative_train, pd_neutral_train, pd_positive_train], ignore_index=True)
        print("# examine the class distribution - Before Modification")
        print(pd_train_df['Sentiment'].value_counts())
        
        pd_train_neg = pd_train_df.copy()
        pd_train_neg.ix[pd_train_neg.Sentiment == 1, 'Sentiment'] = 0
        pd_train_neg.ix[pd_train_neg.Sentiment == -1, 'Sentiment'] = 1
        pd_test_neg = pd_test_data.copy()
        pd_test_neg.ix[pd_test_neg.Sentiment == 1, 'Sentiment'] = 0
        pd_test_neg.ix[pd_test_neg.Sentiment == -1, 'Sentiment'] = 1
        print("Negative Dataset")
        print(pd_train_neg['Sentiment'].value_counts())
        print(pd_test_neg['Sentiment'].value_counts())
              
        X_train_neg = pd_train_neg['Sentence']
        y_train_neg = pd_train_neg['Sentiment']
        X_test_neg = pd_test_neg['Sentence']
        y_test_neg = pd_test_neg['Sentiment']
        
        vect = CountVectorizer()
        X_train_neg_dtm = vect.fit_transform(X_train_neg)
        
        #getting tf-idf values
        tfidf_transformer = TfidfTransformer()
        X_train_neg_tfidf = tfidf_transformer.fit_transform(X_train_neg_dtm)
        
        print("# transform testing data (using fitted vocabulary) into a document-term matrix")
        X_test_neg_dtm = vect.transform(X_test_neg)
        X_test_neg_tfidf = tfidf_transformer.fit_transform(X_test_neg_dtm)
        
        
        print("--------------- Checking Positive Values ----------------------")
        pd_train_pos = pd_train_df.copy()
        pd_train_pos.ix[pd_train_pos.Sentiment == -1, 'Sentiment'] = 0
        pd_test_pos = pd_test_data.copy()
        pd_test_pos.ix[pd_test_pos.Sentiment == -1, 'Sentiment'] = 0
        print("Positive Dataset")
        print(pd_train_pos['Sentiment'].value_counts())
        print(pd_train_pos['Sentiment'].value_counts())
              
        X_train_pos = pd_train_pos['Sentence']
        y_train_pos = pd_train_pos['Sentiment']
        X_test_pos = pd_test_pos['Sentence']
        y_test_pos = pd_test_pos['Sentiment']
        
        vect = CountVectorizer()
        X_train_pos_dtm = vect.fit_transform(X_train_pos)
        
        #getting tf-idf values
        tfidf_transformer = TfidfTransformer()
        X_train_pos_tfidf = tfidf_transformer.fit_transform(X_train_pos_dtm)
        
        print("# transform testing data (using fitted vocabulary) into a document-term matrix")
        X_test_pos_dtm = vect.transform(X_test_pos)
        X_test_pos_tfidf = tfidf_transformer.fit_transform(X_test_pos_dtm)

        # Returning                 
        # Training For Negative DTM and TFIDF
        # Testing For Negative DTM and TFIDF
        # Training For Positive DTM and TFIDF
        # Testing For Positive DTM and TFIDF
        
        return X_train_neg_dtm, X_train_neg_tfidf, y_train_neg, \
        X_test_neg_dtm, X_test_neg_tfidf, y_test_neg, \
        pd_test_neg, \
        X_train_pos_dtm, X_train_pos_tfidf, y_train_pos, \
        X_test_pos_dtm, X_test_pos_tfidf, y_test_pos, \
        pd_test_pos

    