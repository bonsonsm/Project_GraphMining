# -*- coding: utf-8 -*-
import pandas as pd

import nltk

import networkx as nx
import pylab
import matplotlib.pyplot as plt

from text_utilities.TextUtilities import TextFunctions
from model_utilities.ModelUtilities import ModelFunctions



#from nltk.stem.wordnet import WordNetLemmatizer


#from collections import OrderedDict


class BasicTweetSentimentPrediction:
            
    def run1(self, full_path, sheet_name, header):
        print("### Run 1 ### - Start")
        # You specifiy the train and validate percentages
        # the remaining is assumed to be the test percentage
        train_percent=.7
        validate_percent=0
        X_name='Sentence'
        y_name='Sentiment'
        
        tu = TextFunctions()
        pd_preprocess_df, pd_negative_train, pd_negative_validate, pd_negative_test, \
            pd_neutral_train, pd_neutral_validate, pd_neutral_test, \
            pd_positive_train, pd_positive_validate, pd_positive_test = \
            tu.preprocessing(full_path, sheet_name, header, X_name, y_name, \
                train_percent, validate_percent)
        # Check if the validate dataset is required
        if validate_percent != 0:
            pd_validate_data = pd.concat([pd_negative_validate, pd_neutral_validate, pd_positive_validate], ignore_index=True)
            print("Total pd_validate_data Records:",len(pd_validate_data.index))
        # create the test data by merging the test datas of pd_negative_test, pd_neutral_test, pd_positive_test
        pd_test_data = pd.concat([pd_negative_test, pd_neutral_test, pd_positive_test], ignore_index=True)
        print("Total pd_test_data Records:",len(pd_test_data.index))
        
        mu = ModelFunctions()
        # Training For Negative DTM and TFIDF
        # Testing For Negative DTM and TFIDF
        # Training For Positive DTM and TFIDF
        # Testing For Positive DTM and TFIDF
        X_train_neg_dtm, X_train_neg_tfidf, y_train_neg, \
        X_test_neg_dtm, X_test_neg_tfidf, y_test_neg, \
        pd_test_neg, \
        X_train_pos_dtm, X_train_pos_tfidf, y_train_pos, \
        X_test_pos_dtm, X_test_pos_tfidf, y_test_pos, \
        pd_test_pos = \
        mu.createDTM_TFIDF(pd_negative_train, pd_neutral_train, pd_positive_train, pd_test_data)
        
        ######### Computing for Negative ###########
        print("##### Creating Models with DTM for Negative #####")
        neg_dtm_nb = mu.compute_MultiNomial_NB(X_train_neg_dtm, y_train_neg, X_test_neg_dtm, y_test_neg, pd_test_neg)
        neg_dtm_lr = mu.compute_LogisticRegression(X_train_neg_dtm, y_train_neg, X_test_neg_dtm, y_test_neg, pd_test_neg)
        neg_dtm_svm = mu.compute_SVM(X_train_neg_dtm, y_train_neg, X_test_neg_dtm, y_test_neg, pd_test_neg)

        print("##### Creating Models with TFIDF for Negative #####")
        neg_tfidf_nb = mu.compute_MultiNomial_NB(X_train_neg_tfidf, y_train_neg, X_test_neg_tfidf, y_test_neg, pd_test_neg)
        neg_tfidf_lr = mu.compute_LogisticRegression(X_train_neg_dtm, y_train_neg, X_test_neg_dtm, y_test_neg, pd_test_neg)
        neg_tfidf_svm = mu.compute_SVM(X_train_neg_dtm, y_train_neg, X_test_neg_dtm, y_test_neg, pd_test_neg)
        
        ######### Computing for Positive ###########
        print("##### Creating Models with DTM for Positive #####")
        pos_dtm_nb = mu.compute_MultiNomial_NB(X_train_pos_dtm, y_train_pos, X_test_pos_dtm, y_test_pos, pd_test_pos)
        pos_dtm_lr = mu.compute_LogisticRegression(X_train_pos_dtm, y_train_pos, X_test_pos_dtm, y_test_pos, pd_test_pos)
        pos_dtm_svm = mu.compute_SVM(X_train_pos_dtm, y_train_pos, X_test_pos_dtm, y_test_pos, pd_test_pos)

        print("##### Creating Models with TFIDF for Positive #####")
        pos_tfidf_nb = mu.compute_MultiNomial_NB(X_train_pos_tfidf, y_train_pos, X_test_pos_tfidf, y_test_pos, pd_test_pos)
        pos_tfidf_lr = mu.compute_LogisticRegression(X_train_pos_tfidf, y_train_pos, X_test_pos_tfidf, y_test_pos, pd_test_pos)
        pos_tfidf_svm = mu.compute_SVM(X_train_pos_tfidf, y_train_pos, X_test_pos_tfidf, y_test_pos, pd_test_pos)

        print("### Run 1 ### - End")  

def main():
    obj_sentiment_prediction = BasicTweetSentimentPrediction()
    #directory = "..\\..\\data\\input\\"
    directory = "..\\data\\input\\"
    filename ="MoviewReview_RottenTomatoes_Train.xlsx"
    full_path=directory + filename
    sheet_name="train"
    header=0
    obj_sentiment_prediction.run1(full_path, sheet_name, header)
    
    
if __name__ == '__main__':
    main()
