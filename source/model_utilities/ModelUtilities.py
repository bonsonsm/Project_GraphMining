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

from basic_utilities.BasicUtilities import BasicFunctions

class ModelFunctions:
    
    def createModel(self, config_param):
        bu = BasicFunctions()
        str_output = ""
        
        calculatesm = config_param["execute.calculatesm"]
        windowsize = config_param["creategraph.window_size"]

        str_output = calculatesm
        print("calculatesm:", calculatesm)
        
        str_output = "\n############ Creating model for "+calculatesm+" similarity matrix for window size "+windowsize+" ##################\n"        
        
        #object_name_full = calculatesm +".validationfull.pkl"
        object_name_correct = calculatesm +".validationcorrect.pkl"
        object_name_test = calculatesm +".test.pkl"      
        #object_name_test = "preprocessdata.pd_test_data.pkl"      
        
        #df_validationmcsfull = bu.loadObject(config_param["data.intermediatefolder"] + object_name_full)
        df_validationmcscorrect = bu.loadObject(config_param["data.intermediatefolder"] + object_name_correct)
        df_testdatamcs = bu.loadObject(config_param["data.intermediatefolder"] + object_name_test)
        
        Xcorrect = df_validationmcscorrect.loc[:,['NegativeScore','NeutralScore','PositiveScore']]
        Xcorrect = Xcorrect.round(2)
        ycorrect = df_validationmcscorrect.loc[:,['ActualSentiment']]

        TextX = df_testdatamcs.loc[:,['NegativeScore','NeutralScore','PositiveScore']]
        TextX = TextX.round(2)
        Testy = df_testdatamcs.loc[:,['ActualSentiment']]
        
        bu.saveExcelFile("Test Probabilities.xlsx", df_testdatamcs, config_param)

        # Create another copy of the variables for doing positive and negative classifier checks
        ycorrect_positive = ycorrect.copy()
        ycorrect_negative = ycorrect.copy()
        ycorrect_neutral = ycorrect.copy()
        
        Testy_positive = Testy.copy()
        Testy_negative = Testy.copy()
        Testy_neutral = Testy.copy()
        
        ycorrect_positive.loc[:,'ActualSentiment'] = ycorrect_positive['ActualSentiment'].apply(lambda x: x if x == "Positive" else "Others")
        Testy_positive.loc[:,'ActualSentiment'] = Testy_positive['ActualSentiment'].apply(lambda x: x if x == "Positive" else "Others")
        #positive_accuracy, accuracy, y_prob_positive = self.compute_LogisticRegression(Xcorrect,ycorrect_positive,TextX,Testy_positive,TextX)
        positive_accuracy, accuracy, y_positive_pred_class = self.compute_SVM_Linear(Xcorrect,ycorrect_positive,TextX,Testy_positive,TextX)
        str_output = str_output + "\nLR - Positive Accuracy:"
        str_output = str_output + str(accuracy)
        print("LR - Positive Accuracy:", accuracy)

        ycorrect_negative.loc[:,'ActualSentiment'] = ycorrect_negative['ActualSentiment'].apply(lambda x: x if x == "Negative" else "Others")
        Testy_negative.loc[:,'ActualSentiment'] = Testy_negative['ActualSentiment'].apply(lambda x: x if x == "Negative" else "Others")
        #negative_accuracy, accuracy, y_prob_negative = self.compute_LogisticRegression(Xcorrect,ycorrect_negative,TextX,Testy_negative,TextX)
        negative_accuracy, accuracy, y_negative_pred_class = self.compute_SVM_Linear(Xcorrect,ycorrect_negative,TextX,Testy_negative,TextX)
        str_output = str_output + "\nLR - Negative Accuracy:"
        str_output = str_output + str(accuracy)
        print("LR - Negative Accuracy:", accuracy)

        ycorrect_neutral.loc[:,'ActualSentiment'] = ycorrect_neutral['ActualSentiment'].apply(lambda x: x if x == "Neutral" else "Others")
        Testy_neutral.loc[:,'ActualSentiment'] = Testy_neutral['ActualSentiment'].apply(lambda x: x if x == "Neutral" else "Others")
        #neutral_accuracy, accuracy, y_prob_neutral = self.compute_LogisticRegression(Xcorrect,ycorrect_neutral,TextX,Testy_neutral,TextX)
        neutral_accuracy, accuracy, y_neutral_pred_class = self.compute_SVM_Linear(Xcorrect,ycorrect_neutral,TextX,Testy_neutral,TextX)
        str_output = str_output + "\nLR - Neutral Accuracy:"
        str_output = str_output + str(accuracy)     
        print("LR - Neutral Accuracy:", accuracy)
        
            
        
        #Xfull = df_validationmcsfull.loc[:,['NegativeScore','NeutralScore','PositiveScore']]
        #yfull = df_validationmcsfull.loc[:,['ActualSentiment']]        
        #yfull.loc[:,'ActualSentiment'] = yfull['ActualSentiment'].apply(lambda x: x if x == "Positive" else "Others")
        #Testy.loc[:,'ActualSentiment'] = Testy['ActualSentiment'].apply(lambda x: x if x == "Positive" else "Others")
        #positive_accuracy, accuracy, y_prob_full = self.compute_LogisticRegression(Xfull,yfull,TextX,Testy,TextX)
        #str_output = str_output + "\nLR - Positive Accuracy:"
        #str_output = str_output + str(accuracy)   
        #print("LR - Positive FULL Accuracy:", accuracy)
        
        #pd_probability = pd.DataFrame(y_prob_positive,columns=['Positive'])
        #pd_probability['Negative'] = pd.Series(y_prob_negative, index=pd_probability.index)
        #pd_probability['Neutral'] = pd.Series(y_prob_neutral, index=pd_probability.index)
        #print("All length:", len(y_prob_full))
        #pd_probability['All'] = pd.Series(y_prob_full, index=pd_probability.index)
        
        df_testdatamcs['Positive Model'] = pd.Series(y_positive_pred_class, index=df_testdatamcs.index)
        df_testdatamcs['Negative Model'] = pd.Series(y_negative_pred_class, index=df_testdatamcs.index)
        df_testdatamcs['Neutral Model'] = pd.Series(y_neutral_pred_class, index=df_testdatamcs.index)
        
        #print("All length:", len(y_prob_full))
        #df_testdatamcs['All Model'] = pd.Series(y_prob_full, index=df_testdatamcs.index)
        
        #print("ycorrect length:", len(Testy))
        #pd_probability['ActualSentiment'] = Testy
        
        bu.saveExcelFile("Final Probabilities.xlsx", df_testdatamcs, config_param)
        
        output_file_name = config_param["data.outputfolder"] + calculatesm + "_w" + windowsize + "_"+"output.txt"
        bu.saveTextFile(output_file_name, str_output)
        
        #DF_ForModel = pd.read_excel(full_path, sheet_name, header)
        #print(DF_ForModel.head())  
        #X = DF_ForModel.loc[:,['NegativeScore','NeutralScore','PositiveScore']]
        #print(X.head())  
        #y = DF_ForModel.loc[:,['ActualSentiment']]
        #DF_TestData = pd.read_excel("TestDataFrame.xlsx", sheet_name, header)
        #TextX = DF_TestData.loc[:,['NegativeScore','NeutralScore','PositiveScore']]
        #print(X.head())  
        #Testy = DF_TestData.loc[:,['ActualSentiment']]
        
        
        
        
        
            

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
        print("X_train_dtm")
        print(X_train_dtm)
        print("y_train.values.ravel()")
        print(y_train.values.ravel())
        print("X_test_dtm")
        print(X_test_dtm)
        logreg.fit(X_train_dtm, y_train.values.ravel())
        # make class predictions for X_test_dtm
        y_pred_class = logreg.predict(X_test_dtm)
        print("y_pred_class")
        print(y_pred_class)
        print("y_test")
        print(y_test)
        # calculate predicted probabilities for X_test_dtm (well calibrated)
        y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
        print("y_pred_prob")
        print(y_pred_prob)
        
        #print(type(y_pred_prob))        
        #print(y_pred_prob)
        # calculate accuracy
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        #print("calculate accuracy of class predictions")
        accuracy = metrics.accuracy_score(y_test, y_pred_class)
        #print(metrics.accuracy_score(y_test, y_pred_class))
        #print("roc auc")
        # calculate AUC
        #print(metrics.roc_auc_score(y_test, y_pred_prob))
        return logreg, accuracy, y_pred_prob
        
    def compute_SVM_Linear(self, X_train_dtm, y_train, X_test_dtm, y_test, X_test):
        print("########## Computing SVM Linear##########")
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 5.0  # SVM regularization parameter
        lin_svc = svm.LinearSVC(C=C).fit(X_train_dtm, y_train.values.ravel())
        y_pred_class = lin_svc.predict(X_test_dtm)
        print("print the confusion matrix")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        print("calculate accuracy of class predictions")
        accuracy = metrics.accuracy_score(y_test, y_pred_class)
        return lin_svc, accuracy, y_pred_class
        
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

    