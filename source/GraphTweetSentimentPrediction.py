# -*- coding: utf-8 -*-
import pandas as pd

import nltk

import networkx as nx
import pylab
import matplotlib.pyplot as plt

from text_utilities.TextUtilities import TextFunctions
from model_utilities.ModelUtilities import ModelFunctions
from graph_utilities.GraphUtilities import GraphFunctions




class GraphTweetSentimentPrediction:
            
    def run1(self, full_path, sheet_name, header):
        print("### Run 1 ### - Start")
        # You specifiy the train and validate percentages
        # the remaining is assumed to be the test percentage
        train_percent=.4
        validate_percent=0.4
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
        
        gu = GraphFunctions()
        #raw_data = {'id': ['1','2'], 'sentence':['What is good for the gardner is also good for the goose','He is a good boy'], 'sentiment': [0,2]}
        #pd_neutral_train = pd.DataFrame(raw_data, columns = ['id', 'sentence', 'sentiment'])

        sentences_to_take=0
        window_size=1
        
        negative_graph, avg_tokens_per_sentence_main = gu.createGraph(pd_negative_train, sentences_to_take, window_size)
        neutral_graph, avg_tokens_per_sentence_main = gu.createGraph(pd_neutral_train, sentences_to_take, window_size)
        positive_graph, avg_tokens_per_sentence_main = gu.createGraph(pd_positive_train, sentences_to_take, window_size)
        
        print("negative_graph: ",negative_graph.nodes)
        print("neutral_graph: ",neutral_graph.nodes)
        print("positive_graph: ",positive_graph.nodes)
        
        #gu.drawGraph(negative_graph)
        gu.printGraphTable(negative_graph)
        
        # Computing the Containment Similarity Matrix for the Validate Data
        gu.compute_csm(negative_graph, neutral_graph, positive_graph, pd_validate_data, 0)
        
        print("### Run 1 ### - End")  

def main():
    obj_sentiment_prediction = GraphTweetSentimentPrediction()
    #directory = "..\\..\\data\\input\\"
    directory = "..\\data\\input\\"
    filename ="MoviewReview_RottenTomatoes_Train.xlsx"
    full_path=directory + filename
    sheet_name="train"
    header=0
    obj_sentiment_prediction.run1(full_path, sheet_name, header)
    
    
if __name__ == '__main__':
    main()
