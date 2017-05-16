# -*- coding: utf-8 -*-
import pandas as pd
from timeit import default_timer as timer

from text_utilities.TextUtilities import TextFunctions
from graph_utilities.GraphUtilities import GraphFunctions
from basic_utilities.BasicUtilities import BasicFunctions
from model_utilities.ModelUtilities import ModelFunctions

class GraphTweetSentimentPrediction:
            
    def run1(self, config_param):
        print("### Run 1 ### - Start")
        # You specifiy the train and validate percentages
        # the remaining is assumed to be the test percentage
        
        full_path = config_param["data.inputfolder"] + config_param["data.inputfilename"]
        sheet_name = config_param["data.inputsheet_name"]
        header = int(config_param["data.inputheader"])

        train_percent=float(config_param["data.inputtrain_percent"])
        validate_percent=float(config_param["data.inputvalidate_percent"])
        X_name=config_param["data.inputX_name"]
        y_name=config_param["data.inputy_name"]
        
        bu = BasicFunctions()
        gu = GraphFunctions()
        mu = ModelFunctions()
        tu = TextFunctions()
        
        # This condition overrides all the other executino parameters 
        # and returns after execution. 
        if config_param["execute.temp"] == "1":
            print("Executing Temp")

            main_sentence = "A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story ."
            main_data = {'id': ['1'], 'sentence':[main_sentence], 'sentiment': [0]}
            df_maingraph_data = pd.DataFrame(main_data, columns = ['id', 'sentence', 'sentiment'])
            main_graph, avg = gu.createGraph(df_maingraph_data)
            #gu.drawGraph(main_graph)

            #test_sentence_1 = "not much of a story"    #ThisWorks
            test_sentence_1 = "not much of a story of a what is good"    #This DOES NOT Works
            test_data = {'id': ['1'], 'sentence':[test_sentence_1], 'sentiment': [0]}
            df_testgraph_data = pd.DataFrame(test_data, columns = ['id', 'sentence', 'sentiment'])
            test_graph, avg = gu.createGraph(df_testgraph_data)
            gu.drawGraph(test_graph)
            
            mcsns = gu.getMCSNS(main_graph, test_graph)
            print("MCSNS:", mcsns)
            mcsues = gu.getMCSUES(main_graph, test_graph)
            print("MCSUES:", mcsues)
            mcsdes = gu.getMCSDES(main_graph, test_graph)
            print("MCSDES:", mcsdes)
            return
        
        if config_param["execute.preprocessdata"] == "1":
            
            str_output = ""
            #str_output = str_output + "\n" + "In execute.preprocessdata"
            print("In execute.preprocessdata")
            pd_preprocess_df, pd_negative_train, pd_negative_validate, pd_negative_test, \
                pd_neutral_train, pd_neutral_validate, pd_neutral_test, \
                pd_positive_train, pd_positive_validate, pd_positive_test = \
                tu.preprocessing(full_path, sheet_name, header, X_name, y_name, \
                    train_percent, validate_percent, 'None', config_param)
                
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_preprocess_df.pkl", pd_preprocess_df)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_negative_train.pkl", pd_negative_train)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_negative_validate.pkl", pd_negative_validate)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_negative_test.pkl", pd_negative_test)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_neutral_train.pkl", pd_neutral_train)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_neutral_validate.pkl", pd_neutral_validate)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_neutral_test.pkl", pd_neutral_test)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_positive_train.pkl", pd_positive_train)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_positive_validate.pkl", pd_positive_validate)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_positive_test.pkl", pd_positive_test)

            # Check if the validate dataset is required
            if validate_percent != 0:
                pd_validate_data = pd.concat([pd_negative_validate, pd_neutral_validate, pd_positive_validate], ignore_index=True)
                bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_validate_data.pkl", pd_validate_data)
                #str_output = str_output + "\nTotal pd_validate_data Records:" + str(len(pd_validate_data.index))
                print("Total pd_validate_data Records:",len(pd_validate_data.index))
            # create the test data by merging the test datas of pd_negative_test, pd_neutral_test, pd_positive_test
            pd_test_data = pd.concat([pd_negative_test, pd_neutral_test, pd_positive_test], ignore_index=True)
            bu.saveObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_test_data.pkl", pd_test_data)

            print("Total pd_test_data Records:",len(pd_test_data.index))
            #str_output = str_output + "\nTotal pd_test_data Records:" +str(len(pd_test_data.index))
            bu.saveExcelFile("pd_negative_train.xlsx", pd_negative_train, config_param)
            bu.saveExcelFile("pd_neutral_train.xlsx", pd_neutral_train, config_param)
            bu.saveExcelFile("pd_positive_train.xlsx", pd_positive_train, config_param)
            bu.saveExcelFile("pd_validate_data.xlsx", pd_validate_data, config_param)
            bu.saveExcelFile("pd_test_data.xlsx", pd_test_data, config_param)

            str_output = str_output + "\n############Pre Process Data Completed##################\n"
            calculatesm = config_param["execute.calculatesm"]
            windowsize = config_param["creategraph.window_size"]
            output_file_name = config_param["data.outputfolder"] + calculatesm + "_w" + windowsize + "_"+"output.txt"
            bu.saveTextFile(output_file_name, str_output)
            
        if config_param["execute.creategraphs"] == "1":
            
            #raw_data = {'id': ['1','2'], 'sentence':['What is good for the gardner is also good for the goose','He is a good boy'], 'sentiment': [0,2]}
            #pd_neutral_train = pd.DataFrame(raw_data, columns = ['id', 'sentence', 'sentiment'])
    
            sentences_to_take = int(config_param["creategraph.sentences_to_take"])
            window_size = int(config_param["creategraph.window_size"])
            
            pd_negative_train = bu.loadObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_negative_train.pkl")
            pd_neutral_train = bu.loadObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_neutral_train.pkl")
            pd_positive_train = bu.loadObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_positive_train.pkl")
            
            negative_graph, avg_tokens_per_sentence_main = gu.createGraph(pd_negative_train, sentences_to_take, window_size)
            neutral_graph, avg_tokens_per_sentence_main = gu.createGraph(pd_neutral_train, sentences_to_take, window_size)
            positive_graph, avg_tokens_per_sentence_main = gu.createGraph(pd_positive_train, sentences_to_take, window_size)
            
            str_output = ""
            str_output = str_output + "\n############ Create Graphs ##################"
            str_output = str_output + "\nNegative Graph \nNodes:"+str(len(negative_graph.nodes()))+" Edges:"+str(len(negative_graph.edges()))
            str_output = str_output + "\nNegative Graph \n"+gu.printGraphTable(negative_graph)
            
            str_output = str_output + "\nNeutral Graph \nNodes:"+str(len(neutral_graph.nodes()))+" Edges:"+str(len(neutral_graph.edges()))
            str_output = str_output + "\nNeutral Graph \n"+gu.printGraphTable(neutral_graph)
            
            str_output = str_output + "\nPositive Graph\nNodes:"+str(len(positive_graph.nodes()))+" Edges:"+str(len(positive_graph.edges()))
            str_output = str_output + "\nPositive Graph \n"+gu.printGraphTable(positive_graph)
            
            str_output = str_output + "\n############ Create Graphs Completed ##################"
            print("negative_graph: ",len(negative_graph.nodes()))
            print("neutral_graph: ", len(neutral_graph.nodes()))
            print("positive_graph: ", len(positive_graph.nodes()))
            
            bu.saveObject(config_param["data.intermediatefolder"] + "creategraph.negative_graph.pkl", negative_graph)
            bu.saveObject(config_param["data.intermediatefolder"] + "creategraph.neutral_graph.pkl", neutral_graph)
            bu.saveObject(config_param["data.intermediatefolder"] + "creategraph.positive_graph.pkl", positive_graph)
            
            calculatesm = config_param["execute.calculatesm"]
            output_file_name = config_param["data.outputfolder"] + calculatesm + "_w" + windowsize + "_"+"output.txt"
            bu.saveTextFile(output_file_name, str_output)
            
            #gu.drawGraph(negative_graph)
            #gu.printGraphTable(negative_graph)
        
        # This executes for creation of validation data
        if config_param["execute.computecsm_validationdata"] == "1":
            negative_graph = bu.loadObject(config_param["data.intermediatefolder"] + "creategraph.negative_graph.pkl")
            neutral_graph = bu.loadObject(config_param["data.intermediatefolder"] + "creategraph.neutral_graph.pkl")
            positive_graph = bu.loadObject(config_param["data.intermediatefolder"] + "creategraph.positive_graph.pkl")
            pd_validate_data = bu.loadObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_validate_data.pkl")
            #run_type = config_param["computecsm.runtype"]
            start_compute_sm = timer()
            str_called_from = "validation"
            # Computing the Containment Similarity Matrix for the validation data
            gu.compute_sm(negative_graph, neutral_graph, positive_graph, pd_validate_data, config_param, str_called_from)
            end_compute_sm = timer()
            print("Time taken by compute_csm():",end_compute_sm - start_compute_sm)
        
        # This executes for creation of test data
        if config_param["execute.computecsm_testdata"] == "1":
            negative_graph = bu.loadObject(config_param["data.intermediatefolder"] + "creategraph.negative_graph.pkl")
            neutral_graph = bu.loadObject(config_param["data.intermediatefolder"] + "creategraph.neutral_graph.pkl")
            positive_graph = bu.loadObject(config_param["data.intermediatefolder"] + "creategraph.positive_graph.pkl")
            pd_test_data = bu.loadObject(config_param["data.intermediatefolder"] + "preprocessdata.pd_test_data.pkl")
            start_compute_sm = timer()
            str_called_from = "test"
            # Computing the Containment Similarity Matrix for the test data
            gu.compute_sm(negative_graph, neutral_graph, positive_graph, pd_test_data, config_param, str_called_from)
            end_compute_sm = timer()
            print("Time taken by compute_csm():",end_compute_sm - start_compute_sm)
        
        if config_param["execute.csm_validationdata_model"] == "1":
            mu.createModel(config_param)
            
        print("### Run 1 ### - End")  
    
def main():
    path = "..\\config\\config.ini"
    bu = BasicFunctions()
    config_param = bu.readConfig(path)
    
    obj_sentiment_prediction = GraphTweetSentimentPrediction()
    obj_sentiment_prediction.run1(config_param)
    
    
if __name__ == '__main__':
    main()
