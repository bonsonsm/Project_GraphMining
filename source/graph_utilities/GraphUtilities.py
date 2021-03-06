# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk

import networkx as nx
import pylab
import matplotlib.pyplot as plt
#from custom.graph_traversal import GraphTraversal as gt

from nltk.stem import WordNetLemmatizer
from collections import OrderedDict

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm, datasets

from basic_utilities.BasicUtilities import BasicFunctions
from text_utilities.TextUtilities import TextFunctions

import collections

class GraphFunctions:
    
    
    def getMCS(self, G_source, G_new):
        """
        Creator: Bonson
        Return the MCS of the G_new graph that is present 
        in the G_source graph
        
        #Why can't you just filter for common edges and then compute the largest connected component?
        
        """
        #self.printGraphTable(G_source)
        #self.printGraphTable(G_new)
        
        matching_graph=nx.Graph()
        
        for n1,n2,attr in G_new.edges(data=True):
            #key = DG.node[n1]['order']
            #value  = str(DG.node[n2]['order']) + " "+ n1 + " "+n2
            #print(DG.node[n1]['order'], DG.node[n2]['order'], n1,n2)
            #d[key] = value
            #l_value = str(DG.node[n1]['order']) + " "+str(DG.node[n2]['order']) + " "+ n1 + " "+n2
            #l.append(l_value)
            
            #print(n1)
            #print(n2)
            
            if G_source.has_edge(n1,n2) :
                #print(G_source.has_edge(n1,n2))
                matching_graph.add_edge(n1,n2,weight=1)
                matching_graph.node[n1]['order'] = G_new.node[n1]['order']
                matching_graph.node[n2]['order'] = G_new.node[n2]['order']
        
        graphs = list(nx.connected_component_subgraphs(matching_graph))
        
        mcs_length = 0
        mcs_graph = nx.Graph()
        for i, graph in enumerate(graphs):
            
            if len(graph.nodes()) > mcs_length:
                mcs_length = len(graph.nodes())
                mcs_graph = graph
        
        #print(len(nx.connected_components(mcs_graph)))
        #print(mcs_graph.nodes())
        #self.drawGraph(mcs_graph)
        total_weight=0
        for n1,n2,attr in mcs_graph.edges(data=True):
           #print(attr)
           #w = G_source[n1][n2]['weight']
           w = attr['weight']
           total_weight=total_weight+w
        #print(total_weight)
        #print("***printing MCS***")
        #self.printGraphTable(mcs_graph)
        return mcs_graph, total_weight
                
      
    def createGraph(self, df_data, sentences_to_take=0, window_size=1):
        #print("### createGraph ### - Start")
        if sentences_to_take == 0:
            SENTENCES_CONSIDERED=len(df_data.index)
        else:
            SENTENCES_CONSIDERED=sentences_to_take
        sentence = ""
        sentence_count=1
        tokens_added=[]
        i=0
        node_id=0
        total_token_count = 0
        avg_tokens_per_sentence = 0
        DG=nx.DiGraph()
        for index, row in df_data.iterrows():
            sentence = row[1]
            print(sentence)
            if sentence_count > SENTENCES_CONSIDERED:
                sentence_count = sentence_count -1
                break
            else:
                tokens = nltk.word_tokenize(sentence)
                token_count = len(tokens)
                total_token_count = total_token_count + token_count
                #print("token_count", token_count)
                for i in range(token_count):
                    #print("#### Next Token ####")
                    #To take a minimum of 2 tokens
                    if i == 0:
                        continue
                    
                    if window_size == 1 and i>= 1:
                        if DG.has_edge(tokens[i-1], tokens[i]):
                            w = DG[tokens[i-1]][tokens[i]]['weight']
                            w=w+1
                            #print("*** In 1 - HAS Edge**** Adding Weight ***",tokens[i-1],"->",tokens[i]," weigth:",w)
                            DG.add_edges_from([(tokens[i-1], tokens[i])], weight=w)
                        else:
                            #print("*** In 1 - NEW Edge**** Adding Weight ***",tokens[i-1],"->",tokens[i]," weigth:1")
                            DG.add_edges_from([(tokens[i-1], tokens[i])], weight=1)
                    
                        if tokens[i-1] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + i
                            else:
                                node_id= node_id + i-1
    
                            DG.node[tokens[i-1]]['order'] = node_id
                            tokens_added.append(tokens[i-1])
                            
                        if tokens[i] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + 1
                            else:
                                node_id= node_id + 1
                            DG.node[tokens[i]]['order'] = node_id
                            tokens_added.append(tokens[i])
                        
                    if window_size == 2 and i>= 2:
                        
                        if DG.has_edge(tokens[i-2], tokens[i]):
                            w = DG[tokens[i-2]][tokens[i]]['weight']
                            w=w+1
                            #print("*** In 2 - HAS Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i]," weigth:",w)
                            DG.add_edges_from([(tokens[i-2], tokens[i])], weight=w)
                        else:
                            #print("*** In 2 - NEW Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i]," weigth:1")
                            DG.add_edges_from([(tokens[i-2], tokens[i])], weight=1)
                        
                        if DG.has_edge(tokens[i-2], tokens[i-1]):
                            w = DG[tokens[i-2]][tokens[i-1]]['weight']
                            w=w+1
                            #print("*** In 2 - HAS Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i-1]," weigth:",w)
                            DG.add_edges_from([(tokens[i-2], tokens[i-1])], weight=w)
                        else:
                            #print("*** In 2 - NEW Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i-1]," weigth:1")
                            DG.add_edges_from([(tokens[i-2], tokens[i-1])], weight=1)
                        ################Adding to Tokens################################    
                        if tokens[i-2] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + i
                            else:
                                node_id= node_id + i-1
                            DG.node[tokens[i-2]]['order'] = node_id
                            tokens_added.append(tokens[i-2])
                            
                        if tokens[i-1] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + i
                            else:
                                node_id= node_id + i-1
                            DG.node[tokens[i-1]]['order'] = node_id
                            tokens_added.append(tokens[i-1])
                        
                        if tokens[i] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + 1
                            else:
                                node_id= node_id + 1
                            DG.node[tokens[i]]['order'] = node_id
                            tokens_added.append(tokens[i])

                        ################For the last nodes################################  
                        if token_count - i == 1:
                            #print("*** Last Nodes ***",tokens[i-1],"->",tokens[i])
                            if DG.has_edge(tokens[i-1], tokens[i]):
                                w = DG[tokens[i-1]][tokens[i]]['weight']
                                w=w+1
                                #print("*** In 2 - HAS Edge**** Adding Weight ***",tokens[i-1],"->",tokens[i]," weigth:",w)
                                DG.add_edges_from([(tokens[i-1], tokens[i])], weight=w)
                            else:
                                #print("*** In 2 - NEW Edge**** Adding Weight ***",tokens[i-1],"->",tokens[i]," weigth:1")
                                DG.add_edges_from([(tokens[i-1], tokens[i])], weight=1)
                            
                    if window_size == 3 and i>= 3:
                        if DG.has_edge(tokens[i-3], tokens[i]):
                            w = DG[tokens[i-3]][tokens[i]]['weight']
                            w=w+1
                            #print("*** In 3a - HAS Edge**** Adding Weight ***",tokens[i-3],"->",tokens[i]," weigth:",w)
                            DG.add_edges_from([(tokens[i-3], tokens[i])], weight=w)
                        else:
                            #print("*** In 3a - NEW Edge**** Adding Weight ***",tokens[i-3],"->",tokens[i]," weigth:1")
                            DG.add_edges_from([(tokens[i-3], tokens[i])], weight=1)
                            
                        if DG.has_edge(tokens[i-3], tokens[i-1]):
                            w = DG[tokens[i-3]][tokens[i-1]]['weight']
                            w=w+1
                            #print("*** In 3b - HAS Edge**** Adding Weight ***",tokens[i-3],"->",tokens[i-1]," weigth:",w)
                            DG.add_edges_from([(tokens[i-3], tokens[i-1])], weight=w)
                        else:
                            #print("*** In 3b - NEW Edge**** Adding Weight ***",tokens[i-3],"->",tokens[i-1]," weigth:1")
                            DG.add_edges_from([(tokens[i-3], tokens[i-1])], weight=1)
                        
                        if DG.has_edge(tokens[i-3], tokens[i-2]):
                            w = DG[tokens[i-3]][tokens[i-2]]['weight']
                            w=w+1
                            DG.add_edges_from([(tokens[i-3], tokens[i-2])], weight=w)
                            #print("*** In 3c - HAS Edge**** Adding Weight ***",tokens[i-3],"->",tokens[i-2]," weigth:",w)
                        else:
                            #print("*** In 3c - NEW Edge**** Adding Weight ***",tokens[i-3],"->",tokens[i-2]," weigth:1")
                            DG.add_edges_from([(tokens[i-3], tokens[i-2])], weight=1)
                        
                        ################Adding to Tokens################################    
                        if tokens[i-3] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + i
                            else:
                                node_id= node_id + i-1
                            DG.node[tokens[i-3]]['order'] = node_id
                            tokens_added.append(tokens[i-3])
                            
                        if tokens[i-2] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + i
                            else:
                                node_id= node_id + i-1
                            DG.node[tokens[i-2]]['order'] = node_id
                            tokens_added.append(tokens[i-2])
                            
                        if tokens[i-1] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + i
                            else:
                                node_id= node_id + i-1
                            DG.node[tokens[i-1]]['order'] = node_id
                            tokens_added.append(tokens[i-1])
                        
                        if tokens[i] not in tokens_added:
                            if sentence_count >= 2:
                                node_id= node_id + 1
                            else:
                                node_id= node_id + 1
                            DG.node[tokens[i]]['order'] = node_id
                            tokens_added.append(tokens[i])
                            
                        ################For the last nodes################################  
                        if token_count - i == 1:
                            #print("*** Last Nodes ***",tokens[i-2],"->",tokens[i])
                            #print("*** Last Nodes ***",tokens[i-2],"->",tokens[i-1])
                            #print("*** Last Nodes ***",tokens[i-1],"->",tokens[i])
                            
                            if DG.has_edge(tokens[i-2], tokens[i]):
                                w = DG[tokens[i-2]][tokens[i]]['weight']
                                w=w+1
                                #print("*** In 3 - HAS Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i]," weigth:",w)
                                DG.add_edges_from([(tokens[i-2], tokens[i])], weight=w)
                            else:
                                #print("*** In 3 - NEW Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i]," weigth:1")
                                DG.add_edges_from([(tokens[i-2], tokens[i])], weight=1)
                                
                            if DG.has_edge(tokens[i-2], tokens[i-1]):
                                w = DG[tokens[i-2]][tokens[i-1]]['weight']
                                w=w+1
                                #print("*** In 3 - HAS Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i-1]," weigth:",w)
                                DG.add_edges_from([(tokens[i-2], tokens[i-1])], weight=w)
                            else:
                                #print("*** In 3 - NEW Edge**** Adding Weight ***",tokens[i-2],"->",tokens[i-1]," weigth:1")
                                DG.add_edges_from([(tokens[i-2], tokens[i-1])], weight=1)                            
                            
                            if DG.has_edge(tokens[i-1], tokens[i]):
                                w = DG[tokens[i-1]][tokens[i]]['weight']
                                w=w+1
                                #print("*** In 3 - HAS Edge**** Adding Weight ***",tokens[i-1],"->",tokens[i]," weigth:",w)
                                DG.add_edges_from([(tokens[i-1], tokens[i])], weight=w)
                            else:
                                #print("*** In 3 - NEW Edge**** Adding Weight ***",tokens[i-1],"->",tokens[i]," weigth:1")
                                DG.add_edges_from([(tokens[i-1], tokens[i])], weight=1)
                            
            sentence_count=sentence_count + 1
            
        #print("Total Sentence Count",sentence_count)
        #print("Total Token Count",total_token_count)
        avg_tokens_per_sentence = total_token_count / sentence_count
        #print("avg_tokens_per_sentence",avg_tokens_per_sentence)
        #print(list(DG.nodes()))
        #print(list(DG.edges()))
        print("### createGraph ### - End")
        return DG, avg_tokens_per_sentence
        
    def drawGraph(self, DG):
        print("### In DrawGraph")
        node_labels = {node:node for node in DG.nodes()}
        val_map = {'FIRST': 1.0,'NORMAL': 0.5714285714285714,'LAST': 0.0}
        values = [val_map.get(node, 1.0) for node in DG.nodes()]
                  
        edge_labels=dict([((u,v,),d['weight']) for u,v,d in DG.edges(data=True)])
        #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
        edge_colors = ['black' for edge in DG.edges()]
        pos=nx.spring_layout(DG)
        nx.draw_networkx_labels(DG, pos, labels=node_labels)
        nx.draw_networkx_edge_labels(DG,pos,edge_labels=edge_labels)
        #nx.draw(DG,pos, node_color = values, node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
        nx.draw(DG,pos, node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
        pylab.show()
                              
        nx.draw(DG)        
        
    def printGraphTable(self, DG):
        #print("### In PrintGraphTable")
        str_graph=""
        #print(DG.edges(data=True))
        l = []
        for n1,n2,attr in DG.edges(data=True):
            #key = DG.node[n1]['order']
            #value  = str(DG.node[n2]['order']) + " "+ n1 + " "+n2
            #print(DG.node[n1]['order'], DG.node[n2]['order'], n1,n2)
            #d[key] = value
            l_value = str(DG.node[n1]['order']) + " "+str(DG.node[n2]['order']) + " "+ n1 + " "+n2+" "+str(attr['weight'])
            l.append(l_value)
        l.sort()
        for v in l:
            #print(v)
            str_graph = str_graph + v +"\n"
        #od = collections.OrderedDict(sorted(d.items()))
        #for k, v in od.items(): 
        #    print(k, v)
        
        #for a, b, data in sorted(DG.edges(data=True), key=lambda(a, b, data): data['weight']):
        #    print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
        return str_graph
        
    def getMCSNS(self, maingraph, subgraph):
        """
        Guage the similarity between the N-Gram graphs.
        Total No. of Nodes in MCS / Number of edges of smaller graph
        """
        print("In getMCSNS")
        mcs, weight = self.getMCS(maingraph, subgraph)
        print("after ", weight)
        mcs_nodes = len(mcs.nodes())
        print("mcs_nodes ", mcs_nodes)
        nodes_maingraph = len(maingraph.nodes())
        print("nodes_maingraph ", nodes_maingraph)
        nodes_subgraph = len(subgraph.nodes())
        print("nodes_subgraph ", nodes_subgraph)
        if nodes_maingraph == 0 or nodes_subgraph == 0:
            return 0,0,0
        mcsns=0.0
        print("Numerator:",mcs_nodes)
        if(nodes_subgraph<nodes_maingraph):
            mcsns = mcs_nodes/nodes_subgraph
            print("denominator:",nodes_subgraph)
        else:
            mcsns = mcs_nodes/nodes_maingraph
            print("denominator:",nodes_maingraph)
        print("End getMCSNS")
        return mcsns, mcs, weight;
        
    def getMCSUES(self, maingraph, subgraph):
        """
        Guage the similarity between the N-Gram graphs.
        Total No. of edges in MCS regardless of direction / Number of edges of smaller graph
        """
        #print("In getMCSUES")
        mcs, weight = self.getMCS(maingraph, subgraph)
        mcs_edges = len(mcs.edges())
        nodes_maingraph = len(maingraph.nodes())
        nodes_subgraph = len(subgraph.nodes())
        if nodes_maingraph == 0 or nodes_subgraph == 0:
            return 0
        mcsues=0.0
        #print("Numerator:",mcs_edges)
        if(nodes_subgraph<nodes_maingraph):
            mcsues = mcs_edges/nodes_subgraph
            #print("Denominator:",edges_subgraph)
        else:
            mcsues = mcs_edges/nodes_maingraph
            #print("Denominator:",edges_maingraph)
        return mcsues, mcs, weight;
        
    def getMCSDES(self, maingraph, subgraph):
        """
        Guage the similarity between the N-Gram graphs.
        Total No. of Nodes in MCS with same direction / Number of edges of smaller graph
        """
        #print("In getMCSDES")
        mcs, weight = self.getMCS(maingraph, subgraph)
        mcs_edges = len(mcs.edges())
        nodes_maingraph = len(maingraph.nodes())
        nodes_subgraph = len(subgraph.nodes())
        if nodes_maingraph == 0 or nodes_subgraph == 0:
            return 0
        mcsdes=0.0
        #print("Numerator:",mcs_edges)
        if(nodes_subgraph<nodes_maingraph):
            #mcsdes = mcs_edges/nodes_subgraph
            mcsdes = weight/nodes_subgraph
            #print("Denominator:",nodes_subgraph)
        else:
            #mcsdes = mcs_edges/nodes_maingraph
            mcsdes = weight/nodes_maingraph
            #print("Denominator:",nodes_maingraph)
        return mcsdes, mcs, weight;
    
    def getContainmentSimilarity(self, maingraph, subgraph):
        """
        Guage the similarity between the N-Gram graphs.
        Number of common edges between the two graphs / Number of edges of smaller graph
        NOTE: The common edges is not dependent on MCS
        It means all the common edges of the subgraph even if it in not part of the MCS
        """
        #print("### In getContainmentSimilarity")
        edges_maingraph = len(maingraph.edges())
        #print("Total Edges", edges_main_graph)
        edges_subgraph = len(subgraph.edges())
        
        if edges_maingraph == 0 or edges_subgraph == 0:
            return 0
        #print("Total Edges sub_graph", edges_sub_graph)        
        #edges_mcs = len(mcs.edges())
        common_edge_count = 0
        for node_s,node_e,edge_w in subgraph.edges_iter(data=True):
            #print("{0}->{1}".format(node_s,node_e))
            if node_s in maingraph.nodes() and node_e in maingraph.nodes() and node_e in maingraph.neighbors(node_s):
                common_edge_count+=1
        #print("common_edge_count",common_edge_count)
        cs=0.0
        if(edges_subgraph<edges_maingraph):
            # Number of edges of subgraph is less
            #print("edges_subgraph",edges_subgraph)                
            cs = common_edge_count/edges_subgraph
        else:
            # Number of edges of main graph is less
            #print("edges_maingraph",edges_maingraph)                
            cs = common_edge_count/edges_maingraph
        #print("Containment Similarity:", cs)
        return cs;
        
    def compute_sm(self, dg_neg, dg_zero, dg_pos, pd_validate_data, config_param, str_called_from):
        """
        Compute and return the Containment Similarity Matrix DF that contains 
        sentence_id
        sentence
        neg_score
        zero_score
        pos_score
        actual
        classification
        """
        print(" Compute CSM ### - Start")
        
        not_accurate=0
        accurate=0
        duplicate=0
        
        cal_sentiment=""
        actual_sentiment=""
        two_max_values=""
        
        # temporary assignment just o initialize variables
        n_mcs=dg_neg
        z_mcs=dg_neg
        p_mcs=dg_neg
        
        calculatesm = config_param["execute.calculatesm"]
        
        print("calculatesm:", calculatesm)
        str_output = "\n############ Computing "+calculatesm+" similarity matrix for "+str_called_from+" data##################\n"        
        
        neg_value = 0
        zero_value = 0
        pos_value = 0
        
        row_list =[]
        ctr = 0
        
        #tu = TextFunctions()
        
        for index, row in pd_validate_data.iterrows():
            two_max_values="No"
            ctr = ctr +1
            #print("ctr for pd_validate_data", ctr)
            accuracy=0
            sentence_id = row['SentenceID']
            
            #print("Computing for Sentence ID:",sentence_id)
            sentence = row['Sentence']
            str_output = str_output + "\n-----SENTENCE-------\n"+sentence            
            actual_sentiment_score = row['Sentiment']
            
            #if actual_sentiment_score == 0 or actual_sentiment_score ==1:
            #    actual_sentiment="Negative"
            #if actual_sentiment_score == 2:
            #    actual_sentiment="Neutral"
            #if actual_sentiment_score == 3 or actual_sentiment_score ==4:
            #    actual_sentiment="Positive"

            if actual_sentiment_score == -1:
                actual_sentiment="Negative"
            if actual_sentiment_score == 0:
                actual_sentiment="Neutral"
            if actual_sentiment_score == 1:
                actual_sentiment="Positive"
            
            
            
            #print("sentence_id",sentence_id)
            #print("sentence",sentence)
            #print("actual_sentiment",actual_sentiment_score)
            #print("actual_sentiment",actual_sentiment)
            
            raw_data = {'id': [sentence_id], 'sentence':[sentence], 'sentiment': [actual_sentiment_score]}
            pd_sentence = pd.DataFrame(raw_data, columns = ['id', 'sentence', 'sentiment'])
            sentences_to_take=1
            window_size=1
            dg_sentence_graph, avg_tokens_per_sentence_main = self.createGraph(pd_sentence, sentences_to_take, window_size)
            str_output = str_output + "\n"+ self.printGraphTable(dg_sentence_graph)
            #dg_sentence_graph = self.createSentenceGraph(sentence)
            str_output = str_output + "\nCalculating " + calculatesm
            if calculatesm == "csm":
                neg_value = self.getContainmentSimilarity(dg_sentence_graph, dg_neg)
                zero_value = self.getContainmentSimilarity(dg_sentence_graph, dg_zero)
                pos_value = self.getContainmentSimilarity(dg_sentence_graph, dg_pos)
            elif calculatesm == "mcsns":
                neg_value, n_mcs, n_weight = self.getMCSNS(dg_sentence_graph, dg_neg)
                zero_value, z_mcs, z_weight = self.getMCSNS(dg_sentence_graph, dg_zero)
                pos_value, p_mcs, p_weight = self.getMCSNS(dg_sentence_graph, dg_pos)
            elif calculatesm == "mcsues":
                neg_value, n_mcs, n_weight = self.getMCSUES(dg_sentence_graph, dg_neg)
                zero_value, z_mcs, z_weight = self.getMCSUES(dg_sentence_graph, dg_zero)
                pos_value, p_mcs, p_weight = self.getMCSUES(dg_sentence_graph, dg_pos)    
            elif calculatesm == "mcsdes":
                neg_value, n_mcs, n_weight = self.getMCSDES(dg_sentence_graph, dg_neg)
                zero_value, z_mcs, z_weight = self.getMCSDES(dg_sentence_graph, dg_zero)
                pos_value, p_mcs, p_weight = self.getMCSDES(dg_sentence_graph, dg_pos)    
            
            if neg_value == zero_value == pos_value:
                str_output = str_output + "\nNegative MCS: 0" 
                str_output = str_output + "\nNegative Value: 0" 
                
                str_output = str_output + "\nNeutral MCS: " 
                str_output = str_output + " Neutral Value: "
                
                str_output = str_output + "\nPositive MCS: " 
                str_output = str_output + " Positive Value: "
                cal_sentiment = "None"
            else:
                str_output = str_output + "\nNegative MCS: " + self.printGraphTable(n_mcs)
                str_output = str_output + "\nNegative Value: " + str(neg_value)
                
                str_output = str_output + "\nNeutral MCS: " + self.printGraphTable(z_mcs)
                str_output = str_output + " Neutral Value: " + str(zero_value)
                
                str_output = str_output + "\nPositive MCS: " + self.printGraphTable(p_mcs)
                str_output = str_output + " Positive Value: " + str(pos_value)
                
                #print("neg_value:",neg_value)
                #print("zero_value:",zero_value)
                #print("pos_value:",pos_value)
                #if neg_value == zero_value == pos_value == 0:
                #    cal_sentiment = "None"
                #elif neg_value >= zero_value and neg_value >= pos_value:
                if neg_value >= zero_value and neg_value >= pos_value:
                    cal_sentiment = "Negative"
                    if neg_value == zero_value or neg_value == pos_value:
                        two_max_values="Yes"    
                elif zero_value >= neg_value and zero_value >= pos_value:
                    cal_sentiment = "Neutral"   
                    if zero_value == neg_value or zero_value == pos_value:
                        two_max_values="Yes"    
                elif pos_value >= neg_value and pos_value >= zero_value:
                    cal_sentiment = "Positive"
                    if pos_value == neg_value or pos_value == zero_value:
                        two_max_values="Yes"  
                #print("cal_sentiment:",cal_sentiment)
                
                str_output = str_output + "\nCalculated Sentiment : " + cal_sentiment
                str_output = str_output + " two_max_values : " + two_max_values
                str_output = str_output + "\nActual Sentiment : "+actual_sentiment
                
                if cal_sentiment == actual_sentiment:
                    accuracy=1
                    accurate += 1
                    if two_max_values=="Yes"  :
                        duplicate += 1
                else:
                    accuracy=0
                    not_accurate += 1
                
                #row_dict["SentenceID"]=sentence_id
                #row_dict["Sentence"]=sentence
                #row_dict["NegativeScore"]=neg_value
                #row_dict["NeutralScore"]=zero_value
                #row_dict["PositiveScore"]=pos_value
                #row_dict["AutomatedSentiment"]=cal_sentiment
                #row_dict["ActualSentiment"]=actual_sentiment
                #row_dict["Accuracy"]=accuracy
                
                # Not calling this as this as not using it
                #SentiWordnetSentiment, pos_score, neg_score, neutral_score = tu.getSentiWordnetScore(sentence)
                SentiWordnetSentiment = pos_score = neg_score = neutral_score = 0
                
                # take the SentiWordnetSentiment value if 2 terms are giving similar sentiments
                if two_max_values=="Yes":
                    cal_sentiment=SentiWordnetSentiment
                
                row_list.append([sentence_id,sentence,neg_value,zero_value,pos_value,actual_sentiment,cal_sentiment,accuracy,two_max_values, SentiWordnetSentiment, pos_score, neg_score, neutral_score])
                
                #if ctr == 5:
                #    break
        
        
        str_output = str_output + "\n----------SUMMARY ---------------\n"
        print("Total correct predictions including Two equal scores:",accurate)
        str_output = str_output + "\nTotal records: " + str(accurate + not_accurate)
        str_output = str_output + "\nTotal correct predictions including Two equal scores: " + str(accurate)
        print("Total two equal scores:",duplicate)
        str_output = str_output + "\nTotal two equal scores: " + str(duplicate)
        print("Total correct predictions with one highest score:",accurate - duplicate)
        str_output = str_output + "\nTotal correct predictions with one highest score: " + str(accurate - duplicate)
        print("Total Incorrect Predictions:",not_accurate)
        str_output = str_output + "\nTotal Incorrect Predictions: " + str(not_accurate)
        print("Accuracy %: ",(accurate - duplicate) / (accurate + not_accurate))
        str_output = str_output + "\nAccuracy%: " + str((accurate - duplicate) / (accurate + not_accurate))
        
        column_names = ['SentenceID','Sentence', 'NegativeScore',"NeutralScore","PositiveScore","ActualSentiment","AutomatedSentiment","Accuracy","TwoMaxValues","SentiWordnetSentiment", "Positive-Senti", "Negative-Senti", "Neutral-Senti"]            
        
        bu = BasicFunctions()
        #if run_type == "validation":
        
            
        if str_called_from == "validation":
            excel_name_full = calculatesm +".validationfull.xlsx"
            object_name_full = calculatesm +".validationfull.pkl"
            
            excel_name_correct = calculatesm +".validationcorrect.xlsx"
            object_name_correct = calculatesm +".validationcorrect.pkl"
            
            newDF = pd.DataFrame(data=row_list, columns = column_names)
            # Get only those records where the results are correct
            newDF_correct = newDF.loc[(newDF['TwoMaxValues'] == 'No') & newDF['Accuracy'] == 1]
            #print(newDF)
            #writer = pd.ExcelWriter('..\\data\\intermediate\\Validation_MCS.xlsx')
            writer = pd.ExcelWriter(config_param["data.intermediatefolder"] + excel_name_full)
            newDF.to_excel(writer, 'DataFrame')
            writer.save()
            bu.saveObject(config_param["data.intermediatefolder"] + object_name_full, newDF)
            writer = pd.ExcelWriter(config_param["data.intermediatefolder"] + excel_name_correct)
            newDF_correct.to_excel(writer, 'DataFrame')
            writer.save()
            bu.saveObject(config_param["data.intermediatefolder"] + object_name_correct, newDF_correct)
            
        if str_called_from == "test":
            excel_name_test = calculatesm +".test.xlsx"
            object_name_test = calculatesm +".test.pkl"
        
            TestDF = pd.DataFrame(data=row_list, columns = column_names)
            writer = pd.ExcelWriter(config_param["data.intermediatefolder"] + excel_name_test)
            TestDF.to_excel(writer, 'DataFrame')
            writer.save()
            bu.saveObject(config_param["data.intermediatefolder"] + object_name_test, TestDF)
        
        str_output = str_output + "\n############ Ended Computing "+calculatesm+" similarity matrix for "+str_called_from+" data##################\n"        
        calculatesm = config_param["execute.calculatesm"]
        windowsize = config_param["creategraph.window_size"]
        output_file_name = config_param["data.outputfolder"] + calculatesm + "_w" + windowsize + "_"+"output.txt"
        bu.saveTextFile(output_file_name, str_output)
            
        print("### Compute CSM ### - End")
        #return newDF_correct        
        
