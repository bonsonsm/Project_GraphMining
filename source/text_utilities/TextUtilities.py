# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import configparser

from nltk.stem import WordNetLemmatizer
from collections import OrderedDict

import nltk
#from nltk.stem import *
from nltk.corpus import sentiwordnet as swn

from basic_utilities.BasicUtilities import BasicFunctions

class TextFunctions:
    
    def getWordScore(self, text, positive_words, negative_words):
        pos = neg = 0
        for word in text.split():
            print(word)
            if word in positive_words:
                pos = pos +1
                print("Positive")
            if word in negative_words:
                neg = neg +1
                print("Negative")
        print("Pos",pos)
        print("Neg",neg)
                
    def getSentiWordnetScore(self, text):
        tokens=nltk.word_tokenize(text) #for tokenization, row is line of a file in which tweets are saved.
        tagged_text=nltk.pos_tag(tokens) #for POSTagging
        pos_score = neg_score = token_count = obj_score = pos_words = neg_words = obj_words = 0
        for word, tag in tagged_text:
            print("Word:",word)
            print(tag)
            ss_set = None
            if 'JJ' in tag and swn.senti_synsets(word, 'a'):
                ss_set = list(swn.senti_synsets(word, 'a'))
            elif 'VB' in tag and swn.senti_synsets(word, 'v'):
                ss_set = list(swn.senti_synsets(word, 'v'))
            elif 'NN' in tag and swn.senti_synsets(word, 'n'):
                ss_set = list(swn.senti_synsets(word, 'n'))
            elif 'RB' in tag and swn.senti_synsets(word, 'r'):
                ss_set = list(swn.senti_synsets(word, 'r'))
            # if senti-synset is found        
            if ss_set:
                # add scores for all found synsets
                p_score = ss_set[0].pos_score()
                n_score = ss_set[0].neg_score()
                o_score = ss_set[0].obj_score()
                
                if p_score > 0:
                    pos_words += 1
                    pos_score += p_score
                if n_score > 0:
                    neg_words += 1
                    neg_score += n_score
                if o_score > 0:
                    obj_words += 1
                    obj_score += o_score
                print("pos_score", pos_score)
                print("pos_words", pos_words)
                print("neg_score", neg_score)
                print("neg_words", neg_words)
        
                #pos_score += ss_set[0].pos_score()
                #neg_score += ss_set[0].neg_score()
                #obj_score += ss_set[0].obj_score()
                #print(obj_score)
                token_count += 1
        if pos_words>0:
            pos_score = pos_score/pos_words
        else:
            pos_score =0
        if neg_score>0:
            neg_score = neg_score/neg_words
        else:
            neg_score = 0
        
        print(pos_score)
        print(neg_score)
                
        #print("Positive = ",pos_score)
        #print("Negative = ",neg_score)
        #print("Object = ",obj_score)
        return_value = ""
        if pos_score > neg_score:
            return_value = "Positive"
        else:
            return_value = "Negative"
                   
        return return_value, pos_score, neg_score, obj_score
        
    def createWordExpandDictionary(self):
        config = configparser.RawConfigParser()
        #config.read('..\\WordExpandDictionary.properties')
        with open(r'..\\config\\WordExpandDictionary.properties') as f:
        #with open(r'..\\..\\config\\WordExpandDictionary.properties') as f:
            config.read_string('[config]\n' + f.read())
        #for k, v in config['config'].items():
        #    print(k, v)
        #print(config['config']["-lrb-"])    
        #print(type(config['config']))
        return config['config']

    def createPositiveNegativeWordsDictionary(self):
        f_pos = open('..\\..\\config\\positive-words.txt', 'r')
        pos_list = f_pos.readlines()
        pos_list = [i.strip() for i in pos_list]
        f_neg = open('..\\..\\config\\negative-words.txt', 'r')
        neg_list = f_neg.readlines()
        neg_list = [i.strip() for i in neg_list]
        return pos_list, neg_list

    def replace_all(self,text, dic):
        #print(text)
        for i, j in dic.items():
            if "&nbsp" in i:
                #if key contains space then space needs to be added in value
                #print(i)
                i=i.replace("&nbsp"," ")
                #print(i)
                #replace stopwords from dictionary
                if j=="&nbsp":
                    j=" "
                else:
                    j=j+" "
            else:
                #replace stopwords from dictionary
                if j=="&nbsp":
                    j=" "
            #replace with expanded words from dictionary
            text = text.replace(i, j)
            #remove extra spaces 
            text = text.replace("  "," ")
        return text

    def testWordDic(self, pd_preprocess_df):
        wordDic=self.createWordExpandDictionary()
        print("wordDic:", wordDic["'ll"])
        pd_preprocess_df.loc[:,'sentence'] = pd_preprocess_df['sentence'].apply(lambda x: self.replace_all(x,wordDic))    
        print(pd_preprocess_df.iloc[1,1])
        return pd_preprocess_df.iloc[1,1]
        
            
    def train_validate_test_split(self, df, train_percent=.4, \
            validate_percent=.4, seed=None):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = df.ix[perm[:train_end]]
        validate = df.ix[perm[train_end:validate_end]]
        test = df.ix[perm[validate_end:]]
        return train, validate, test
    
    def preprocessing(self,full_path, sheet_name, header, X_name, y_name, \
            train_percent, validate_percent, seed_val, config_param):
        print("### Pre Processing ### - Start")
        str_output = ""
        wordDic=self.createWordExpandDictionary()
        #print("wordDic:", wordDic)
        pd_preprocess_df = pd.read_excel(full_path, sheet_name, header)
        #print(pd_preprocess_df.loc[0,X_name])
        #print(pd_preprocess_df.loc[9,X_name])
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: x if type(x)!=str else x.lower())
        #print(pd_preprocess_df.loc[0,X_name])
        #print(pd_preprocess_df.loc[7, X_name])
        #print(pd_preprocess_df.loc[9,X_name])
        #For Dictionary and stopwords
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: self.replace_all(x,wordDic))    
        
        # putting space between last word and full stop or comma
        
        #print(pd_preprocess_df.loc[0,X_name])
        #print(pd_preprocess_df.loc[9,X_name])
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].str.split()
        #print(pd_preprocess_df.loc[0,X_name])
        #print(pd_preprocess_df.loc[7, X_name])
        #print(pd_preprocess_df.loc[9,X_name])
        wnl = WordNetLemmatizer()
        #print("Lemetizing for Verbs")
        # For Verbs
        #This will remove repeating words in the sentence
        #pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: list(OrderedDict.fromkeys([wnl.lemmatize(item,'v') for item in x])))
        #This will NOT remove repeating words in the sentence
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: [wnl.lemmatize(item,'v') for item in x])
        #print(pd_preprocess_df.loc[0,X_name])
        #print(pd_preprocess_df.loc[7, X_name])
        #print(pd_preprocess_df.loc[9,X_name])
        # For Nouns        
        #print("Lemetizing for Nouns")
        #This will remove repeating words in the sentence
        #pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: list(OrderedDict.fromkeys([wnl.lemmatize(item,'n') for item in x])))
        #This will NOT remove repeating words in the sentence
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: [wnl.lemmatize(item,'n') for item in x])
        #print(pd_preprocess_df.loc[0,X_name])
        #print(pd_preprocess_df.loc[7, X_name])
        #print(pd_preprocess_df.loc[9,X_name])
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: ' '.join(x))
        #print(pd_preprocess_df.loc[0,X_name])
        #print(pd_preprocess_df.loc[7, X_name])
        #print(pd_preprocess_df.loc[9,X_name])
        str_output = str_output+"\nTotal Records:" +str(len(pd_preprocess_df.index))
        print("Total Records:",len(pd_preprocess_df.index))
        pd_negative = pd_preprocess_df[(pd_preprocess_df[y_name]==0) | (pd_preprocess_df[y_name]==1)]
        pd_negative.loc[:,y_name] = -1
        str_output = str_output+"\nTotal Negative Records:" +str(len(pd_negative.index))
        print("Total Negative Records:",len(pd_negative.index))
        pd_neutral= pd_preprocess_df[pd_preprocess_df[y_name]==2]
        pd_neutral.loc[:,y_name] = 0
        str_output = str_output+"\nTotal Neutral Records:" +str(len(pd_neutral.index))
        print("Total Neutral Records:",len(pd_neutral.index))
        pd_positive = pd_preprocess_df[(pd_preprocess_df[y_name]==3) | (pd_preprocess_df[y_name]==4)]
        pd_positive.loc[:,y_name] = 1
        str_output = str_output+"\nTotal Positive Records:" +str(len(pd_positive.index))+"\n"
        print("Total Positive Records:",len(pd_positive.index))
        print("######## Dividing Negative Set ###########")
        #pd_negative_train, pd_negative_validate, pd_negative_test = self.train_validate_test_split(pd_negative, train_percent=.7, validate_percent=0, seed=None)
        pd_negative_train, pd_negative_validate, pd_negative_test = self.train_validate_test_split(pd_negative, train_percent, validate_percent, seed=seed_val)
        #print("Total pd_negative_validate Records:",len(pd_negative_validate.index))
        #print("Total pd_negative_test Records:",len(pd_negative_test.index))
        print("######## Dividing Neutral Set ###########")
        #pd_neutral_train, pd_neutral_validate, pd_neutral_test = self.train_validate_test_split(pd_neutral, train_percent=.7, validate_percent=0, seed=None)
        pd_neutral_train, pd_neutral_validate, pd_neutral_test = self.train_validate_test_split(pd_neutral, train_percent, validate_percent, seed=seed_val)
        #print("Total pd_neutral_validate Records:",len(pd_neutral_validate.index))
        #print("Total pd_neutral_test Records:",len(pd_neutral_test.index))
        print("######## Dividing Positive Set ###########")
        #pd_positive_train, pd_positive_validate, pd_positive_test = self.train_validate_test_split(pd_positive, train_percent=.7, validate_percent=0, seed=None)
        pd_positive_train, pd_positive_validate, pd_positive_test = self.train_validate_test_split(pd_positive, train_percent, validate_percent, seed=seed_val)
        
        str_output = str_output+"\nTotal Train Records:" +str(len(pd_negative_train.index) + len(pd_neutral_train.index) + len(pd_positive_train.index))
        
        str_output = str_output+"\npd_negative_train Records:" +str(len(pd_negative_train.index))
        print("Total pd_negative_train Records:",len(pd_negative_train.index))        
        
        str_output = str_output+"\npd_neutral_train Records:" +str(len(pd_neutral_train.index))
        print("Total pd_neutral_train Records:",len(pd_neutral_train.index))        
        
        str_output = str_output+"\npd_positive_train Records:" +str(len(pd_positive_train.index))+"\n"
        print("\npd_positive_train Records:",len(pd_positive_train.index))        
        
        #print("Total pd_positive_validate Records:",len(pd_positive_validate.index))
        #print("Total pd_positive_test Records:",len(pd_positive_test.index))
        
        str_output = str_output+"\nTotal Validate Records:" +str(len(pd_negative_validate.index) + len(pd_neutral_validate.index) + len(pd_positive_validate.index))
        
        str_output = str_output+"\npd_negative_validate Records:" +str(len(pd_negative_validate.index))
        str_output = str_output+"\npd_neutral_validate Records:" +str(len(pd_neutral_validate.index))
        str_output = str_output+"\npd_positive_validate Records:" +str(len(pd_positive_validate.index))+"\n"
        
        str_output = str_output+"\nTotal Test Records:" +str(len(pd_negative_test.index) + len(pd_neutral_test.index) + len(pd_positive_test.index))
        str_output = str_output+"\npd_negative_test Records:" +str(len(pd_negative_test.index))
        str_output = str_output+"\npd_neutral_test Records:" +str(len(pd_neutral_test.index))
        str_output = str_output+"\npd_positive_test Records:" +str(len(pd_positive_test.index))+"\n"
        
        
        print("### Pre Processing ### - End")
        bu = BasicFunctions()
        calculatesm = config_param["execute.calculatesm"]
        windowsize = config_param["creategraph.window_size"]
        output_file_name = config_param["data.outputfolder"] + calculatesm + "_w" + windowsize + "_"+"output.txt"
        bu.saveTextFile(output_file_name, str_output)
        
        #return pd_preprocess_df, df_first_half, df_second_half, pd_first_negative, pd_first_neutral, pd_first_positive
        return pd_preprocess_df, pd_negative_train, pd_negative_validate, pd_negative_test, \
            pd_neutral_train, pd_neutral_validate, pd_neutral_test, \
            pd_positive_train, pd_positive_validate, pd_positive_test
            
if __name__ == '__main__':
    tu = TextFunctions()
    #tu.createWordExpandDictionary()
    raw_data = {'id': ['1','2'], 'sentence':["What is 'll good ... for the gardner is also good for the goose","play like the old disease of week small screen melodrama ."], 'sentiment': [0,2]}
    pd_neutral_train = pd.DataFrame(raw_data, columns = ['id', 'sentence', 'sentiment'])
    #text = tu.testWordDic(pd_neutral_train)
    #sentiment = tu.getSentiWordnetScore(text)
    #print(sentiment)
    
    pos_dic, neg_dic = tu.createPositiveNegativeWordsDictionary()
    tu.getWordScore(pd_neutral_train.iloc[1,1], pos_dic, neg_dic)
    
