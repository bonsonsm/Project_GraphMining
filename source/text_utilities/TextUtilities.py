# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from collections import OrderedDict

class TextFunctions:
        
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
            train_percent=.7, validate_percent=0, seed=None):
        
        print("### Pre Processing ### - Start")
        pd_preprocess_df = pd.read_excel(full_path, sheet_name, header)
        
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: x if type(x)!=str else x.lower())
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].str.split()
        wnl = WordNetLemmatizer()
        # For Verbs
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: list(OrderedDict.fromkeys([wnl.lemmatize(item,'v') for item in x])))
        # For Nouns        
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: list(OrderedDict.fromkeys([wnl.lemmatize(item,'n') for item in x])))
        pd_preprocess_df.loc[:,X_name] = pd_preprocess_df[X_name].apply(lambda x: ' '.join(x))
        
        print("Total Records:",len(pd_preprocess_df.index))
        pd_negative = pd_preprocess_df[(pd_preprocess_df[y_name]==0) | (pd_preprocess_df[y_name]==1)]
        pd_negative.loc[:,y_name] = -1
        print("Total Negative Records:",len(pd_negative.index))
        pd_neutral= pd_preprocess_df[pd_preprocess_df[y_name]==2]
        pd_neutral.loc[:,y_name] = 0
        print("Total Neutral Records:",len(pd_neutral.index))
        pd_positive = pd_preprocess_df[(pd_preprocess_df[y_name]==3) | (pd_preprocess_df[y_name]==4)]
        pd_positive.loc[:,y_name] = 1

        print("Total Positive Records:",len(pd_positive.index))
        
        print("######## Dividing Negative Set ###########")
        #pd_negative_train, pd_negative_validate, pd_negative_test = self.train_validate_test_split(pd_negative, train_percent=.7, validate_percent=0, seed=None)
        pd_negative_train, pd_negative_validate, pd_negative_test = self.train_validate_test_split(pd_negative, train_percent, validate_percent, seed=None)
        print("Total pd_negative_train Records:",len(pd_negative_train.index))        
        print("Total pd_negative_validate Records:",len(pd_negative_validate.index))
        print("Total pd_negative_test Records:",len(pd_negative_test.index))

        print("######## Dividing Neutral Set ###########")
        #pd_neutral_train, pd_neutral_validate, pd_neutral_test = self.train_validate_test_split(pd_neutral, train_percent=.7, validate_percent=0, seed=None)
        pd_neutral_train, pd_neutral_validate, pd_neutral_test = self.train_validate_test_split(pd_neutral, train_percent, validate_percent, seed=None)
        print("Total pd_neutral_train Records:",len(pd_neutral_train.index))        
        print("Total pd_neutral_validate Records:",len(pd_neutral_validate.index))
        print("Total pd_neutral_test Records:",len(pd_neutral_test.index))

        print("######## Dividing Positive Set ###########")
        #pd_positive_train, pd_positive_validate, pd_positive_test = self.train_validate_test_split(pd_positive, train_percent=.7, validate_percent=0, seed=None)
        pd_positive_train, pd_positive_validate, pd_positive_test = self.train_validate_test_split(pd_positive, train_percent, validate_percent, seed=None)
        print("Total pd_positive_train Records:",len(pd_positive_train.index))        
        print("Total pd_positive_validate Records:",len(pd_positive_validate.index))
        print("Total pd_positive_test Records:",len(pd_positive_test.index))
        
       
        print("### Pre Processing ### - End")
        #return pd_preprocess_df, df_first_half, df_second_half, pd_first_negative, pd_first_neutral, pd_first_positive
        return pd_preprocess_df, pd_negative_train, pd_negative_validate, pd_negative_test, \
            pd_neutral_train, pd_neutral_validate, pd_neutral_test, \
            pd_positive_train, pd_positive_validate, pd_positive_test