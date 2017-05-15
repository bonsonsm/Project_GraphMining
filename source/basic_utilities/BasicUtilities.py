# -*- coding: utf-8 -*-
import configparser
import pickle

import pandas as pd

class BasicFunctions:
    
    def readConfig(self, path):
        config = configparser.RawConfigParser()
        config.read('..\\config\\config.ini')
        config_param ={}
        #InstallationSection
        config_param["base_path"]=config.get('InstallationSection', 'base_path')
        #ExecutionSection
        config_param["execute.temp"]=config.get('ExecutionSection', 'execute.temp')
        config_param["execute.calculatesm"]=config.get('ExecutionSection', 'execute.calculatesm')
        config_param["execute.preprocessdata"]=config.get('ExecutionSection', 'execute.preprocessdata')
        config_param["execute.creategraphs"]=config.get('ExecutionSection', 'execute.creategraphs')
        config_param["execute.computecsm_validationdata"]=config.get('ExecutionSection', 'execute.computecsm_validationdata')
        config_param["execute.computecsm_testdata"]=config.get('ExecutionSection', 'execute.computecsm_testdata')
        config_param["execute.csm_validationdata_model"]=config.get('ExecutionSection', 'execute.csm_validationdata_model')
        #InputSection
        config_param["data.inputtrain_percent"]=config.get('InputSection', 'data.inputtrain_percent')
        config_param["data.inputvalidate_percent"]=config.get('InputSection', 'data.inputvalidate_percent')
        config_param["data.inputX_name"]=config.get('InputSection', 'data.inputX_name')
        config_param["data.inputy_name"]=config.get('InputSection', 'data.inputy_name')
        
        config_param["data.inputfolder"]=config.get('InputSection', 'data.inputfolder')
        config_param["data.intermediatefolder"]=config.get('InputSection', 'data.intermediatefolder')
        config_param["data.outputfolder"]=config.get('InputSection', 'data.outputfolder')
        
        config_param["data.inputfilename"]=config.get('InputSection', 'data.inputfilename')
        config_param["data.inputsheet_name"]=config.get('InputSection', 'data.inputsheet_name')
        config_param["data.inputheader"]=config.get('InputSection', 'data.inputheader')

        config_param["creategraph.sentences_to_take"]=config.get('InputSection', 'creategraph.sentences_to_take')
        config_param["creategraph.window_size"]=config.get('InputSection', 'creategraph.window_size')
        #config_param["computecsm.runtype"]=config.get('InputSection', 'computecsm.runtype')

        print(config_param)
        return config_param    
        
    def saveObject(self, file_name, obj):
        #print(" -- Saving Object -- ", file_name)
        output = open(file_name, 'wb')
        # -1 = Pickle the list using the highest protocol available
        pickle.dump(obj, output, -1)
        output.close()
        
    def loadObject(self, file_name):
        #print(" -- Loading Object -- ", file_name)
        pkl_file = open(file_name, 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()
        return obj
        
    def saveTextFile(self, file_name, strValue):
        with open(file_name, "a") as f:
            f.write(strValue)
        f.close()
        #file = open(file_name,"w") 
        #file.write(strValue) 
        #file.close()
        
    def saveExcelFile(self, file_name, df, config_param):
        file_name = config_param["data.intermediatefolder"] + file_name
        writer = pd.ExcelWriter(file_name)
        df.to_excel(writer, 'DataFrame')
        writer.save()
        
