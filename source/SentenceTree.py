# -*- coding: utf-8 -*-

from nltk.parse.stanford import StanfordDependencyParser
import os
from nltk.parse.stanford import StanfordParser
#from nltk.parse import stanford
#import StanfordDependencies
import nltk
from nltk.tree import *

java_path = "C:/Program Files/Java/jdk1.8.0_102/bin" # replace this
os.environ['JAVAHOME'] = java_path
os.environ['STANFORD_PARSER'] = 'D:/Deepa/StanfordJars/stanford-parser-full-2015-04-20/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'D:/Deepa/StanfordJars/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'
dep_parser=StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
path_to_jar = 'D:/Deepa/StanfordJars/stanford-parser-full-2015-04-20/stanford-parser.jar'
path_to_models_jar =  'D:/Deepa/StanfordJars/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'
#works
#print([parse.tree() for parse in dep_parser.raw_parse('I shot an elephant in my sleep')])
print('--------------------------------------------')


#print(parsed_Sentence)
class SentenceTree:
    def extractwordsfromtree(self,parsed_tree):
        leaf_values = parsed_tree[0].leaves()
#        for line in parsed_tree:
#           print('Line :', type(line))
#           leaf_values = line.leaves()
#           print(leaf_values)          
           #line.draw()#GUI
        return ' '.join(leaf_values)
           
    def getsentencetree(self, sentence):
        parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        parsed_Sentence= list(parser.raw_parse(sentence))
        return parsed_Sentence
    def extarctphrasefromtree(self, parsed_tree,phrase):
        myPhrases = []
        print(type(parsed_tree[0]))
        my_tree= parsed_tree[0]
        print(my_tree.label())
        if (my_tree.label()==phrase):
            myPhrases.append( my_tree.copy(True) )
        for child in my_tree:
            if (type(child) is nltk.tree):
                list_of_phrases = self.extarctphrasefromtree(child, phrase)
                if(len(list_of_phrases) > 0):
                    myPhrases.extend(list_of_phrases)
        return myPhrases
        
if __name__=='__main__':
    objSentenceTree =SentenceTree()
    parsed_Sentence = objSentenceTree.getsentencetree("I shot an elephant in my sleep")
    leafValue = objSentenceTree.extractwordsfromtree(parsed_Sentence)
    print(leafValue)
    #objSentenceTree.extarctphrasefromtree(parsed_Sentence,'NP')

#dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

#result = dep_parser.raw_parse('I shot an elephant in my sleep')
#dep = result.__next__() 
#dep_parserList = list(dep.triples())
#print(dep_parserList)
