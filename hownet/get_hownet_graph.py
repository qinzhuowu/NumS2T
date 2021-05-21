#coding:utf-8
import numpy as np
import sys

import codecs
import collections
import random
from os.path import isfile, join
from hownet import How_Similarity
reload(sys)
sys.setdefaultencoding( "utf-8" )

how1=How_Similarity()
def input_data():
    encode_train_dataset = codecs.open('..//..//..//GCN_Data//jieba//train_recut.enc', "r", encoding="UTF-8").readlines()
    
    vocab=how1.vocab
    for line in encode_train_dataset:
        index_list=line.strip().split()
        line_vocab_list=[]
        for word in index_list:
            if how1.getSememeByZh(word)!=None:
                line_vocab_list.append(word)
        for i in range(0,len(line_vocab_list)):
            for j in range(0,len(line_vocab_list)):
                distance=how1.calcSememeDistance(line_vocab_list[i],line_vocab_list[j])
                if distance<20:
                    if distance>2:
                        print(line_vocab_list[i])
                        print(line_vocab_list[j])
                        print(distance)

input_data()
