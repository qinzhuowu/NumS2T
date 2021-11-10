# coding: utf-8
import time
import json
import copy
from copy import deepcopy
import re
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )


vocab_5fold=[]
UNK_vocab=[]
UNK2word_vocab={}
input1=open("vocab_5fold","r").readlines()
for word in input1:
    vocab_5fold.append(word.strip())

input2=open("UNK_vocab","r").readlines()
for word in input2:
    UNK_vocab.append(word.strip())

for word in UNK_vocab:
	if len(word)>1:
		for i in range(1,len(word)):
			word1=word[0:i]
			word2=word[i:]
			if word1 in vocab_5fold and word2 in vocab_5fold:
				list1=[word1,word2]
				UNK2word_vocab[word]=list1

output=open("UNK2word_vocab","w")
for word in UNK2word_vocab:
    output.write(word+"###"+" ".join(UNK2word_vocab[word])+"\n")