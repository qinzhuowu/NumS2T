# coding: utf-8
from __future__ import division 
import random
import numpy as np
import json
import copy
#import jieba
import codecs
import re
import sys
from expressions_transfer import *
from parameter import *
#reload(sys)
#sys.setdefaultencoding( "utf-8" )


#Max_Question_len=120
#Max_Expression_len=50

#dataset="APE"

def build_vocab():
    word_lst = []  
    key_list=[] 
    i=0
    for l in open('data/ape/train.ape.json'):
        if i%1000==0:
            print(i)
        i+=1
        l = json.loads(l)
        question = l['original_text']

        tags = jieba.lcut(question) #jieba分词  
        for t in tags:  
            if len(t) > 1:
                word_lst.append(t) 

  
    word_dict= {}  
    with open("data/apewordVocab.txt",'w') as wf2: #打开文件  
  
        for item in word_lst:  
            if item not in word_dict: #统计数量  
                word_dict[item] = 1  
            else:  
                word_dict[item] += 1  
  
        orderList=list(word_dict.values())  
        orderList.sort(reverse=True)  
        max_order_list=orderList[0:5000]
        # print orderList  
        for i in range(len(max_order_list)):  
            for key in word_dict:  
                if word_dict[key]==max_order_list[i]:  
                    wf2.write(key+' '+str(word_dict[key])+'\n') #写入txt文档  
                    key_list.append(key)  
                    word_dict[key]=0  


def load_data(filename,divide=1):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    print(filename)
    question_num=0
    not_equal=0
    cannot_eval=0
    word_lst=[]
    word_space=[]
    word_dense=[]
    for line in open('data/apewordVocab.txt'):
        key=line.strip().split()[0]
        if USE_APE_word:
            if len(key)>1:
                word_lst.append(key)
        if USE_APE_char:
            if len(key)>5:
                word_lst.append(key)
    word_lst.sort(key = lambda i:len(i),reverse=True) 
    for key in word_lst:
        value_list=[]
        for char in range(0,len(key),1):
            value_list.append(key[char:char+1])
        
        value=" ".join(value_list)

        word_space.append(" "+value+" ")
        word_dense.append(" "+key+" ")

    for l in open(filename):
        question_num+=1
        #if question_num%50000==0:
        #    print(question_num)
        if question_num%divide==0:
            l = json.loads(l)
            question, equation, answer = l['segmented_text'].strip(), l['equation'], l['ans']
            flag=0
            #if "(".encode("UTF-8") in question or " / ".encode("UTF-8") in question or "%".encode("UTF-8") in question:
            #    print(question)
            #    print(equation)s
            #    print(answer)
            #    flag=1
            # 处理带分数
            question = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', question)
            equation = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', equation)
            equation = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', equation)
            #equation = re.sub('(\d+) \( (\d+ / \d+) \)', '\\1(\\2/\\3)', equation)
            #answer = re.sub('(\d+) \( (\d+ / \d+) \)', '\\1(\\2/\\3)', answer)
            equation = re.sub('(\d+) \(', '\\1(', equation)
            answer = re.sub('(\d+) \(', '\\1(', answer)
            # 分数去括号
            #question = re.sub('\((\d+/\d+)\)', '\\1', question)
            # 分数合并
            question = re.sub('\( (\d+) / (\d+) \)', '(\\1/\\2)', question)
            equation = re.sub('\( (\d+) / (\d+) \)', '(\\1/\\2)', equation)
            
            # 分数加括号
            #question = re.sub(' (\d+) / (\d+) ', ' (\\1/\\2) ', question)
            #equation = re.sub(' (\d+) / (\d+) ', ' (\\1/\\2) ', equation)
            # 处理百分数
            question = re.sub('([\.\d]+)%', '(\\1/100)', question)
            equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
            answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
            # 冒号转除号、剩余百分号处理
            question = question.replace('%', ' / 100')
            equation = equation.replace(':', '/').replace('%', '/100')
            answer = answer.replace(':', '/').replace('%', '/100')
            equation = equation.replace('"千米/小时"', '')
            #if flag==1:
            #    print(question)
            #    print(equation)
            #    print(answer)
            #    flag=1
            if equation[:2] == 'x=':
                equation = equation[2:]

            idx_ = 0
            question=" "+question+" "
            for idx_ in range(len(word_dense)):
                if word_space[idx_] in question:
                    question=question.replace(word_space[idx_],word_dense[idx_])
            question=question.strip()
            try:
                if is_equal(eval(equation), eval(answer)):
                    D.append((question, equation, answer))
                else:
                    #print("not equal")
                    #print(question)
                    #print(equation)
                    #print(eval(equation))
                    #print(answer)
                    #print(eval(answer))
                    not_equal+=1
            except:
                #print(question)
                #print(equation)
                D.append((question, equation, answer))
                cannot_eval+=1
                continue
    with open(filename+"clear",'w') as wf2:
        for item in D:
            wf2.write(item[0]+"\n")
            wf2.write(item[1]+"\n")
    print(question_num)
    print(not_equal)
    print(cannot_eval)
    print(len(D))
    return D
def load_Math23K_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename,'r')
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
                
            question, equation, answer = data_d['segmented_text'].strip(), data_d['equation'], data_d['ans']
            
            equation = equation.replace('"千米/小时"', '')
            if equation[:2] == 'x=':
                equation = equation[2:]
            js = ""
            #try:
            #    if is_equal(eval(equation), eval(answer)):
            data.append((question, equation, answer))
            #    else:
            #        print(equation)
            #        print(answer)
            #        print(eval(equation))
            #except:
            #    continue
    return data


def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b

def remove_bucket(equation):
    """去掉冗余的括号
    """
    l_buckets, buckets = [], []
    for i, c in enumerate(equation):
        if c == '(':
            l_buckets.append(i)
        elif c == ')':
            buckets.append((l_buckets.pop(), i))
    eval_equation = eval(equation)
    for l, r in buckets:
        new_equation = '%s %s %s' % (
            equation[:l], equation[l + 1:r], equation[r + 1:]
        )
        try:
            if is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                equation = new_equation
        except:
            pass
    return equation.replace(' ', '')


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|\d*\(?\d*\.?\d+/\d+\)?\d*")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    count_empty=0

    UNK2word_vocab={}
    input1=open("data//UNK2word_vocab","r").readlines()
    for word in input1:
        UNK2word_vocab[word.strip().split("###")[0]]=word.strip().split("###")[1]
    count_too_lang=0
    exp_too_lang=0
    for d in data:
        nums = []
        input_seq = []
        seg_line = d[0].strip()
        for UNK_word in UNK2word_vocab:
            if UNK_word in seg_line:
                seg_line=seg_line.replace(UNK_word,UNK2word_vocab[UNK_word])
        seg=seg_line.split(" ")
        equations = d[1]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                if len(s)>0:
                    input_seq.append(s)
                else:
                    count_empty=count_empty+1
        if copy_nums < len(nums):
            copy_nums = len(nums)


        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)

        if len(input_seq) > Max_Question_len :
            count_too_lang+=1
            continue

        '''
            for idx_ in range(len(num_pos)-1,-1,-1):
                if num_pos[idx_]>Max_Question_len:
                    num_pos.pop(idx_)
                    nums.pop(idx_)
        input_seq=input_seq[0:Max_Question_len]
        '''
        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*|\d*\(\d+\.\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) >= 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
                elif n[0]=='(' and n[-1] ==')':
                    n_1=n[1:-1]
                    if n_1 in st:
                        p_start = st.find(n_1)
                        p_end = p_start + len(n_1)
                        if p_start > 0:
                            res += seg_and_tag(st[:p_start])
                        if nums.count(n) >= 1:
                            res.append("N"+str(nums.index(n)))
                        else:
                            res.append(n)
                        if p_end < len(st):
                            res += seg_and_tag(st[p_end:])
                        return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        if USE_just_char_number==True:
            realnum_input=[]
            realnum_pos=[]
            prob_start=0
            for i in range(len(num_pos)):
                num_index=num_pos[i]
                realnum_input.extend(input_seq[prob_start:num_index])
                realnum_pos.append(len(realnum_input))
                prob_start=num_index+1
                num_word=nums[i]
                for num_char in num_word:
                    realnum_input.append(num_char)
            realnum_input.extend(input_seq[prob_start:])
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        if len(out_seq) >0:
            if len(out_seq)> Max_Expression_len:
                exp_too_lang+=1
            else:
                if USE_just_char_number==True:
                    pairs.append((realnum_input, out_seq, nums, realnum_pos))
                else:
                    pairs.append((input_seq, out_seq, nums, num_pos))
    print("count_empty")
    print(count_empty)
    print("data_set_size is %d, num of exp>60  is %d,about %.4f" %(len(pairs),exp_too_lang,float(exp_too_lang)/len(pairs)))
    print("data_set_size is %d, num of problem>150  is %d,about %.4f" %(len(pairs),count_too_lang,float(count_too_lang)/len(pairs)))

    if dataset=="APE":
        orderList=list(generate_nums_dict.values())  
        orderList.sort(reverse=True)  
        max_order_list=orderList[0:10]
        min_generate_vocab_appear=max_order_list[-1]
        temp_g = []
        for g in generate_nums:
            if generate_nums_dict[g] >= min_generate_vocab_appear:
                temp_g.append(g)
        print("generate_num size is %d" %(len(temp_g)))
        print("min_generate_vocab_appear times is %d" %(min_generate_vocab_appear))
    else:
        temp_g = []
        for g in generate_nums:
            if generate_nums_dict[g] >= 5:
                temp_g.append(g)
    return pairs, temp_g, copy_nums


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count,outlang_vocab):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
        self.index2word =outlang_vocab+["PAD", "NUM", "UNK"] + self.index2word
        input_lang_vocab=[]
        for word_ in self.index2word:
            if word_ not in input_lang_vocab:
                input_lang_vocab.append(word_)
        self.index2word = input_lang_vocab

        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def trim_max(self, trim_max_num,outlang_vocab):  # trim words below a certain count threshold
        keep_words = []
        orderList=list(self.word2count.values())  
        orderList.sort(reverse=True)  
        max_order_list=orderList[0:trim_max_num]
        min_count=max_order_list[-1]+1

        print("max 4000 words need words at least appear times:"+str(min_count))

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1
        self.index2word =outlang_vocab+["PAD", "NUM", "UNK"] + self.index2word
        input_lang_vocab=[]
        for word_ in self.index2word:
            if word_ not in input_lang_vocab:
                input_lang_vocab.append(word_)
        self.index2word = input_lang_vocab

        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

def generate_how_dict_vocab(lang):
    hownet_dict_vocab={}
    hownet_dict_all={}
    hownet_dict_category={}
    vocab_list=[]
    uselese_tag=["属性值","文字","属性","ProperName|专","surname|姓","部件","人","human|人","time|时间"]
    index_=0
    start = time.time()
    file1=open("..//hownet//hownet_dict_all").readlines()
    for x in file1:
        x_list=x.strip().split("###")
        word=x_list[0]
        if len(word) > 1:
            #print(word)
            #print(len(word))
            word_list=[]
            if len(x_list[1])!=0:
                for y in x_list[1].strip().split(" "):
                    y=y
                    if y not in uselese_tag and len(y)>0:
                        word_list.append(y)
            hownet_dict_all[word]=word_list
            vocab_list.append(word)
            for cate_ in word_list:
                if cate_ in hownet_dict_category:
                    category_list=hownet_dict_category[cate_]
                    if word not in category_list:
                        hownet_dict_category[cate_].append(word)
                else:
                    category_list=[]
                    category_list.append(word)
                    hownet_dict_category[cate_]=category_list
    #print(hownet_dict_all)
    print(hownet_dict_all["电话线"])
    start = time.time()
    count_all=0
    hownet_dict_tag={}
    for i in range(len(vocab_list)):
        word1=vocab_list[i]
        cate1=hownet_dict_all[word1]
        if len(cate1)==0:
            empty_list=[]
            hownet_dict_vocab[word1]=empty_list
        else:
            for word_ in cate1:
                if len(word_) >0 and len(word1)>0 and word1 !=None:

                    if word_ not in hownet_dict_tag:
                        empty_list=[]
                        hownet_dict_tag[word_]=empty_list
                    word_list=hownet_dict_tag[word_]
                    if word1 not in word_list:
                        word_list.append(word1)
                        hownet_dict_tag[word_]=word_list

            connect_word=[]
            for j in range(len(vocab_list)):
                if i!=j:
                    word2=vocab_list[j]
                    cate2=hownet_dict_all[word2]
                    flag=0
                    for word_ in cate1:
                        if word_ in cate2:
                            #print(word_+"#"+word1+"#"+word2)
                            flag=1
                            break
                    if flag==1:
                        count_all+=1
                        connect_word.append(word2)
            hownet_dict_vocab[word1]=connect_word

    print("training time", time_since(time.time() - start))
    print(len(vocab_list))
    print(count_all)
    if os.path.exists("..//hownet//hownet_dict_vocab"):
        category_vocab=[]
        category_vocab.append("PAD")
        for word in hownet_dict_category:
            category_vocab.append(word)
    else:        
        output=open("..//hownet//hownet_dict_vocab","w")
        for word in hownet_dict_vocab:
            output.write(word+"###"+" ".join(hownet_dict_vocab[word])+"\n")
        output=open("..//hownet//hownet_dict_tag","w")
        for word in hownet_dict_tag:
            output.write(word+"###"+" ".join(hownet_dict_tag[word])+"\n")

        output=open("..//hownet//hownet_dict_category","w")
        output1=open("..//hownet//hownet_category_vocab","w")
        category_vocab=[]
        category_vocab.append("PAD")
        output1.write("PAD"+"\n")
        for word in hownet_dict_category:
            output.write(word+"###"+" ".join(hownet_dict_category[word])+"\n")
            category_vocab.append(word)
            output1.write(word+"\n")

    return hownet_dict_all,hownet_dict_category,category_vocab

def get_file_dict_vocab_by_file():
    file_dict_vocab={}
    file1=open("..//hownet//hownet_dict_vocab").readlines()
    print(file1[0])
    for x in file1:
        x_list=x.strip().split("###")
        word=x_list[0]
        word_list=[y for y in x_list[1].split(" ")]
        file_dict_vocab[word]=word_list
    return file_dict_vocab
def get_edge_matrix(hownet_dict_vocab,input_list):
    input_edge=[]
    for i in range(len(input_list)):
        temp_list=[]
        for j in range(len(input_list)):
            temp_list.append(0)
        input_edge.append(temp_list)
    for i in range(len(input_list)):
        word1 = input_list[i]
        input_edge[i][i]=1
        #if i>0:
        #    input_edge[i][i-1]=1
        #if i<len(input_list)-1:
        #    input_edge[i][i+1]=1
        if word1 in hownet_dict_vocab:
            cate1 = hownet_dict_vocab[word1]
            if len(cate1) >0:
                for j in range(len(input_list)):
                    word2= input_list[j]
                    #if word2 in word1
                    if word2==word1 and len(word1)>3 and word1!="NUM":
                        input_edge[i][j]=1
                        input_edge[j][i]=1
                    '''
                    elif word2 in word1 and len(word1)>3 and word1!="NUM":
                        input_edge[i][j]=1
                        input_edge[j][i]=1
                    elif word1 in word2 and len(word2)>3 and word2!="NUM":
                        input_edge[i][j]=1
                        input_edge[j][i]=1

                    if word2 in cate1:
                        input_edge[i][j]=1
                        input_edge[j][i]=1
                    '''
    return input_edge

def get_middle_exp(output_list):
    operator=["+", "-","*", "/", "^"]
    middle_exp=[]
    for exp in output_list:
        if exp in operator:
            list_exp=[exp]
        else:
            list_exp=[exp,exp,exp]
        for i in range(len(middle_exp)-1,-1,-1):
            curr_list=middle_exp[i]
            if curr_list[0] in operator:
                if len(curr_list)<3:
                    middle_exp[i].append(exp)
                    break
        middle_exp.append(list_exp)

    assert len(middle_exp) == len(output_list)
    return middle_exp
def indexes_from_middle_output(lang, sentence, tree=False):
    res = []
    for word_list in sentence:
        temp_res=[]
        if len(word_list)==2:
            word_list.append(word_list[-1])

        if len(word_list)==1:
            word_list.append(word_list[0])
            word_list.append(word_list[0])
        for word in word_list:
            if word in lang.word2index:
                temp_res.append(lang.word2index[word])
            else:
                temp_res.append(5)
        if len(word_list)!=3:
            print("******************************")
            print(sentence)
            print("******************************")    
        res.append(temp_res)
    return res

def get_category_list(hownet_dict_all,hownet_dict_category,category_vocab,input_list):
    category_name_list=[] #cate_name
    category_index_list=[]#cate_index
    category_match_list=[]#cate_match_pos
    category_match_word_list=[]

    punc_list=[",","：","；","？","！","，","“","”",",",".","?","，","。","？","．","；","｡"]
    for i in range(len(input_list)):
        for j in range(len(input_list)):
            word1 = input_list[i]
            word2 = input_list[j]
            if word1 != word2:
                if word1 in hownet_dict_all and word2 in hownet_dict_all and word1 not in punc_list and word1!="NUM" and word2 not in punc_list and word2!="NUM":
                    cate1 = hownet_dict_all[word1]
                    cate2 = hownet_dict_all[word2]
                    if len(cate1) >0 and len(cate2) >0:
                        for cate_ in cate1:
                            if cate_ in cate2 and cate_ not in category_name_list:
                                category_name_list.append(cate_)
                                category_index_list.append(category_vocab.index(cate_))
                                match_temp=[]
                                match_word=""
                                category_word_temp=hownet_dict_category[cate_]
                                for k in range(len(input_list)):
                                    if input_list[k] in category_word_temp:
                                        match_temp.append(k)
                                        match_word+=input_list[k]+" "
                                category_match_list.append(match_temp)
                                category_match_word_list.append(match_word)
    return category_name_list,category_index_list,category_match_list,category_match_word_list

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res
def load_dense_drop_repeat(lang,path):
    vocab=lang.index2word
    vocab_size=len(vocab)
    in_vocab_num=0
    with codecs.open(path, "r", "utf-8") as f:
        first_line = True
        #print(len(f))
        for line in f:
            #print(line)
            if first_line:
                first_line = False
                size = int(line.rstrip().split()[1])
                print(size)
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            if vec[0] in vocab:
                count=vocab.index(vec[0])
                try:
                    list_vec=np.array([float(x) for x in vec[-300:]])
                    matrix[count, :] = list_vec
                    in_vocab_num+=1
                except:
                    print("****************************")
                    print("UnicodeEncodeError: 'decimal' codec can't encode character in position 0: invalid decimal Unicode string")
                    print(line)

    with open("data//pre_word_embeddings",'w') as wf2: #打开文件  
        for item in range(len(vocab)):  
            vec=[str(x) for x in matrix[item]] 
            wf2.write(vocab[item]+" "+" ".join(vec)+"\n")
    return matrix, size, in_vocab_num

def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)
def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    max_prob_len=0
    max_exp_len=0
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    if dataset=="APE":
        input_lang.trim_max(4000,output_lang.index2word)
    else:
        input_lang.build_input_lang(trim_min_count,output_lang.index2word)
    
    category_vocab=[]
    hownet_dict_vocab={}
    if USE_KAS2T_encoder==True:
        hownet_dict_all,hownet_dict_category,category_vocab=generate_how_dict_vocab(input_lang)
        hownet_dict_vocab=get_file_dict_vocab_by_file()
    
    start = time.time()
    matrix=[]
    if USE_Glove_embedding==True:
        matrix, size, in_vocab_num=load_dense_drop_repeat(input_lang,"..//..//Solve_1101//data//sgns.baidubaike.bigram-char")
        print("word_vector size:,vocab_size:,in_vocab_size:")
        print(size)
        print(len(input_lang.index2word))
        print(in_vocab_num)

    print("training time", time_since(time.time() - start))
    print("--------------------------------")

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)
            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])

        if USE_Seq2Seq==True:
            output_cell=indexes_from_sentence(output_lang, pair[1])
        else:
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        if USE_KAS2T_encoder==False:
            middle_exp_cell=[]
            input_edge=[]
            category_index_list=[]
            category_match_list=[]
            train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))
        else:
            middle_exp=get_middle_exp(pair[1])
            middle_exp_cell= indexes_from_middle_output(output_lang, middle_exp, tree)
            input_edge=get_edge_matrix(hownet_dict_vocab,pair[0])
            category_name_list,category_index_list,category_match_list,category_match_word_list=get_category_list(hownet_dict_all,hownet_dict_category,category_vocab,pair[0])
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))


    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    print(output_lang.index2word)
    print("max problem length is %d, max expression length is %d." %(max_prob_len,max_exp_len))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        if USE_Seq2Seq==True:
            output_cell=indexes_from_sentence(output_lang, pair[1])
        else:
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        if USE_KAS2T_encoder==False:
            middle_exp_cell=[]
            input_edge=[]
            category_index_list=[]
            category_match_list=[]
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))
        else:
            middle_exp=get_middle_exp(pair[1])
            middle_exp_cell= indexes_from_middle_output(output_lang, middle_exp, tree)
            input_edge=get_edge_matrix(hownet_dict_vocab,pair[0])
            category_name_list,category_index_list,category_match_list,category_match_word_list=get_category_list(hownet_dict_all,hownet_dict_category,category_vocab,pair[0])
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))

    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs,matrix,category_vocab,hownet_dict_vocab

def prepare_valid_data(input_lang, output_lang,pairs_tested, tree=False):
    test_pairs=[]
    if USE_KAS2T_encoder==True:
        hownet_dict_all,hownet_dict_category,category_vocab=generate_how_dict_vocab(input_lang)
        hownet_dict_vocab=get_file_dict_vocab_by_file()
    
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        if USE_Seq2Seq==True:
            output_cell=indexes_from_sentence(output_lang, pair[1])
        else:
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        if USE_KAS2T_encoder==False:
            middle_exp_cell=[]
            input_edge=[]
            category_index_list=[]
            category_match_list=[]
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))
        else:
            middle_exp=get_middle_exp(pair[1])
            middle_exp_cell= indexes_from_middle_output(output_lang, middle_exp, tree)
            input_edge=get_edge_matrix(hownet_dict_vocab,pair[0])
            category_name_list,category_index_list,category_match_list,category_match_word_list=get_category_list(hownet_dict_all,hownet_dict_category,category_vocab,pair[0])
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))

    print('Number of testind data %d' % (len(test_pairs)))
    return test_pairs


# Pad a with the PAD symbol
PAD_token=0
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq
# prepare the batches
def pad_middle_exp(seq, seq_len, max_length):
    for _ in range(max_length - seq_len):
        pad_list=[PAD_token,PAD_token,PAD_token]
        seq.append(pad_list)
    return seq
def pad_input_edge(input_edge, seq_len, max_length):
    for i in range(len(input_edge)):
        input_edge[i]+=[PAD_token for _ in range(max_length-seq_len)]
    for i in range(max_length-seq_len):
        temp_list=[PAD_token for _ in range(max_length)]
        input_edge.append(temp_list)
    return input_edge

def prepare_train_batch(pairs_to_batch, batch_size,PAD_token):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    input_edge_batches = []
    rule3_list_batches=[]
    unit_list_batches=[]
    output_middle_batches=[]
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        if USE_KAS2T_encoder==False:
            for _, i, _, j, _, _, _,_,_,_,_ in batch:
                input_length.append(i)
                output_length.append(j)
        else:
            for _, i, _, j, _, _, _,_,_,_,_ in batch:
                input_length.append(i)
                output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        input_edge_batch = []
        unit_list_batch=[]
        rule3_list_batch=[]
        output_middle_batch=[]
        if USE_KAS2T_encoder==False:
            for i, li, j, lj, num, num_pos, num_stack,_,_,_,_ in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(i, li, input_len_max))
                output_batch.append(pad_seq(j, lj, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                empty_list=[]
                unit_list_batch.append(empty_list)
                input_edge_batch.append(empty_list)
                #input_edge_batch.append(input_edge)
                rule3_list_batch.append(empty_list)
                output_middle_batch.append(empty_list)
        else:
            for i, li, j, lj, num, num_pos, num_stack,unit_list,rule3_list,middle_exp_cell,input_edge in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(i, li, input_len_max))
                output_batch.append(pad_seq(j, lj, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                unit_list_batch.append(unit_list)
                input_edge_batch.append(pad_input_edge(input_edge, li, input_len_max))
                #input_edge_batch.append(input_edge)
                rule3_list_batch.append(rule3_list)
                output_middle_batch.append(pad_middle_exp(middle_exp_cell,lj,output_len_max))

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        if USE_KAS2T_encoder==True:
            input_edge_batches.append(input_edge_batch)
            unit_list_batches.append(unit_list_batch)
            rule3_list_batches.append(rule3_list_batch)
            output_middle_batches.append(output_middle_batch)
        else:
            empty_list=[]
            unit_list_batches.append(empty_list)
            input_edge_batches.append(empty_list)
            #input_edge_batch.append(input_edge)
            rule3_list_batches.append(empty_list)
            output_middle_batches.append(empty_list)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches,unit_list_batches,rule3_list_batches,output_middle_batches,input_edge_batches


def indexes_to_sentence(lang, index_list, tree=False):
    res = []
    for index in index_list:
        if index < lang.n_words:
            res.append(lang.index2word[index])
    return res
