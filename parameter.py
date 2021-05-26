# coding: utf-8
import torch
USE_CUDA = torch.cuda.is_available()

batch_size = 64
embedding_size = 300
pos_embedding_size=128
hidden_size = 512
n_epochs = 160
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 3
n_layers = 2
Max_Question_len=120
Max_Expression_len=50
dataset="Math_23K"
#dataset="Math_23K""APE"

if dataset=="APE":
	n_epochs=80
else:
	n_epochs=160

#APE  word  彩电 是 两种 电视机
#APE  char 彩 电 是 两 种 电 视 机
USE_APE_word=False
USE_APE_char=True

USE_Glove_embedding=True
USE_KAS2T_encoder=True

#add our numeric encoder
USE_number_enc=True
USE_self_attn=True

if USE_number_enc==True:
	hidden_size=hidden_size+pos_embedding_size
if USE_self_attn==True:
	hidden_size=hidden_size+pos_embedding_size

#just symbol (baseline)
#USE_just_symbol=False
#just digit number(eg: 1 5 0)
USE_just_char_number=False

USE_Seq2Tree=True
USE_Seq2Seq=False

USE_gate=True
USE_cate=True
USE_compare=True

#GTS USE_KAS2T_encoder=False
#KAS2T USE_KAS2T_encoder=True
USE_G2T_stanford=True