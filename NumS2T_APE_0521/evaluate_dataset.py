# coding: utf-8
from __future__ import division 
from pre_data import *
from masked_cross_entropy import *
from expressions_transfer import *
from models import *

import time
import torch.optim
from torch.optim import lr_scheduler

USE_CUDA = torch.cuda.is_available()

batch_size = 64
embedding_size = 300
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 3
n_layers = 2



train_data = load_Math23K_data('data/Math_23K.json')
print(len(train_data))

pairs, generate_nums, copy_nums = transfer_num(train_data)
print(len(pairs))


valid_data = load_data('data/ape/valid.ape.json',1)
train_data = load_data('data/ape/train.ape.json',1)
test_data = load_data('data/ape/test.ape.json',1)
print(len(train_data))
print(len(valid_data))
print(len(test_data))
print(len(train_data)+len(valid_data)+len(test_data))

pairs, generate_nums, copy_nums = transfer_num(train_data)
print(len(pairs))
#n_epochs = 50