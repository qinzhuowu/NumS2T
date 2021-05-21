from __future__ import division 
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import torch.nn.functional as F
import numpy as np
import copy
import math
byte_flag=0
if "17NLG" in os.getcwd():
    byte_flag=1
USE_CUDA = torch.cuda.is_available()
print_dims = False

class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            if byte_flag==1:
                num_mask=num_mask.byte()
            else:
                num_mask=num_mask.bool()
            score = score.masked_fill_(num_mask, -1e12)
        return score

class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            if byte_flag==1:
                seq_mask=seq_mask.byte()
            else:
                seq_mask=seq_mask.bool()
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)

class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,category_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.shrink_size=hidden_size-embedding_size-embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.category_embedding = nn.Embedding(category_size, int(self.shrink_size/2), padding_idx=0)
        #self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.gat_1 = GATLayer(self.shrink_size, self.shrink_size, 0.5, 0.1,concat=False )
        self.gat_dense=nn.Linear(self.shrink_size*2,self.shrink_size)


        self.gru_pade = nn.GRU(embedding_size, int(self.shrink_size/2), n_layers, dropout=dropout, bidirectional=True)

        self.gat_n = [GraphAttentionLayer(int(self.shrink_size/2), int(self.shrink_size/2), 0.5, 0.1,concat=False ) for _ in range(8)]
        for i, attention in enumerate(self.gat_n):
            self.add_module('attention_{}'.format(i), attention)
        
        self.pos_embedding = PositionalEncoding(int(self.shrink_size/2), 160)
        self.encoder_layers =EncoderLayer(self.shrink_size, 16, self.shrink_size, dropout)
        self.encoder_layers2 =EncoderLayer(self.shrink_size, 4, self.shrink_size, dropout)

        
        self.enc_linear=nn.Linear(self.shrink_size, self.shrink_size)
        self.enc_dense=nn.Linear(self.shrink_size, 1)
        self.num_pos_embedding = nn.Embedding(2, embedding_size, padding_idx=0)


        self.type_linear=nn.Linear(self.shrink_size, self.shrink_size)
        self.type_dense=nn.Linear(self.shrink_size, 5)
        self.type_pos_embedding = nn.Embedding(5, embedding_size, padding_idx=0)
        
        self.nums_char_embedding=nn.Embedding(16,embedding_size,padding_idx=0)
        self.pade_embedding = nn.Embedding(2, embedding_size, padding_idx=0)
        self.num_gru=nn.GRU(embedding_size, embedding_size, n_layers, dropout=dropout, bidirectional=True)

        self.self_attn=SelfAttn(embedding_size)
        self.pair_score=nn.Linear(self.hidden_size,1)

    def forward(self, input_seqs, input_lengths,cate_word_edge,cate_index_input,cate_length,cate_id_match,nums_index,num_pos, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)


        #problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :int(self.shrink_size/2)] + pade_outputs[:, :,int(self.shrink_size/2):]  # S x B x H
        #problem_output=F.max_pool1d(pade_outputs.permute(1,2,0), pade_outputs.shape[0]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)

        #encoder_outputs_knowledge=self.gat_1(pade_outputs.transpose(0,1),input_edge_batch)#B*S*H
        #concat_pade_outputs=self.gat_dense(torch.cat((pade_outputs,encoder_outputs_knowledge.transpose(0,1)),2))
        #return concat_pade_outputs, problem_output
        
        ##category_embedded=self.category_embedding(cate_index_input)#B*cate_num*hidden
        #cate_length=[]#B cate_id_match=[]#B*C*[]
        max_cate_len=max(cate_length)
        

        padding_hidden=self.category_embedding(torch.LongTensor([0]).cuda()).squeeze(0)
        max_cate_len = max(cate_length)
        if max_cate_len>0:
            category_embedded_temp=[]
            for idx in range(len(cate_length)):
                idx_cate_len=cate_length[idx]
                if idx_cate_len == 0:
                    category_embedded_temp.append(torch.stack([padding_hidden for cate_idx in range(max_cate_len)],dim=0))#C*H
                else:
                    temp_hidden_category=[]
                    for i in range(idx_cate_len):
                        temp_hidden=[]
                        for j in cate_id_match[idx][i]:
                            temp_hidden.append(pade_outputs[j][idx])#cate*hidden
                        gather_hidden=torch.stack(temp_hidden,dim=0).mean(0)#hidden
                        temp_hidden_category.append(gather_hidden)
                    for i in range(idx_cate_len,max_cate_len):
                        temp_hidden_category.append(padding_hidden)
                    category_embedded_temp.append(torch.stack(temp_hidden_category,0))#C*hidden
            category_embedded = torch.stack(category_embedded_temp,0).detach()#B*S*H

            concat_input_category=torch.cat((pade_outputs.transpose(0,1),category_embedded),1)#B*S+cate_num*hidden
        else:
            concat_input_category=pade_outputs.transpose(0,1)
        
        input_sequence=input_seqs.transpose(0,1)

        encoder_outputs_knowledge = torch.stack([att(concat_input_category,cate_word_edge) for att in self.gat_n], dim=2)#B*S*N*head
        encoder_outputs_knowledge=encoder_outputs_knowledge.mean(2)#B*S*N
        
        #pos_embed= self.pos_embedding(input_lengths,max_cate_len)#B[B,S,H]
        pos_embed= self.pos_embedding(input_lengths,max_cate_len)#B[B,S,H]
        output=torch.cat((encoder_outputs_knowledge,pos_embed),2)#B[B,S,H]
        
        src_mask = (input_sequence != 0).unsqueeze(-2)
        if max_cate_len > 0:
            cat_mask = (cate_index_input != 0).unsqueeze(-2) #B*1*C
            src_cat_mask=torch.cat((src_mask,cat_mask),2)
        else:
            src_cat_mask=src_mask
        output, attention = self.encoder_layers(output, src_cat_mask)#[B,S,H] B*S*S
        

        ##############################################################################
        output=output[:,:input_seqs.size(0),:]#B*S*H
        #problem_output=F.max_pool1d(pade_outputs.permute(1,2,0), pade_outputs.shape[0]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)
        problem_output=F.max_pool1d(output.permute(0,2,1).contiguous(), output.shape[1]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)

        
        gate_hidden= self.enc_linear(output)#B*S*H
        gate_num=torch.sigmoid(self.enc_dense(gate_hidden).squeeze(2))#B*S
        zeros = torch.zeros_like(gate_num)
        ones= torch.ones_like(gate_num)
        num_indicator = torch.where(gate_num > 0.5, ones, zeros)#B*S
        num_indicator_id=num_indicator.long()
        num_pos_embedded=self.num_pos_embedding(num_indicator_id)#B*S*E
        
        #gate_num=None
        #num_indicator=None

        
        type_hidden= self.type_linear(output)#B*S*H
        type_num=torch.sigmoid(self.type_dense(type_hidden))#B*S*5
        _,type_num_indicator = torch.max(type_num,dim=2)#B*S
        type_num_indicator_id=type_num_indicator.long()
        type_num_pos_embedded=self.type_pos_embedding(type_num_indicator_id)#B*S*E
        
        #type_num=None

        batch_size=len(input_lengths)
        seq_len=input_seqs.size(0)
        nums_hidden=[]
        for b in range(batch_size):
            nums_prob_hidden=[]
            for num_i in range(len(nums_index[b])):
                nums_char_list=nums_index[b][num_i]
                nums_char_embedded=self.nums_char_embedding(torch.LongTensor(nums_char_list).cuda()).unsqueeze(1) #num_len*1*E
                #nums_dense_hidden=self.num_dense(nums_char_embedded)
                nums_gru_out,nums_gru_hidden=self.num_gru(nums_char_embedded)#num_len*1*E,1*1*E
                nums_gru_final=nums_gru_out[-1, :, :self.embedding_size] + nums_gru_out[0, :, self.embedding_size:]#1*E
                nums_prob_hidden.append(nums_gru_final.squeeze(0))##E
            nums_hidden.append(nums_prob_hidden)

        pade_hidden=self.pade_embedding(torch.LongTensor([0]).cuda()).squeeze(0)#E
        all_problem_hidden=[]
        problem_num_mask=[]
        for b in range(batch_size):
            single_problem_hidden=[]
            single_num_mask=[]
            for num_i in range(seq_len):
                if num_i in num_pos[b]:
                    num_index=num_pos[b].index(num_i)
                    single_problem_hidden.append(nums_hidden[b][num_index])
                    single_num_mask.append(1)
                else:
                    single_num_mask.append(0)
                    single_problem_hidden.append(embedded[num_i,b,:])
            single_problem_hidden=torch.stack(single_problem_hidden,0)#S*H
            all_problem_hidden.append(single_problem_hidden)
            problem_num_mask.append(single_num_mask)
        all_problem_hidden=torch.stack(all_problem_hidden,1)#S*B*E
        problem_num_mask=torch.FloatTensor(problem_num_mask).cuda()#B*S
        
        num_self_attn=self.self_attn(all_problem_hidden,problem_num_mask)#B*S*S
        attn_problem_hidden=num_self_attn.bmm(all_problem_hidden.transpose(0,1))#B*S*E
        pade_outputs_cat= torch.cat((output.transpose(0,1),all_problem_hidden,attn_problem_hidden.transpose(0,1)),2)#S*B*H
        problem_output_cat= torch.cat((problem_output,all_problem_hidden[-1,:,:],attn_problem_hidden[:,-1,:]),1)#B*H

        num_pair_score=self.pair_score(pade_outputs_cat.transpose(0,1)).squeeze(2)#B*S


        return pade_outputs_cat, problem_output_cat,gate_num,num_indicator,problem_output,type_num,num_pair_score
class SelfAttn(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, all_problem_hidden,problem_num_mask ):
        #all_problem_hidden S*B*H  problem_num_mask B*S
        batch_size=all_problem_hidden.size(1)
        max_len = all_problem_hidden.size(0)

        repeat_dims1=[1,1,max_len,1]
        repeat_dims2=[1,max_len,1,1]
        sen1=all_problem_hidden.transpose(0,1).unsqueeze(2)#B*S*1*H
        sen2=all_problem_hidden.transpose(0,1).unsqueeze(1)#B*1*S*H

        sen1=sen1.repeat(repeat_dims1)
        sen2=sen2.repeat(repeat_dims2)#S*S*B*H

        energy_in=torch.cat((sen1, sen2), 3)#B*S*S*2H
        score_feature = torch.tanh(self.attn(energy_in))#B*S*S*H
        attn_energies = self.score(score_feature).squeeze(3)  # B*S*S
        '''
        num_dims1=[1,1,max_len]
        num_dims2=[1,max_len,1]
        num_mask1=problem_num_mask.unsqueeze(2)#B*S*1
        num_mask2=problem_num_mask.unsqueeze(1)#B*1*S
        #num2num=num_mask1.bmm(num_mask2)#B*S*S
        #word2num=num_mask2.repeat(num_dims2)
        #num2word=num_mask1.repeat(num_dims1)
        num_mask=num_mask1.bmm(num_mask2)
        if byte_flag==1:
            num_mask=num_mask.byte()
        else:
            num_mask=num_mask.bool()
        attn_energies = attn_energies.masked_fill_(num_mask, -1e12)
        '''
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S*S
        return attn_energies


class EncoderAPESeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,pretrain_emb, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.pos_embed_size=128
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.shrink_size=hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrain_emb))
        #self.embedding.weight.requires_grad = False
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, self.shrink_size, n_layers, dropout=dropout, bidirectional=True)

        self.dense_emb=nn.Linear(self.embedding_size, self.pos_embed_size)

        self.enc_linear=nn.Linear(self.shrink_size, self.shrink_size)
        self.enc_dense=nn.Linear(self.shrink_size, 1)
        self.num_pos_embedding = nn.Embedding(2, self.pos_embed_size, padding_idx=0)
        
        self.type_linear=nn.Linear(self.shrink_size, self.shrink_size)
        self.type_dense=nn.Linear(self.shrink_size, 5)
        self.type_pos_embedding = nn.Embedding(5, self.pos_embed_size, padding_idx=0)

        self.nums_char_embedding=nn.Embedding(16,self.pos_embed_size,padding_idx=0)
        self.pade_embedding = nn.Embedding(2, self.pos_embed_size, padding_idx=0)
        self.num_gru=nn.GRU(self.pos_embed_size, self.pos_embed_size, n_layers, dropout=dropout, bidirectional=True)

        self.self_attn=SelfAttn(self.pos_embed_size)
        self.pair_score=nn.Linear(self.hidden_size,1)
    def forward(self, input_seqs, input_lengths,nums_index,num_pos, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.shrink_size] + pade_outputs[0, :, self.shrink_size:]
        pade_outputs = pade_outputs[:, :, :self.shrink_size] + pade_outputs[:, :, self.shrink_size:]  # S x B x H
        
        gate_hidden= self.enc_linear(pade_outputs.transpose(0,1))#B*S*H
        gate_num=torch.sigmoid(self.enc_dense(gate_hidden).squeeze(2))#B*S
        zeros = torch.zeros_like(gate_num)
        ones= torch.ones_like(gate_num)
        num_indicator = torch.where(gate_num > 0.5, ones, zeros)#B*S
        num_indicator_id=num_indicator.long()
        num_pos_embedded=self.num_pos_embedding(num_indicator_id)#B*S*E

        
        
        #gate_num=None
        #num_indicator=None
        
        type_hidden= self.type_linear(pade_outputs.transpose(0,1))#B*S*H
        type_num=torch.sigmoid(self.type_dense(type_hidden))#B*S*5
        _,type_num_indicator = torch.max(type_num,dim=2)#B*S
        type_num_indicator_id=type_num_indicator.long()
        type_num_pos_embedded=self.type_pos_embedding(type_num_indicator_id)#B*S*E
        
        #type_num=None

        batch_size=len(input_lengths)
        seq_len=input_seqs.size(0)
        nums_hidden=[]
        for b in range(batch_size):
            nums_prob_hidden=[]
            for num_i in range(len(nums_index[b])):
                nums_char_list=nums_index[b][num_i]
                nums_char_embedded=self.nums_char_embedding(torch.LongTensor(nums_char_list).cuda()).unsqueeze(1) #num_len*1*E
                #nums_dense_hidden=self.num_dense(nums_char_embedded)
                nums_gru_out,nums_gru_hidden=self.num_gru(nums_char_embedded)#num_len*1*E,1*1*E
                nums_gru_final=nums_gru_out[-1, :, :self.pos_embed_size] + nums_gru_out[0, :, self.pos_embed_size:]#1*E
                nums_prob_hidden.append(nums_gru_final.squeeze(0))##E
            nums_hidden.append(nums_prob_hidden)

        dense_emb=self.dense_emb(embedded)
        pade_hidden=self.pade_embedding(torch.LongTensor([0]).cuda()).squeeze(0)#E
        all_problem_hidden=[]
        problem_num_mask=[]
        for b in range(batch_size):
            single_problem_hidden=[]
            single_num_mask=[]
            for num_i in range(seq_len):
                if num_i in num_pos[b]:
                    num_index=num_pos[b].index(num_i)
                    single_problem_hidden.append(nums_hidden[b][num_index])
                    single_num_mask.append(1)
                else:
                    single_num_mask.append(0)
                    single_problem_hidden.append(dense_emb[num_i,b,:])
            single_problem_hidden=torch.stack(single_problem_hidden,0)#S*H
            all_problem_hidden.append(single_problem_hidden)
            problem_num_mask.append(single_num_mask)
        all_problem_hidden=torch.stack(all_problem_hidden,1)#S*B*E
        problem_num_mask=torch.FloatTensor(problem_num_mask).cuda()#B*S
        
        num_self_attn=self.self_attn(all_problem_hidden,problem_num_mask)#B*S*S
        attn_problem_hidden=num_self_attn.bmm(all_problem_hidden.transpose(0,1))#B*S*E
        
        #pade_outputs_cat= torch.cat((pade_outputs,all_problem_hidden,attn_problem_hidden.transpose(0,1)),2)#S*B*H
        #problem_output_cat= torch.cat((problem_output,all_problem_hidden[-1,:,:],attn_problem_hidden[:,-1,:]),1)#B*H
        pade_outputs_cat=pade_outputs
        problem_output_cat=problem_output
        num_pair_score=self.pair_score(pade_outputs_cat.transpose(0,1)).squeeze(2)#B*S
        return pade_outputs_cat, problem_output_cat,gate_num,num_indicator,problem_output,type_num,num_pair_score


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)
        self.concat_encoder_outputs = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums,indices,masked_index):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        
        #encoder_outputs_knowledge=input_edge_batch.bmm(encoder_outputs.transpose(0, 1)) # B x S*S  B x S x H B x S x H
        #concat_encoder_outputs=torch.cat((encoder_outputs, encoder_outputs_knowledge.transpose(0,1)), dim=2)
        #current_attn = self.attn(current_embeddings.transpose(0, 1), concat_encoder_outputs, seq_mask) # B x S
        #current_context = current_attn.bmm(concat_encoder_outputs.transpose(0, 1))  #B x S S*B*N  B x 1 x N
        #current_context=self.concat_encoder_outputs(current_context)
        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight
class Pivot(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding
    def __init__(self, hidden_size, dropout=0.5):
        super(Pivot, self).__init__()
        self.hidden_size = hidden_size
        
        self.dec_linear=nn.Linear(hidden_size*2,hidden_size)
        self.pivot_score= nn.Linear(hidden_size, 1)

        self.gru = nn.GRU(hidden_size, hidden_size, 2, dropout=dropout, bidirectional=True)
        
    def forward(self, encoder_outputs,dec_states):
        #encoder_outputs S*B*H  dec_states B*S*2H
        enc_len=encoder_outputs.size(0)
        dec_outputs=self.dec_linear(dec_states).transpose(0,1)#S*B*H
        gru_input=torch.cat((encoder_outputs,dec_outputs),0)#S1+S2*B*H

        pade_hidden = None
        pade_outputs, pade_hidden = self.gru(gru_input, pade_hidden)#S*B*H
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        gru_output=pade_outputs[:enc_len,:,:].transpose(0,1)#B*S*H
        binary_num_pivot=torch.sigmoid(self.pivot_score(gru_output).squeeze(2))#B*S

        return binary_num_pivot

class Sentence_level_semantics(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding
    def __init__(self, hidden_size,embedding_size, dropout=0.5):
        super(Sentence_level_semantics, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.shrink_size=hidden_size-embedding_size-embedding_size-embedding_size

        self.attn=nn.Linear(hidden_size+self.shrink_size,hidden_size)
        self.attn1=nn.Linear(hidden_size+self.shrink_size,hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.score1 = nn.Linear(hidden_size, 1)
        
        
    def forward(self, problem_output,current_context):
        #problem_output B*H1  current_context B*1*H2
        batch=current_context.size(0)
        repeat_dims=[batch,1,1]
        repeat_dims2=[1,batch,1]
        
        enc_sen=problem_output.unsqueeze(1)#B*1*H1
        dec_sen=current_context.squeeze(1).unsqueeze(0)#1*B*H2
        enc_sen=enc_sen.repeat(*repeat_dims2)  # B x B x H1
        dec_sen=dec_sen.repeat(*repeat_dims)  # B x B x H2

        energy_in=torch.cat((enc_sen, dec_sen), 2)#B*B*2H
        score_feature = torch.tanh(self.attn(energy_in))#B*B*H
        attn_energies = self.score(score_feature).squeeze(2)  # B*B
        #attn_energies = self.score(energy_in).squeeze(2)  # B*B
        attn_score = nn.functional.softmax(attn_energies, dim=1)  # B x B

        enc_sen1=problem_output.unsqueeze(0)#1*B*H
        dec_sen1=current_context #B*1*H
        enc_sen1=enc_sen1.repeat(*repeat_dims)  # B x B x H
        dec_sen1=dec_sen1.repeat(*repeat_dims2)  # B x B x H

        energy_in1=torch.cat((enc_sen1, dec_sen1), 2)#B*B*2H
        score_feature1 = torch.tanh(self.attn(energy_in1))#B*B*H
        attn_energies1 = self.score1(score_feature1).squeeze(2)  # B*B
        #attn_energies1 = self.score1(energy_in1).squeeze(2)  # B*B
        attn_score1 = nn.functional.softmax(attn_energies1, dim=1)  # B x B

        return attn_score,attn_score1

class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, graph_input, adj):
        #[B*S*H] [B*S*S]
        h = self.W(graph_input)
        # [batch_size, N, out_features]
        batch_size, N,  _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N) #B*S*N - B*S*1 - B*S*S B*S*S
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2) ##B*S*S
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, graph_input)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """ d_model hidden 512  max_seq_len largedt len
        """
        super(PositionalEncoding, self).__init__()
        # PE matrix
        position_encoding = np.array([
          [pos / pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]  for pos in range(max_seq_len)])
        # odd line use sin,even line use cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # first line is all 0 as PAD positional encoding
        # word embedding add UNK as word embedding
        # use PAD to represent PAD position
        position_encoding=torch.FloatTensor(position_encoding)
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding),0)
        
        # +1 because adding PAD
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,requires_grad=False)
    def forward(self, input_len,category_num):
        """input_len  [BATCH_SIZE]
        """
        max_len = max(input_len)+category_num
        #tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # pad position add 0  range start by 1 to avoid pad(0)
        input_pos = torch.cuda.LongTensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product self attention
        query: (batch_size, h, seq_len, d_k), seq_len can be either src_seq_len or tgt_seq_len
        key: (batch_size, h, seq_len, d_k), seq_len in key, value and mask are the same
        value: (batch_size, h, seq_len, d_k)
        mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, tgt_seq_len, tgt_seq_len) (legacy)
    """
    if print_dims:
        print("{0}: query: type: {1}, shape: {2}".format("attention func", query.type(), query.shape))
        print("{0}: key: type: {1}, shape: {2}".format("attention func", key.type(), key.shape))
        print("{0}: value: type: {1}, shape: {2}".format("attention func", value.type(), value.shape))
        print("{0}: mask: type: {1}, shape: {2}".format("attention func", mask.type(), mask.shape))
    d_k = query.size(-1)

    # scores: (batch_size, h, seq_len, seq_len) for self_attn, (batch_size, h, tgt_seq_len, src_seq_len) for src_attn
    scores = torch.matmul(query, key.transpose(-2, -1)/math.sqrt(d_k)) #B,H,S,S
    # print(query.shape, key.shape, mask.shape, scores.shape)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)##B,H,S,S
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0
        self.dim_per_head = model_dim//num_heads
        self.h = num_heads
        #self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(model_dim, model_dim)) for i in range(4)])
        self.key_dim=12
        self.value_dim=32
        self.linear_k = nn.Linear(model_dim, self.key_dim * num_heads)
        self.linear_v = nn.Linear(model_dim, self.value_dim * num_heads)
        self.linear_q = nn.Linear(model_dim, self.key_dim * num_heads)

        self.linear_x=nn.Linear(self.value_dim * num_heads,model_dim)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))
        
    def forward(self, query, key, value, mask=None):
        """
            query: (batch_size, seq_len, d_model), seq_len can be either src_seq_len or tgt_seq_len
            key: (batch_size, seq_len, d_model), seq_len in key, value and mask are the same
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, tgt_seq_len, tgt_seq_len) (legacy)
        """
        if print_dims:
            print("{0}: query: type: {1}, shape: {2}".format(self.__class__.__name__, query.type(), query.shape))
            print("{0}: key: type: {1}, shape: {2}".format(self.__class__.__name__, key.type(), key.shape))
            print("{0}: value: type: {1}, shape: {2}".format(self.__class__.__name__, value.type(), value.shape))
            print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))
        if mask is not None:
            mask = mask.unsqueeze(1)#B,1,1,S
        nbatches = query.size(0)
        
        # 1) Do all linear projections in batch from d_model to (h, d_k)
        key = self.linear_k(key).view(nbatches, -1, self.h, self.key_dim).transpose(1,2)#B*S*(dim_per_head * num_heads)
        value = self.linear_v(value).view(nbatches, -1, self.h, self.value_dim).transpose(1,2)#B,H,S,dim
        query = self.linear_q(query).view(nbatches, -1, self.h, self.key_dim).transpose(1,2)
        #query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch
        x, p_attn = attention(query, key, value, mask=mask, dropout=self.dropout) # (batch_size, h, seq_len, d_k),#B,H,S,S
        if print_dims:
            print("{0}: x (after attention): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))

        # 3) Concatenate and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.value_dim)##B,S,H,dim
        x = self.linear_x(x) # (batch_size, seq_len, d_model)
        if print_dims:
            print("{0}: x (after concatenation and linear): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return x,p_attn

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return self.w2(self.dropout(F.relu(self.w1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(n_features))
        self.b_2 = nn.Parameter(torch.zeros(n_features))
        self.eps = eps
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean)/(std + self.eps) + self.b_2
class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.norm = LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attn_mask=None):
        """norm -> self_attn -> dropout -> add -> 
        norm -> feed_forward -> dropout -> add"""
        # self attention  ##[B,S,H] B*S*S
        norm_inputs=self.norm(inputs)
        context, attention = self.attention(norm_inputs, norm_inputs, norm_inputs, attn_mask)#[B,S,H] B*S*S
        context=self.dropout(context)
        context=inputs+context
        # feed forward network
        output=self.norm(context)
        output = self.feed_forward(output)#[B,S,H] #ff x+output
        output=self.dropout(output)
        output=context+output
        return output, attention


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.out_features=32
        self.fc= nn.Linear(in_features, self.out_features, bias=False)
        self.attn_fc= nn.Linear(2*self.out_features,1, bias=False)
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, graph_input, adj):
        #[B*S*H] [B*S*S]
        h=self.fc(graph_input)
        # [batch_size, N, out_features]
        batch_size, N, out_dim = h.size()#B,N,H
        a_input=torch.cat([h.unsqueeze(2).repeat(1,1,N,1),h.unsqueeze(1).repeat(1,N,1,1)],dim=3)#B,N,N,2*dim
        attention=self.leakyrelu(self.attn_fc(a_input).squeeze(3))#B,S,S
        attention = attention.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)#B,S,S
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, graph_input)##B,S,S*#B,S,H  #B,S,H  not use h,use original_input
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
